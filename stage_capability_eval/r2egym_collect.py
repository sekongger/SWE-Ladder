"""Collect independent stage artifacts with the R2E-Gym XML scaffold.

Only public input fields enter the agent process. Private scoring is a separate
command. Journals are resumable only under the exact saved run configuration.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import multiprocessing
import os
import re
import shlex
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
R2E_GYM_SRC = ROOT / 'R2E-Gym-eval/src'
if str(R2E_GYM_SRC) not in sys.path:
    sys.path.insert(0, str(R2E_GYM_SRC))

try:
    from .package_common import public_task_row
    from .repro import _public_oracle_loc, artifact_sha256, validate_artifact
except ImportError:  # direct script execution
    sys.path.insert(0, str(ROOT))
    from stage_capability_eval.package_common import public_task_row
    from stage_capability_eval.repro import _public_oracle_loc, artifact_sha256, validate_artifact

CONFIG = ROOT / 'R2E-Gym-eval/src/r2egym/agenthub/config/r2egym/edit_non_fn_calling.yaml'
BUDGETS = {'loc': 30, 'repro_given_loc': 20, 'patch_given_loc': 20}
INSTRUCTIONS = {
    'loc': '''Localize the production code responsible for the issue. Inspect the buggy
repository without changing tracked files. Do not implement a fix. Submit a JSON
object in finish.result with ranked repository-relative files and qualified
function/class names. Provide at most 5 files and 10 function candidates:
{"files":["path/to/module.py"],"functions":[{"file":"path/to/module.py","function":"Class.method"}]}.
Rank the most likely locations first. Use names present in the buggy source.''',
    'repro_given_loc': '''Write a pytest reproduction of the issue using the shared location
hint below. Do not repair the issue or change any existing tracked file. Create
new test files in the repository. Tests must assert the correct expected behavior:
at least one test must fail with an assertion on the buggy source, and every test
must pass after a correct production fix. Avoid unconditional failures, source
text checks, version checks, or changing production source from the test.
Run your reproduction on the buggy source. Submit JSON in finish.result:
{"files":["test_repro.py"],"command":["python","-m","pytest","test_repro.py","-q"]}.
List every new file needed by the test. Use pytest with explicit file targets;
do not use shell commands, pytest collection-only mode, or a custom JUnit option.
The same frozen files will be evaluated on isolated buggy and fixed repositories.''',
    'patch_given_loc': '''Repair the issue using the shared location hint below. Modify
production source only. You may inspect and run existing tests. Do not modify
existing tests or include new tests/reproduction scripts in your submitted diff.
Remove temporary files before calling finish. The final repository diff is your
submission; it will be checked using private regression and issue tests.''',
}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def build_stage_task(row, stage):
    if stage not in BUDGETS:
        raise ValueError('unknown stage')
    task = public_task_row(row)
    prompt = ('Repository: /testbed\nEvaluation task: ' + stage.upper() + '\n\n'
              + INSTRUCTIONS[stage] + '\n\nGitHub issue:\n' + task['problem_statement'])
    if stage != 'loc':
        oracle = _public_oracle_loc(row['oracle_loc'])
        prompt += '\n\nShared location hint:\n' + json.dumps(oracle, ensure_ascii=False)
    task['problem_statement'] = prompt
    # DockerRuntime requires these two objects even when rewards are unused.
    # Supplying empty specs prevents its cached hidden eval script from being
    # installed as /run_tests.sh in the model-visible container.
    task['make_test_spec'] = json.dumps(dict(
        instance_id=task['instance_id'], repo=task['repo'], version=task['version'],
        repo_script_list=[], env_script_list=[], eval_script_list=['true'],
        arch='x86_64', FAIL_TO_PASS=[], PASS_TO_PASS=[], language='py',
        docker_specs={}, namespace=None))
    task['parsed_commit'] = json.dumps(dict(file_diffs=[], old_commit_hash=task['base_commit'],
        new_commit_hash=task['base_commit'], commit_message='', commit_date='2000-01-01T00:00:00'))
    return task


def parse_finish_json(text):
    if isinstance(text, dict):
        return text
    match = re.search(r'<parameter=result>(.*?)</parameter>', str(text), re.S)
    if match:
        text = match.group(1)
    text = str(text).strip()
    match = re.fullmatch(r'```(?:json)?\s*(.*?)\s*```', text, re.S)
    if match:
        text = match.group(1)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError('finish.result must be a JSON object')
    return value


def checked_run(run, command):
    output, code = run(command)
    if str(code) != '0':
        raise RuntimeError(f'container command failed ({code}): {command}: {output[:500]}')
    return output


def capture_repro(run, base_commit, submission):
    base = shlex.quote(base_commit)
    changed = checked_run(run, f'git diff --name-only --diff-filter=CDMRTUXB {base}').strip()
    if changed:
        raise ValueError('reproduction changed tracked files: ' + changed)
    declared = submission.get('files')
    if not isinstance(declared, list) or not declared or len(declared) > 32:
        raise ValueError('declare 1..32 new artifact files')
    # Agent.get_patch stages all additions, so untracked files alone are not enough.
    new_files = set(checked_run(run, f'git diff --name-only --diff-filter=A {base}').splitlines())
    new_files.update(checked_run(run, 'git ls-files --others --exclude-standard').splitlines())
    validate_artifact({'files': {p: '' for p in declared}, 'command': submission.get('command')})
    files = {}
    for path in declared:
        if path not in new_files or any(part.startswith('.') for part in Path(path).parts):
            raise ValueError('artifact must be a new non-hidden file: ' + path)
        quoted = shlex.quote(path)
        checked_run(run, f'test ! -L {quoted}')
        size = int(checked_run(run, f'wc -c < {quoted}').strip())
        if size > 1024 * 1024:
            raise ValueError('artifact file exceeds 1 MiB')
        files[path] = checked_run(run, f'cat -- {quoted}')
    return validate_artifact({'files': files, 'command': submission['command']})


def finish_result(trajectory):
    for step in reversed(trajectory.trajectory_steps):
        action = step.parsed_action
        if action.function_name == 'finish':
            return parse_finish_json(action.parameters.get('result', ''))
    raise ValueError('no structured finish submission')


def collect_task(row, options):
    started = time.monotonic()
    stage = options['stage']
    folder = Path(options['out']) / row['instance_id']
    folder.mkdir(parents=True, exist_ok=True)
    record = dict(instance_id=row['instance_id'], stage=stage, model_id=options['model'],
                  status='infra_error', failure_category=None, artifact_path=None)
    env = None
    with (folder / 'collector.log').open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            from r2egym.agenthub.agent.agent import Agent, AgentArgs
            from r2egym.agenthub.environment.env import RepoEnv, EnvArgs
            os.environ['OPENAI_API_BASE'] = options['api_base']
            os.environ.setdefault('OPENAI_API_KEY', 'EMPTY')
            for proxy_name in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
                os.environ.pop(proxy_name, None)
            os.environ['R2EGYM_DOCKER_HOST'] = options['docker_host']
            os.environ['R2EGYM_DOCKER_MODE'] = 'local'
            args = AgentArgs.from_yaml(Path(options['config']))
            args.llm_name = 'openai/' + options['model']
            args.instance_prompt = '{problem_statement}'
            args.other_args = {**(args.other_args or {}), 'max_output_tokens': 8192}
            task = build_stage_task(row, stage)
            (folder / 'public_task.json').write_text(json.dumps(task, indent=2))
            env = RepoEnv(EnvArgs(ds=task), backend='docker', use_lsp=False)
            actual_base = checked_run(env.runtime.run, 'git rev-parse HEAD').strip()
            if actual_base != row['base_commit']:
                raise RuntimeError(f'image HEAD mismatch: {actual_base} != {row["base_commit"]}')
            if checked_run(env.runtime.run, 'git status --porcelain').strip():
                raise RuntimeError('initial repository is not clean')
            agent = Agent(name='stage-' + stage, args=args)
            trajectory = agent.run(env, use_fn_calling=False, scaffold='r2egym',
                max_steps=options['max_steps'], max_steps_absolute=options['max_steps'],
                max_token_limit=81920, max_total_time=1800, max_llm_time=180,
                max_exec_time=90, temperature=0, enable_compression=False)
            (folder / 'trajectory.json').write_text(trajectory.model_dump_json(indent=2))
            record.update(exit_reason=trajectory.exit_reason, steps=len(trajectory.trajectory_steps),
                completion_tokens=sum(s.token_usage_completion for s in trajectory.trajectory_steps),
                prompt_tokens=sum(s.token_usage_prompt for s in trajectory.trajectory_steps))
            if trajectory.exit_reason == 'llm_query_error':
                raise RuntimeError('llm_query_error; see trajectory and collector log')
            try:
                patch = checked_run(env.runtime.run, 'git diff --binary ' + shlex.quote(actual_base))
                if stage == 'loc':
                    if patch:
                        raise ValueError('LOC modified repository files')
                    artifact = finish_result(trajectory)
                    if not isinstance(artifact.get('files'), list) or not isinstance(artifact.get('functions', []), list):
                        raise ValueError('location output needs files and functions lists')
                    artifact['files'] = artifact['files'][:5]
                    artifact['functions'] = artifact.get('functions', [])[:10]
                elif stage == 'repro_given_loc':
                    artifact = capture_repro(env.runtime.run, actual_base, finish_result(trajectory))
                    record['artifact_sha256'] = artifact_sha256(artifact)
                else:
                    try:
                        from .execution import patch_paths, is_test_path
                    except ImportError:
                        from stage_capability_eval.execution import patch_paths, is_test_path
                    paths = patch_paths(patch)
                    if any(is_test_path(p) or p.startswith('.git') for p in paths):
                        raise ValueError('PATCH must modify production source only')
                    artifact = {'patch': patch}
                target = folder / 'artifact.json'
                target.write_text(json.dumps(artifact, indent=2, ensure_ascii=False))
                record.update(status='completed', artifact_path=str(target), artifact_sha256=digest(artifact))
            except ValueError as exc:
                record.update(status='model_invalid', failure_category='invalid_artifact', detail=str(exc))
        except Exception as exc:
            traceback.print_exc()
            record.update(status='infra_error', failure_category='collection_runtime', detail=f'{type(exc).__name__}: {exc}')
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    traceback.print_exc()
    record['duration_seconds'] = time.monotonic() - started
    (folder / 'record.json').write_text(json.dumps(record, indent=2))
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=BUDGETS, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--api-base', default='http://127.0.0.1:8132/v1')
    parser.add_argument('--docker-host', default='tcp://127.0.0.1:23750')
    parser.add_argument('--manifest', default=str(ROOT / 'stage_capability_eval/manifests/all_swebench_verified.jsonl'))
    parser.add_argument('--selection', help='JSONL file selecting task IDs')
    parser.add_argument('--config', default=str(CONFIG))
    parser.add_argument('--out', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--max-steps', type=int)
    args = parser.parse_args(argv)
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error('workers and limit must be positive')
    args.max_steps = args.max_steps or BUDGETS[args.stage]
    if args.stage == 'loc':
        rows = read_jsonl(args.manifest)
    else:
        rows = read_jsonl(ROOT / f'stage_capability_eval/manifests/{args.stage}.jsonl')
    if args.selection:
        selected = [r['instance_id'] for r in read_jsonl(args.selection)]
        index = {r['instance_id']: r for r in rows}
        rows = [index[iid] for iid in selected]
    if args.limit:
        rows = rows[:args.limit]
    # Whitelist before crossing the multiprocessing boundary.
    rows = [{**public_task_row(row), **({'oracle_loc': _public_oracle_loc(row['oracle_loc'])}
             if args.stage != 'loc' else {})} for row in rows]
    if not rows or len({r['instance_id'] for r in rows}) != len(rows):
        raise ValueError('empty or duplicate task set')
    options = vars(args)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    options['out'] = str(out)
    spec = {**options, 'tasks_sha256': digest(rows), 'temperature': 0, 'compression': False,
            'config_sha256': hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
            'collector_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'protocol_id': 'stage_capability_v1', 'planned': len(rows)}
    manifest = out / 'run_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != spec:
        raise ValueError('run configuration changed; use a fresh output directory')
    manifest.write_text(json.dumps(spec, indent=2))
    journal_path = out / 'records.jsonl'
    old = {r['instance_id']: r for r in read_jsonl(journal_path)} if journal_path.exists() else {}
    todo = [r for r in rows if old.get(r['instance_id'], {}).get('status') not in ('completed', 'model_invalid')]
    print(json.dumps({'event': 'start', 'stage': args.stage, 'planned': len(rows), 'pending': len(todo)}), flush=True)
    with journal_path.open('a') as journal, ProcessPoolExecutor(max_workers=args.workers,
            mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(collect_task, row, options) for row in todo]
        for future in as_completed(futures):
            record = future.result()
            journal.write(json.dumps(record) + '\n')
            journal.flush()
            os.fsync(journal.fileno())
            old[record['instance_id']] = record
            print(json.dumps({'event': 'task_end', **record}), flush=True)
    summary = {'planned': len(rows), 'completed': sum(r['status'] == 'completed' for r in old.values()),
               'model_invalid': sum(r['status'] == 'model_invalid' for r in old.values()),
               'infra_pending': sum(r['status'] == 'infra_error' for r in old.values())}
    (out / 'collection_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    return 2 if summary['infra_pending'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
