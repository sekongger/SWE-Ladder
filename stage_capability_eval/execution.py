"""Private execution of frozen REPRO artifacts and PATCH submissions."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping

from .r2egym_docker_runner import DockerHost, safe_relative_path
from .repro import artifact_sha256, validate_artifact


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def parse_junit(xml: str) -> dict[str, str]:
    """Parse actual executed cases; never infer a test from a process code."""
    root = ET.fromstring(xml)
    tests: dict[str, str] = {}
    for node in root.iter('testcase'):
        name = node.get('name')
        if not name:
            raise ValueError('testcase has no name')
        key = node.get('classname', '') + '::' + name
        if key in tests:
            raise ValueError('duplicate test identifier in JUnit report')
        tests[key] = ('error' if node.find('error') is not None else
                      'failed' if node.find('failure') is not None else
                      'skipped' if node.find('skipped') is not None else 'passed')
    return tests


def persist_command(folder: Path, name: str, result) -> dict[str, Any]:
    folder.mkdir(parents=True, exist_ok=True)
    stdout, stderr = folder / f'{name}.stdout.log', folder / f'{name}.stderr.log'
    stdout.write_text(result.stdout, encoding='utf-8')
    stderr.write_text(result.stderr, encoding='utf-8')
    return {**result.to_dict(), 'stdout_path': str(stdout), 'stderr_path': str(stderr)}


def patch_paths(patch: str) -> list[str]:
    # All public SWE-bench paths in this manifest are unquoted and whitespace-free.
    # Refuse unsupported/unsafe syntax rather than silently dropping files.
    paths = []
    for line in patch.splitlines():
        if not line.startswith('diff --git '):
            continue
        match = re.fullmatch(r'diff --git a/(\S+) b/(\S+)', line)
        if not match:
            raise ValueError('unsupported diff path syntax')
        for path in match.groups():
            safe_relative_path(path)
            if path not in paths:
                paths.append(path)
    if not paths:
        raise ValueError('empty or malformed unified diff')
    return paths


def is_test_path(path: str) -> bool:
    parts = path.lower().split('/')
    return any(p in ('test', 'tests', 'testing') for p in parts[:-1]) or parts[-1].startswith('test_') or parts[-1].endswith('_test.py')


def build_test_spec(row: Mapping[str, Any], *, reuse_image_dependencies: bool):
    """Build only the offline evaluation spec, without fetching build recipes.

    Existing R2E-Gym images already contain dependencies. The adapted mode
    omits the install step and records this difference from the stock harness.
    Repository test commands, directives and parsers remain those of 4.1.0.
    """
    from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
    from swebench.harness.test_spec.python import make_eval_script_list_py
    from swebench.harness.test_spec.test_spec import TestSpec
    specs = dict(MAP_REPO_VERSION_TO_SPECS[row['repo']][row['version']])
    skipped_install = specs.pop('install', None) if reuse_image_dependencies else None
    commands = make_eval_script_list_py(dict(row), specs, 'testbed', '/testbed',
                                       row['base_commit'], row['test_patch'])
    # The stock script continues after patch errors. Stop before scoring if
    # installing the hidden tests fails; do not score the old tests instead.
    commands = [c + '\n[ $? -eq 0 ] || exit 97' if c.startswith('git apply ') else c
                for c in commands]
    spec = TestSpec(instance_id=row['instance_id'], repo=row['repo'], version=row['version'],
                    repo_script_list=[], env_script_list=[], eval_script_list=commands,
                    arch='x86_64', FAIL_TO_PASS=row['FAIL_TO_PASS'], PASS_TO_PASS=row['PASS_TO_PASS'],
                    language='py', docker_specs={}, namespace=None)
    return spec, skipped_install


class SwebenchVerifier:
    """Pinned SWE-bench 4.1.0 test commands and parser on R2E-Gym images.

    The official script applies the private test patch after applying the model
    source patch. Tests run once as a suite using repo-specific commands.
    """
    def __init__(self, host: DockerHost, log_root: str | Path, *, reuse_image_dependencies: bool = True):
        self.host, self.log_root = host, Path(log_root)
        self.reuse_image_dependencies = reuse_image_dependencies
        self.harness_version = importlib.metadata.version('swebench')
        if self.harness_version != '4.1.0':
            raise RuntimeError('this protocol pins swebench==4.1.0')

    def run(self, row: Mapping[str, Any], candidate_patch: str, *, phase: str = 'candidate',
            reference: bool = False) -> dict[str, Any]:
        from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
        from swebench.harness.constants import START_TEST_OUTPUT, END_TEST_OUTPUT
        iid = safe_relative_path(str(row['instance_id']))
        folder = self.log_root / iid / phase
        folder.mkdir(parents=True, exist_ok=True)
        record: dict[str, Any] = {
            'schema_version': 'stage_capability_execution_v2', 'instance_id': iid,
            'task_type': 'REFERENCE_PREFLIGHT' if reference else 'PATCH_GIVEN_LOC',
            'phase': phase, 'base_commit': row['base_commit'], 'docker_image': row['docker_image'],
            'candidate_patch_sha256': sha(candidate_patch), 'harness_version': self.harness_version,
            'execution_protocol': 'r2egym_image_dependencies_v1' if self.reuse_image_dependencies else 'swebench_install_v1',
            'status': 'infra_error', 'conditional_resolved': None, 'patch_applied': False,
        }
        if not reference:
            try:
                paths = patch_paths(candidate_patch)
                hidden_paths = set(patch_paths(row['test_patch']))
                if any(is_test_path(p) or p in hidden_paths or p.startswith('.git') for p in paths):
                    raise ValueError('submission must modify production source only')
            except ValueError as exc:
                return {**record, 'status': 'model_invalid', 'conditional_resolved': False,
                        'failure_category': 'invalid_source_patch', 'detail': str(exc)}
        try:
            with self.host.task(row['docker_image']) as container:
                record['environment'] = container.identity(row)
                if candidate_patch:
                    applied = container.apply_patch(candidate_patch)
                    record['patch_apply'] = persist_command(folder, 'patch_apply', applied)
                    if applied.returncode != 0 or applied.infra_error or applied.timed_out:
                        status = 'timeout' if applied.timed_out else ('infra_error' if applied.infra_error or reference else 'model_invalid')
                        return {**record, 'status': status, 'failure_category': 'patch_apply',
                                'conditional_resolved': False if status == 'model_invalid' else None}
                record['patch_applied'] = True
                spec, skipped_install = build_test_spec(row, reuse_image_dependencies=self.reuse_image_dependencies)
                record['skipped_install_command'] = skipped_install
                script = spec.eval_script
                record['eval_script_sha256'] = sha(script)
                (folder / 'eval.sh').write_text(script)
                container.put_files({'stage-eval.sh': script}, '/tmp')
                result = container.exec('bash /tmp/stage-eval.sh 2>&1')
                record['execution'] = persist_command(folder, 'test', result)
                if result.timed_out or result.infra_error:
                    return {**record, 'status': 'timeout' if result.timed_out else 'infra_error',
                            'failure_category': 'test_execution'}
                if result.returncode == 97:
                    return {**record, 'status': 'invalid', 'failure_category': 'test_patch_apply'}
                text = result.stdout
                if START_TEST_OUTPUT not in text or END_TEST_OUTPUT not in text:
                    return {**record, 'failure_category': 'missing_test_output_markers'}
                body = text.split(START_TEST_OUTPUT, 1)[1].split(END_TEST_OUTPUT, 1)[0]
                parsed = MAP_REPO_TO_PARSER[row['repo']](body, spec)
                if not parsed:
                    return {**record, 'failure_category': 'no_tests_collected'}
                record['tests'] = parsed
                for group in ('FAIL_TO_PASS', 'PASS_TO_PASS'):
                    record[group] = {name: parsed.get(name, 'NOT_COLLECTED') for name in row[group]}
                f2p, p2p = record['FAIL_TO_PASS'], record['PASS_TO_PASS']
                resolved = bool(f2p) and all(s == 'PASSED' for s in f2p.values()) and all(s == 'PASSED' for s in p2p.values())
                return {**record, 'status': 'completed', 'conditional_resolved': resolved,
                        'failure_category': None if resolved else 'verifier_not_resolved'}
        except Exception as exc:
            return {**record, 'failure_category': 'environment_or_harness', 'detail': f'{type(exc).__name__}: {exc}'}


class DockerReproRunner:
    """Run only the declared model pytest artifact on isolated buggy/fixed images.

    Reference test patches are deliberately never installed. Production tracked
    files must not be overwritten or changed by the reproduction artifact.
    """
    def __init__(self, host: DockerHost, private_records: Mapping[str, Mapping[str, Any]], log_root: str | Path):
        self.host, self.private_records, self.log_root = host, private_records, Path(log_root)

    def __call__(self, task: Mapping[str, Any], version: str, artifact: Mapping[str, Any]) -> dict[str, Any]:
        if version not in ('buggy', 'fixed'):
            raise ValueError('unknown source version')
        artifact = validate_artifact(artifact)
        digest = artifact_sha256(artifact)
        row = self.private_records[task['instance_id']]
        folder = self.log_root / safe_relative_path(row['instance_id']) / version
        base: dict[str, Any] = {'status': 'infra_error', 'artifact_sha256': digest, 'tests': {},
                                'version': version, 'base_commit': row['base_commit']}
        command = list(artifact['command'])
        if command[:3] == ['python', '-m', 'pytest']:
            args = command[3:]
        elif command[:1] == ['pytest']:
            args = command[1:]
        else:
            return {**base, 'status': 'model_invalid', 'failure_category': 'unsupported_runner_command'}
        if not artifact['files'] or any(arg.startswith(('--junit', '--collect-only', '--co')) for arg in args):
            return {**base, 'status': 'model_invalid', 'failure_category': 'invalid_test_submission'}
        # v1 contract: file targets must be declared artifact tests, never hidden tests.
        targets = [arg.split('::')[0] for arg in args if not arg.startswith('-')]
        if not targets or any(p not in artifact['files'] for p in targets):
            return {**base, 'status': 'model_invalid', 'failure_category': 'undeclared_test_target'}
        try:
            with self.host.task(row['docker_image']) as container:
                base['environment'] = container.identity(row)
                tracked = container.exec(['git', 'ls-files'], timeout_seconds=15)
                if tracked.returncode != 0:
                    return {**base, 'failure_category': 'cannot_read_tracked_files'}
                tracked_files = set(tracked.stdout.splitlines())
                for path in artifact['files']:
                    safe_relative_path(path)
                    if path in tracked_files:
                        return {**base, 'status': 'model_invalid', 'failure_category': 'overwrites_tracked_file'}
                if version == 'fixed':
                    applied = container.apply_patch(row['patch'])
                    base['gold_patch_apply'] = persist_command(folder, 'source_patch', applied)
                    if applied.returncode != 0 or applied.infra_error or applied.timed_out:
                        return {**base, 'failure_category': 'gold_patch_apply'}
                before = container.exec(['git', 'diff', '--binary'], timeout_seconds=15)
                if before.returncode != 0:
                    return {**base, 'failure_category': 'source_fingerprint'}
                base['source_diff_sha256'] = sha(before.stdout)
                container.put_files(artifact['files'])
                result = container.exec(['python', '-m', 'pytest', *args, '--junitxml=/tmp/stage-results.xml'])
                base['execution'] = persist_command(folder, 'test', result)
                if result.timed_out or result.infra_error:
                    return {**base, 'status': 'timeout' if result.timed_out else 'infra_error', 'failure_category': 'test_execution'}
                after = container.exec(['git', 'diff', '--binary'], timeout_seconds=15)
                if after.returncode != 0:
                    return {**base, 'failure_category': 'source_fingerprint'}
                if sha(after.stdout) != sha(before.stdout):
                    return {**base, 'status': 'model_invalid', 'failure_category': 'production_source_modified'}
                actual_files = {p: container.read_file('/testbed/' + p) for p in artifact['files']}
                actual_digest = artifact_sha256({'files': actual_files, 'command': artifact['command']})
                if actual_digest != digest:
                    return {**base, 'status': 'model_invalid', 'failure_category': 'submitted_artifact_modified'}
                try:
                    xml = container.read_file('/tmp/stage-results.xml')
                    tests = parse_junit(xml)
                    (folder / 'junit.xml').write_text(xml)
                except Exception as exc:
                    return {**base, 'status': 'model_invalid', 'failure_category': 'no_test_report', 'detail': str(exc)}
                if not tests:
                    return {**base, 'status': 'model_invalid', 'failure_category': 'no_tests_collected'}
                return {**base, 'status': 'completed', 'tests': tests, 'collection_error': result.returncode not in (0, 1),
                        'junit_path': str(folder / 'junit.xml')}
        except Exception as exc:
            return {**base, 'failure_category': 'environment', 'detail': f'{type(exc).__name__}: {exc}'}
