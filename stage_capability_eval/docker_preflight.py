#!/usr/bin/env python3
"""Resumable private old/fixed preflight using pinned repository test suites."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from typing import Any, Mapping

if __package__ in (None, ''):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage_capability_eval.execution import SwebenchVerifier
from stage_capability_eval.r2egym_docker_runner import DockerHost


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def select_rows(source, manifest):
    indexed = {r['instance_id']: r for r in source}
    ids = [r['instance_id'] for r in manifest]
    if len(indexed) != len(source) or len(set(ids)) != len(ids):
        raise ValueError('duplicate source or selection ID')
    missing = set(ids) - indexed.keys()
    if missing:
        raise ValueError(f'selection IDs absent from private source: {sorted(missing)}')
    return [indexed[iid] for iid in ids]


def summarize_reference(phases: Mapping[str, Any]):
    for status in ('infra_error', 'timeout'):
        if any(p['status'] == status for p in phases.values()):
            return {'status': status, 'reference_valid': None, 'failure_category': 'reference_execution'}
    old, fixed = phases['buggy'], phases['fixed']
    of, ff = old.get('FAIL_TO_PASS', {}), fixed.get('FAIL_TO_PASS', {})
    op, fp = old.get('PASS_TO_PASS', {}), fixed.get('PASS_TO_PASS', {})
    all_outcomes = (*of.values(), *ff.values(), *op.values(), *fp.values())
    if any(status in {'ERROR', 'NOT_COLLECTED', 'TIMEOUT'} for status in all_outcomes):
        return {'status': 'infra_error', 'reference_valid': None,
                'failure_category': 'reference_test_error'}
    checks = {
        'both_phases_completed': all(p['status'] == 'completed' for p in phases.values()),
        'same_nonempty_f2p': bool(of) and of.keys() == ff.keys(),
        'all_buggy_f2p_failed': bool(of) and all(s == 'FAILED' for s in of.values()),
        'all_fixed_f2p_passed': bool(ff) and all(s == 'PASSED' for s in ff.values()),
        'all_p2p_passed_both': op.keys() == fp.keys() and all(s == 'PASSED' for s in (*op.values(), *fp.values())),
    }
    valid = all(checks.values())
    return {'status': 'completed', 'reference_valid': valid, 'checks': checks,
            'failure_category': None if valid else 'reference_expectation_mismatch'}


def run_reference_task(row, verifier):
    started = time.monotonic()
    phases = {phase: verifier.run(row, '' if phase == 'buggy' else row['patch'],
                                  phase=phase, reference=True) for phase in ('buggy', 'fixed')}
    return {'schema_version': 'stage_capability_reference_v2', 'instance_id': row['instance_id'],
            'base_commit': row['base_commit'], 'repo': row['repo'], 'docker_image': row['docker_image'],
            'dataset_revision': row.get('dataset_revision'), 'phases': phases,
            **summarize_reference(phases), 'duration_seconds': time.monotonic() - started}


def fingerprint(row, image_id, *, timeout_seconds, harness_version, reuse_image_dependencies):
    fields = ('instance_id', 'repo', 'version', 'base_commit', 'patch', 'test_patch',
              'FAIL_TO_PASS', 'PASS_TO_PASS', 'dataset_revision')
    source_hashes = {p: hashlib.sha256(Path(__file__).with_name(p).read_bytes()).hexdigest()
                     for p in ('docker_preflight.py', 'execution.py', 'r2egym_docker_runner.py')}
    data = {'source': {k: row[k] for k in fields}, 'image_id': image_id,
            'timeout_seconds': timeout_seconds, 'harness_version': harness_version,
            'reuse_image_dependencies': reuse_image_dependencies, 'code': source_hashes}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def run_preflight(rows, host, *, out, log_root, workers=2, reuse_image_dependencies=True):
    """Append/fsync each result; resume only matching terminal records.

    Each attempt owns separate logs. Infra/timeout attempts stay in the journal
    and are retried on the next invocation.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    verifier = SwebenchVerifier(host, log_root, reuse_image_dependencies=reuse_image_dependencies)
    previous = {r['instance_id']: r for r in _jsonl(out)} if out.exists() else {}
    results, todo = [], []
    for row in rows:
        try:
            image_id = host.client.images.get(row['docker_image']).id
        except Exception:
            image_id = None
        digest = fingerprint(row, image_id, timeout_seconds=host.timeout_seconds,
                             harness_version=verifier.harness_version,
                             reuse_image_dependencies=reuse_image_dependencies)
        old = previous.get(row['instance_id'], {})
        if old.get('input_sha256') == digest and old.get('status') == 'completed':
            results.append(old)
        else:
            todo.append((row, image_id, digest))
    print(json.dumps({'event': 'start', 'planned': len(rows), 'resumed': len(results),
                      'pending': len(todo), 'workers': workers}), flush=True)

    def run(item):
        row, image_id, digest = item
        attempt = uuid.uuid4().hex
        print(json.dumps({'event': 'task_start', 'instance_id': row['instance_id'], 'attempt': attempt}), flush=True)
        if image_id is None:
            record = {'instance_id': row['instance_id'], 'status': 'infra_error',
                      'reference_valid': None, 'failure_category': 'image_unavailable'}
        else:
            # Both phases use the same immutable image ID, even if a tag moves.
            task = {**row, 'docker_image': image_id}
            task_verifier = SwebenchVerifier(host, Path(log_root) / attempt,
                                             reuse_image_dependencies=reuse_image_dependencies)
            record = run_reference_task(task, task_verifier)
            record['docker_image'] = row['docker_image']
        return {**record, 'input_sha256': digest, 'image_id': image_id, 'attempt': attempt}

    with out.open('a', encoding='utf-8') as journal, ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(run, item) for item in todo}
        while pending:
            done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            if not done:
                print(json.dumps({'event': 'heartbeat', 'completed': len(results), 'planned': len(rows)}), flush=True)
            for future in done:
                record = future.result()
                journal.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')
                journal.flush()
                os.fsync(journal.fileno())
                results.append(record)
                print(json.dumps({'event': 'task_end', 'completed': len(results), 'planned': len(rows),
                                  'instance_id': record['instance_id'], 'status': record['status'],
                                  'reference_valid': record['reference_valid']}), flush=True)
    index = {r['instance_id']: r for r in results}
    return [index[r['instance_id']] for r in rows]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='stage_capability_eval/evaluator_private/source_records.jsonl')
    parser.add_argument('--manifest', default='stage_capability_eval/manifests/dev_sample.jsonl')
    parser.add_argument('--host', default='tcp://127.0.0.1:23750')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--stock-install', action='store_true', help='Use stock install step instead of existing image dependencies')
    parser.add_argument('--out', default='stage_capability_eval/evaluator_private/reference_dev25.jsonl')
    parser.add_argument('--log-root', default='stage_capability_eval/evaluator_private/execution_logs/dev25')
    args = parser.parse_args(argv)
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error('workers and limit must be positive')
    selected = select_rows(_jsonl(args.source), _jsonl(args.manifest))
    if args.limit: selected = selected[:args.limit]
    host = DockerHost(args.host, timeout_seconds=args.timeout)
    print(json.dumps({'event': 'host', **host.ping()}), flush=True)
    rows = run_preflight(selected, host, out=args.out, log_root=args.log_root, workers=args.workers,
                         reuse_image_dependencies=not args.stock_install)
    summary = {'planned': len(rows), 'valid': sum(r['reference_valid'] is True for r in rows),
               'reference_mismatch': sum(r['reference_valid'] is False for r in rows),
               'pending': sum(r['reference_valid'] is None for r in rows), 'out': args.out}
    Path(args.out + '.summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)
    return 0 if summary['valid'] == summary['planned'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
