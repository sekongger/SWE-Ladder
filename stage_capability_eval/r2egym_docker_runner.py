"""Isolated, bounded Docker execution using the existing R2E-Gym images.

No model-visible mounts, no daemon configuration, no implicit image pulls.
Each verifier invocation owns and removes only the container it creates.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import shlex
import tarfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence


def safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or str(path) != value or '..' in path.parts
            or '.git' in path.parts or '\\' in value or '\x00' in value):
        raise ValueError(f'unsafe artifact path: {value!r}')
    return value


def _archive(files: Mapping[str, str]) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w') as archive:
        for rel, content in sorted(files.items()):
            safe_relative_path(rel)
            info = tarfile.TarInfo(name=rel)
            encoded = content.encode('utf-8')
            info.size, info.mode = len(encoded), 0o644
            archive.addfile(info, io.BytesIO(encoded))
    return stream.getvalue()


@dataclass(frozen=True)
class DockerCommandResult:
    returncode: int | None
    stdout: str = ''
    stderr: str = ''
    timed_out: bool = False
    infra_error: bool = False
    duration_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key in ('stdout', 'stderr'):
            result[key + '_sha256'] = hashlib.sha256(result.pop(key).encode()).hexdigest()
        return result


class DockerTaskContainer:
    def __init__(self, client: Any, image: str, *, timeout_seconds: float = 300):
        self.client, self.timeout_seconds = client, timeout_seconds
        details = client.images.get(image).attrs
        self.image_id = details['Id']
        self.image_digests = details.get('RepoDigests', [])
        self.container = client.containers.create(
            self.image_id, command=['bash', '-lc', 'sleep 86400'], working_dir='/testbed',
            labels={'stage_capability_eval': '1'}, mem_limit='8g', nano_cpus=2_000_000_000,
        )
        try:
            self.container.start()
        except BaseException:
            self.close()
            raise

    def put_files(self, files: Mapping[str, str], destination: str = '/testbed') -> None:
        if files and not self.container.put_archive(destination, _archive(files)):
            raise RuntimeError(f'put_archive failed: {destination}')

    def read_file(self, path: str) -> str:
        stream, _ = self.container.get_archive(path)
        with tarfile.open(fileobj=io.BytesIO(b''.join(stream))) as archive:
            members = [m for m in archive.getmembers() if m.isfile()]
            if len(members) != 1:
                raise ValueError('expected one regular file')
            return archive.extractfile(members[0]).read().decode('utf-8', errors='replace')

    def exec(self, command: str | Sequence[str], *, timeout_seconds: float | None = None) -> DockerCommandResult:
        timeout = self.timeout_seconds if timeout_seconds is None else timeout_seconds
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('timeout must be finite and positive')
        started = time.monotonic()
        script = command if isinstance(command, str) else shlex.join(command)
        token = uuid.uuid4().hex
        out, err = f'/tmp/stage-{token}.out', f'/tmp/stage-{token}.err'
        wrapped = f'timeout --kill-after=5s {timeout}s bash -lc {shlex.quote(script)} >{out} 2>{err}'
        try:
            api = self.client.api
            eid = api.exec_create(self.container.id, ['bash', '-lc', wrapped], workdir='/testbed')['Id']
            api.exec_start(eid, detach=True)
            while True:
                state = api.exec_inspect(eid)
                if not state['Running']:
                    code = state['ExitCode']
                    break
                if time.monotonic() - started > timeout + 10:
                    self.container.kill()
                    return DockerCommandResult(None, timed_out=True, duration_seconds=time.monotonic()-started)
                time.sleep(0.2)
            return DockerCommandResult(code, self.read_file(out), self.read_file(err),
                                       code in (124, 137), False, time.monotonic()-started)
        except Exception as exc:
            return DockerCommandResult(None, stderr=f'{type(exc).__name__}: {exc}',
                                       infra_error=True, duration_seconds=time.monotonic()-started)

    def apply_patch(self, patch: str) -> DockerCommandResult:
        self.put_files({'stage-source.patch': patch}, '/tmp')
        return self.exec('git apply --check /tmp/stage-source.patch && git apply /tmp/stage-source.patch; '
                         'code=$?; rm /tmp/stage-source.patch; exit "$code"')

    def run_test(self, test_name: str, *, timeout_seconds: float | None = None) -> DockerCommandResult:
        """Run one pytest node inside the task worktree.

        Keeping this as a single-node operation lets the reference preflight
        retain the exact F2P/P2P outcome and classify timeouts per test.
        """
        if not test_name or '\x00' in test_name:
            raise ValueError('test_name must be non-empty and NUL-free')
        return self.exec(['python', '-m', 'pytest', '-q', test_name], timeout_seconds=timeout_seconds)

    def identity(self, row: Mapping[str, Any]) -> dict[str, Any]:
        head = self.exec(['git', 'rev-parse', 'HEAD'], timeout_seconds=15)
        if head.returncode != 0 or head.stdout.strip() != row['base_commit']:
            raise RuntimeError('container HEAD does not match frozen base_commit')
        diff = self.exec(['git', 'diff', '--binary'], timeout_seconds=15)
        if diff.returncode != 0:
            raise RuntimeError('cannot fingerprint image working tree')
        return {'image_id': self.image_id, 'image_digests': self.image_digests,
                'base_commit': head.stdout.strip(), 'image_setup_diff_sha256': hashlib.sha256(diff.stdout.encode()).hexdigest()}

    def close(self) -> None:
        self.container.remove(force=True)

    def __enter__(self):
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class DockerHost:
    def __init__(self, base_url: str = 'tcp://127.0.0.1:23750', *, timeout_seconds: float = 300):
        import docker
        self.client = docker.DockerClient(base_url=base_url, timeout=30)
        self.base_url, self.timeout_seconds = base_url, timeout_seconds

    def ping(self) -> dict[str, Any]:
        info = self.client.info()
        return {k: info.get(k) for k in ('Name', 'NCPU', 'MemTotal', 'DockerRootDir', 'ContainersRunning', 'Images')}

    def task(self, image: str) -> DockerTaskContainer:
        return DockerTaskContainer(self.client, image, timeout_seconds=self.timeout_seconds)
