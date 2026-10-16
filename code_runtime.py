"""Lean Docker execution boundary; no OpenCompass import and no host exec."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
import uuid

IMAGE = 'local/opencompass-code-eval:1'
PROTOCOL = 1
LABEL_PROTOCOL = 'org.opencompass.code-eval.protocol'
LABEL_SOURCE = 'org.opencompass.code-eval.source'
DATASETS = {
    'humaneval_cn_gen': dict(label='HumanEval CN', kind='humaneval_cn',
                           path='data/humaneval_cn/human-eval-cn-v2-20210705.jsonl'),
    'mbpp_gen': dict(label='MBPP', kind='mbpp', path='data/mbpp/mbpp.jsonl'),
}
_CACHE = {}
_CACHE_LOCK = threading.Lock()


class CodeEnvironmentError(RuntimeError):
    """Infrastructure failed; never count this as a model's wrong answer."""


def clean_environment():
    allowed = ('PATH', 'HOME', 'TMPDIR', 'DOCKER_HOST', 'DOCKER_CONTEXT',
               'DOCKER_CONFIG', 'DOCKER_TLS_VERIFY', 'DOCKER_CERT_PATH')
    return {**{key: os.environ[key] for key in allowed if key in os.environ},
            'LANG': 'C.UTF-8'}


def source_hash(root):
    digest = hashlib.sha256()
    for name in ('Dockerfile', 'runner.py'):
        digest.update(name.encode())
        digest.update((Path(root) / 'code_sandbox' / name).read_bytes())
    return digest.hexdigest()


def proof_path(root):
    return Path(root) / '.cache/code-eval-environment.json'


def build_command(root):
    return ['docker', 'build', '--label', f'{LABEL_PROTOCOL}={PROTOCOL}',
            '--label', f'{LABEL_SOURCE}={source_hash(root)}', '-t', IMAGE,
            str(Path(root) / 'code_sandbox')]


def container_command(name, image_id=None):
    return ['docker', 'run', '--rm', '--name', name, '--network', 'none',
            '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
            '--user', '65534:65534', '--pids-limit', '32', '--memory', '256m',
            '--memory-swap', '256m', '--cpus', '1', '--ipc', 'none',
            '--tmpfs', '/tmp:rw,noexec,nosuid,nodev,size=32m',
            '--workdir', '/tmp', '--log-driver', 'none', '-i', image_id or IMAGE]


def _control(args, timeout=8):
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout,
                          env=clean_environment(), check=False)


def runtime_status(root=None):
    result = dict(docker_available=bool(shutil.which('docker')), daemon_available=False,
                  image_available=False, image=IMAGE, image_id='', reason='')
    if not result['docker_available']:
        return dict(result, reason='未安装 Docker CLI')
    try:
        info = _control(['docker', 'info', '--format', '{{.ServerVersion}}'])
        if info.returncode:
            return dict(result, reason='代码执行环境不可用：Docker 服务尚未启动或当前连接不可用')
        result['daemon_available'] = True
        fmt = '{{.Id}}|{{index .Config.Labels "' + LABEL_PROTOCOL + '"}}|{{index .Config.Labels "' + LABEL_SOURCE + '"}}'
        image = _control(['docker', 'image', 'inspect', '--format', fmt, IMAGE])
        if image.returncode:
            return dict(result, reason='代码评测镜像尚未构建')
        image_id, protocol, digest = image.stdout.strip().split('|')
        result.update(image_available=True, image_id=image_id,
                      protocol=protocol, source_hash=digest)
        return result
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return dict(result, reason='无法在时限内检查 Docker 代码评测环境')


def environment_status(root, refresh=False):
    root = Path(root)
    cache_key = str(root.resolve())
    with _CACHE_LOCK:
        cached = _CACHE.get(cache_key)
        if not refresh and cached and time.monotonic() - cached[0] < 3:
            return dict(cached[1])
    state = dict(runtime_status(root), ready=False, self_test_passed=False,
                 supported_datasets=[])
    if state['image_available']:
        try:
            proof = json.loads(proof_path(root).read_text(encoding='utf-8'))
            digest = source_hash(root)
            valid = (proof.get('ready') is True and proof.get('protocol') == PROTOCOL
                     and proof.get('image_id') == state['image_id']
                     and proof.get('source_hash') == digest
                     and state.get('protocol') == str(PROTOCOL)
                     and state.get('source_hash') == digest
                     and proof.get('supported_datasets') == list(DATASETS))
            if valid:
                state.update(ready=True, self_test_passed=True,
                             supported_datasets=list(DATASETS), reason='',
                             checked_at=proof.get('checked_at'), checks=proof.get('checks', {}))
            else:
                state['reason'] = '镜像或评分器已变化，需要重新安装并通过隔离自检'
        except (OSError, ValueError):
            state['reason'] = '代码评测镜像尚未通过隔离自检'
    with _CACHE_LOCK:
        _CACHE[cache_key] = (time.monotonic(), dict(state))
    return state


def data_status(root):
    try:
        from code_data import code_data_status
        states = code_data_status(root)
    except ImportError:
        states = {}
    return [dict(id=dataset_id, label=meta['label'],
                 path=states.get(meta['kind'], {}).get('path', str(Path(root) / meta['path'])),
                 ready=states.get(meta['kind'], {}).get('ready', False),
                 reason=states.get(meta['kind'], {}).get('reason', '代码评测数据尚未准备'),
                 count=states.get(meta['kind'], {}).get('count', 0))
            for dataset_id, meta in DATASETS.items()]


def code_readiness(root, dataset_id):
    if dataset_id not in DATASETS:
        return '此数据集的 Docker 隔离评分适配尚未完成'
    state = environment_status(root)
    if not state['ready']:
        return state['reason'] or '代码评测环境尚未通过隔离自检'
    entry = next(item for item in data_status(root) if item['id'] == dataset_id)
    return '' if entry['ready'] else entry['reason']


def _run_container(arguments, payload=None, timeout=20, image_id=None):
    """Bound wall time AND stdout/stderr, even when candidate code tampers."""
    name = 'oc-code-' + uuid.uuid4().hex
    args = container_command(name, image_id) + list(arguments)
    process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, env=clean_environment())
    buffers = [bytearray(), bytearray()]
    exceeded = threading.Event()

    def drain(stream, buffer):
        while True:
            chunk = stream.read(4096)
            if not chunk:
                break
            remaining = 65536 - len(buffer)
            buffer.extend(chunk[:max(remaining, 0)])
            if len(chunk) > remaining:
                exceeded.set()
                process.kill()

    def send():
        try:
            if payload is not None:
                process.stdin.write(json.dumps(payload).encode())
                process.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        finally:
            process.stdin.close()

    workers = [threading.Thread(target=drain, args=(process.stdout, buffers[0]), daemon=True),
               threading.Thread(target=drain, args=(process.stderr, buffers[1]), daemon=True),
               threading.Thread(target=send, daemon=True)]
    for worker in workers:
        worker.start()
    failed = False
    try:
        process.wait(timeout=timeout)
        for worker in workers:
            worker.join(timeout=2)
        if exceeded.is_set():
            raise CodeEnvironmentError('隔离容器输出超过限制')
        if process.returncode in (137, 139):
            return dict(protocol=PROTOCOL, result='resource_limit', passed=False)
        if process.returncode:
            raise CodeEnvironmentError('Docker 代码评分运行失败；请检查环境状态')
        try:
            result = json.loads(bytes(buffers[0]))
        except ValueError as error:
            raise CodeEnvironmentError('隔离评分器未返回有效结果') from error
        if not isinstance(result, dict) or result.get('protocol') != PROTOCOL:
            raise CodeEnvironmentError('隔离评分器协议不匹配')
        return result
    except subprocess.TimeoutExpired as error:
        failed = True
        process.kill()
        process.wait(timeout=3)
        raise CodeEnvironmentError('Docker 评分启动或通信超时，已请求清理容器') from error
    except Exception:
        failed = True
        raise
    finally:
        if failed or process.returncode:
            try:
                _control(['docker', 'rm', '-f', name], timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                pass
        for stream in (process.stdout, process.stderr):
            stream.close()


def execute_case(root, kind, completion, problem, timeout_seconds=3):
    if kind not in {'humaneval_cn', 'mbpp'}:
        raise ValueError('未适配的代码评分任务')
    if not isinstance(completion, str) or len(completion.encode()) > 524288:
        return dict(protocol=PROTOCOL, result='invalid_output', passed=False)
    state = environment_status(root)
    if not state['ready']:
        raise CodeEnvironmentError(state['reason'])
    payload = dict(protocol=PROTOCOL, kind=kind, completion=completion,
                   problem=problem, timeout_seconds=timeout_seconds)
    result = _run_container([], payload, timeout=timeout_seconds + 20, image_id=state['image_id'])
    if result.get('result') not in {'pass', 'wrong_answer', 'failed', 'timeout',
                                     'invalid_output', 'resource_limit'}:
        raise CodeEnvironmentError('隔离评分器返回未知状态')
    if type(result.get('passed')) is not bool:
        raise CodeEnvironmentError('隔离评分结果缺少明确判定')
    if result['passed'] != (result['result'] == 'pass'):
        raise CodeEnvironmentError('隔离评分结果状态不一致')
    return dict(result, image_id=state['image_id'])


def verify_environment(root):
    state = runtime_status(root)
    digest = source_hash(root)
    if not state['image_available']:
        raise CodeEnvironmentError(state['reason'])
    if state.get('protocol') != str(PROTOCOL) or state.get('source_hash') != digest:
        raise CodeEnvironmentError('镜像版本与当前评分器不匹配，请重新构建')
    checked = _run_container(['--self-check'], timeout=30, image_id=state['image_id'])
    expected = {'pass', 'wrong_answer', 'timeout', 'early_exit', 'stdout_isolated'}
    if (checked.get('passed') is not True or set(checked.get('checks', {})) != expected
            or not all(value is True for value in checked['checks'].values())
            or checked.get('uid') != 65534
            or checked.get('supported') != ['humaneval_cn', 'mbpp']):
        raise CodeEnvironmentError('隔离评分器自检未通过')
    proof = dict(ready=True, protocol=PROTOCOL, image=IMAGE, image_id=state['image_id'],
                 source_hash=digest, supported_datasets=list(DATASETS),
                 checks=checked['checks'], checked_at=datetime.now(timezone.utc).isoformat(), reason='')
    path = proof_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(proof, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.chmod(0o600)
    temporary.replace(path)
    environment_status(root, refresh=True)
    return proof


self_test = verify_environment
probe_runtime = runtime_status
