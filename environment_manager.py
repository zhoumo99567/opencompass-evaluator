"""Fixed-action installer for the isolated code-evaluation environment.

No model settings, API keys, shell commands, user URLs, or user paths enter the
installer. Docker build/self-test details belong to the fixed code_runtime
adapter; only that adapter may attest that the execution environment is ready.
"""
import copy
import fcntl
import importlib
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ACTIONS = ('start_runtime', 'install_code', 'check')
MAX_LOGS = 120
MAX_MESSAGE = 320
RUNTIME_WAIT_SECONDS = 120
BUILD_TIMEOUT_SECONDS = 900
STEP_DEFINITIONS = {
    'start_runtime': [('runtime', '启动容器运行时')],
    'install_code': [('runtime', '检查并启动容器运行时'), ('build', '构建固定代码评测镜像'),
                     ('self_test', '验证容器隔离与评分协议'), ('data', '准备固定代码评测数据')],
    'check': [('runtime', '检查容器运行时'), ('self_test', '重新验证容器隔离与评分协议'),
              ('data', '检查代码评测数据')],
}


def _runtime_api():
    return importlib.import_module('code_runtime')


def _data_api():
    return importlib.import_module('code_data')


def clean_subprocess_env():
    # Never copy the parent environment and try to blacklist possible secrets.
    allowed = ('PATH', 'HOME', 'USER', 'LOGNAME', 'TMPDIR', 'LANG', 'LC_ALL',
               'LC_CTYPE', 'DOCKER_HOST', 'DOCKER_CONTEXT', 'DOCKER_CONFIG')
    return {name: os.environ[name] for name in allowed if name in os.environ}


def _now():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def _message(value):
    # This is a final backstop for messages from fixed internal helpers. Raw
    # subprocess output and HTTP error bodies are deliberately never logged.
    value = str(value).replace('\n', ' ').replace('\r', ' ')
    value = re.sub(r'(?i)\bBearer\s+\S+', 'Bearer [REDACTED]', value)
    value = re.sub(r'(?i)\b[A-Z0-9_]*(?:api[_-]?key|token|password|secret|authorization)\s*[:=]\s*\S+', '[REDACTED]', value)
    value = re.sub(r'https?://\S+', '[下载地址]', value)
    return value[:MAX_MESSAGE]


def _empty_job():
    return {'status': 'idle', 'action': None, 'step': None, 'steps': [], 'logs': [],
            'error': None, 'warnings': [], 'id': None, 'created_at': None, 'finished_at': None}


class InstallationError(Exception):
    pass


class EnvironmentManager:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.state_path = self.root / '.cache' / 'code-eval-installation.json'
        self.lock_path = self.root / '.cache' / 'code-eval-installation.lock'
        self.lock = threading.RLock()
        self.busy = False
        self.process_lock = None
        self.job = self._read_job()

    def _read_job(self):
        try:
            data = json.loads(self.state_path.read_text(encoding='utf-8'))
            if isinstance(data, dict) and data.get('status') in ('idle', 'running', 'completed', 'failed', 'interrupted'):
                result = _empty_job()
                result.update({key: data[key] for key in result if key in data})
                result['logs'] = (result.get('logs') or [])[-MAX_LOGS:]
                return result
        except (ValueError, OSError):
            pass
        return _empty_job()

    def _save(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix('.' + secrets.token_hex(4) + '.tmp')
        try:
            with temporary.open('x', encoding='utf-8') as output:
                temporary.chmod(0o600)
                json.dump(self.job, output, ensure_ascii=False, indent=2)
            temporary.replace(self.state_path)
        finally:
            temporary.unlink(missing_ok=True)

    def _acquire_process_lock(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.lock_path.open('a', encoding='utf-8')
        self.lock_path.chmod(0o600)
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return handle
        except BlockingIOError:
            handle.close()
            return None

    @staticmethod
    def _release_process_lock(handle):
        if handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()

    def _externally_busy(self):
        if not self.lock_path.exists():
            return False
        handle = self._acquire_process_lock()
        if handle is None:
            return True
        self._release_process_lock(handle)
        return False

    @staticmethod
    def _application_path():
        if sys.platform != 'darwin':
            return None
        for candidate in ('/Applications/OrbStack.app', '/Applications/Docker.app'):
            if Path(candidate).is_dir():
                return candidate
        return None

    def _environment(self):
        try:
            api = _runtime_api()
            state = api.environment_status(self.root)
            return state, api
        except Exception:
            return {'docker_available': False, 'daemon_available': False, 'ready': False,
                    'image': 'local/opencompass-code-eval:1', 'image_id': None,
                    'self_test_passed': False, 'supported_datasets': [],
                    'reason': '代码评测适配器尚未就绪，请更新项目后重试'}, None

    def _data_status(self, api):
        try:
            rows = api.data_status(self.root) if api else []
            return [{'id': row['id'], 'label': row.get('label') or row['id'],
                     'ready': bool(row.get('ready')), 'message': _message(row.get('reason') or row.get('message') or ('数据已就绪' if row.get('ready') else '数据未准备'))}
                    for row in rows]
        except Exception:
            return []

    def status(self):
        with self.lock:
            busy = self.busy
            job = copy.deepcopy(self.job if busy else self._read_job())
        if not busy:
            busy = self._externally_busy()
        state, api = self._environment()
        application = self._application_path()
        installed = bool(state.get('docker_available'))
        running = bool(state.get('daemon_available'))
        can_start = bool(installed and application)
        runtime = {'installed': installed, 'running': running,
                   'name': 'OrbStack' if application and 'OrbStack' in application else 'Docker',
                   'can_start': can_start,
                   'message': '容器运行时已启动' if running else ('已安装，尚未启动' if can_start else '未检测到可用容器运行时，请先安装或启动 Docker/OrbStack')}
        code = {'ready': bool(state.get('ready')), 'image': state.get('image') or 'local/opencompass-code-eval:1',
                'image_id': state.get('image_id'), 'self_test_passed': bool(state.get('self_test_passed')),
                'supported_datasets': state.get('supported_datasets') or [],
                'message': _message(state.get('reason') or ('隔离环境已就绪' if state.get('ready') else '代码评测环境尚未就绪')),
                'data': self._data_status(api)}
        context_ready = all((self.root / 'code_sandbox' / name).is_file() for name in ('Dockerfile', 'runner.py'))
        job.update(runtime=runtime, code=code, can_install=bool(not busy and installed and (running or can_start) and api and context_ready),
                   can_start_runtime=bool(not busy and can_start and not running), busy=busy)
        return job

    def _log(self, message):
        with self.lock:
            self.job['logs'].append({'at': _now(), 'message': _message(message)})
            self.job['logs'] = self.job['logs'][-MAX_LOGS:]
            self._save()

    def _step(self, step_id, status, message=None):
        with self.lock:
            self.job['step'] = step_id
            for item in self.job['steps']:
                if item['id'] == step_id:
                    item['status'] = status
                    break
            self._save()
        if message:
            self._log(message)

    def start(self, action):
        if not isinstance(action, str) or action not in ACTIONS:
            raise ValueError('未知安装动作，只支持 start_runtime、install_code、check')
        with self.lock:
            if self.busy:
                raise ValueError('环境任务正在运行，请等待完成')
            handle = self._acquire_process_lock()
            if handle is None:
                raise ValueError('另一个服务正在准备环境，请等待完成')
            self.process_lock = handle
            self.busy = True
            self.job = dict(_empty_job(), status='running', action=action,
                            id='environment_' + secrets.token_hex(8), created_at=_now(),
                            steps=[{'id': step_id, 'label': label, 'status': 'pending'}
                                   for step_id, label in STEP_DEFINITIONS[action]])
            try:
                self._save()
                threading.Thread(target=self._work, args=(action,), daemon=True, name='code-environment-setup').start()
            except Exception:
                self.job.update(status='failed', error='无法启动环境任务，请重试', finished_at=_now())
                try:
                    self._save()
                finally:
                    self.busy = False
                    self.process_lock = None
                    self._release_process_lock(handle)
                raise RuntimeError('无法启动环境任务，请重试') from None
        return self.status()

    def _run_command(self, command, timeout, label):
        # The command is built only by this module or fixed code_runtime helpers.
        # No arguments come from the HTTP payload except its allowlisted action.
        try:
            process = subprocess.Popen(command, cwd=self.root, env=clean_subprocess_env(),
                                       stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        except OSError:
            raise InstallationError(label + '无法启动，请检查程序是否已安装') from None
        milestones = set()

        def drain():
            try:
                while True:
                    raw = process.stdout.readline(16_385)
                    if not raw:
                        return
                    text = raw.decode('utf-8', errors='replace').lower()
                    # Log only fixed milestone labels, never raw process output.
                    for marker, message in (('load metadata', '正在检查基础镜像'),
                                            ('downloading', '正在下载固定基础镜像层'),
                                            ('exporting', '正在保存代码评测镜像')):
                        if marker in text and marker not in milestones:
                            milestones.add(marker)
                            self._log(message)
            except (OSError, ValueError):
                return

        reader = threading.Thread(target=drain, daemon=True, name='code-environment-log')
        reader.start()
        try:
            try:
                returncode = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._terminate(process)
                raise InstallationError(label + '超过等待时限，请检查网络和容器运行状态后重试') from None
            if returncode:
                raise InstallationError(f'{label}失败（退出码 {returncode}），请检查容器运行状态和网络后重试')
        finally:
            reader.join(timeout=2)
            if process.stdout:
                process.stdout.close()

    @staticmethod
    def _terminate(process):
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=3)
        except (OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                pass

    def _ensure_runtime(self, allow_start):
        api = _runtime_api()
        runtime = api.runtime_status(self.root)
        if not runtime.get('docker_available'):
            raise InstallationError('未安装 Docker CLI，请先安装 Docker 或 OrbStack')
        if runtime.get('daemon_available'):
            self._log('已连接容器运行时，保持当前 Docker context')
            return
        application = self._application_path()
        if not allow_start or not application:
            raise InstallationError('容器运行时尚未启动，请先启动 Docker/OrbStack 后重试')
        self._log('正在启动已安装的容器应用，保持当前 Docker context')
        self._run_command(['/usr/bin/open', '-g', '-a', application], 30, '启动容器应用')
        deadline = time.monotonic() + RUNTIME_WAIT_SECONDS
        while time.monotonic() < deadline:
            if api.runtime_status(self.root).get('daemon_available'):
                self._log('容器运行时已启动')
                return
            time.sleep(2)
        raise InstallationError('已请求启动容器应用，但 Docker 仍未就绪；请在应用中完成首次设置后重试')

    def _warning(self, message):
        with self.lock:
            self.job['warnings'].append(_message(message))
            self._save()
        self._log(message)

    def _prepare_data(self):
        try:
            result = _data_api().prepare_code_data(self.root, progress=self._log)
        except Exception:
            self._warning('隔离环境已就绪，自动准备数据未完成；请查看各数据集状态后重试')
            return False
        if not isinstance(result, dict):
            self._warning('隔离环境已就绪，但数据准备状态无效；请检查各数据集状态')
            return False
        missing = []
        for name, label in (('humaneval_cn', 'HumanEval CN'), ('mbpp', 'MBPP')):
            item = result.get(name) or {}
            if not item.get('ready'):
                missing.append(name)
                self._warning(label + '：' + _message(item.get('reason') or item.get('message') or '数据尚未准备'))
        if missing:
            self._log('基础隔离环境已就绪；已具备数据的代码评测集可以使用，其余数据需要单独准备')
            return False
        self._log('两份固定代码评测数据均已准备并验证')
        return True

    def _work(self, action):
        try:
            self._step('runtime', 'running', '开始检查代码评测所需的容器运行时')
            self._ensure_runtime(allow_start=action in ('start_runtime', 'install_code'))
            self._step('runtime', 'completed', '容器运行时检查完成')
            if action != 'start_runtime':
                api = _runtime_api()
                if action == 'install_code':
                    self._step('build', 'running', '开始构建固定的代码评分镜像')
                    self._run_command(api.build_command(self.root), BUILD_TIMEOUT_SECONDS, '构建代码评测镜像')
                    self._step('build', 'completed', '代码评测镜像构建完成')
                self._step('self_test', 'running', '验证无网络、只读根目录、权限和资源限制，以及通过/失败测试用例')
                proof = api.verify_environment(self.root)
                if not (isinstance(proof, dict) and proof.get('ready') and proof.get('protocol') == 1 and
                        proof.get('image_id') and set(api.DATASETS) <= set(proof.get('supported_datasets') or [])):
                    raise InstallationError('容器隔离或评分协议自检未通过，请检查容器运行状态后重试')
                self._step('self_test', 'completed', '固定代码评分容器隔离与协议自检通过')
                self._step('data', 'running', '检查固定代码评测数据')
                if action == 'install_code':
                    data_complete = self._prepare_data()
                else:
                    rows = api.data_status(self.root)
                    data_complete = len(rows) >= 2 and all(row.get('ready') for row in rows)
                    if not data_complete:
                        self._warning('容器自检已通过，部分代码评测数据未齐备；请查看各数据集的准备状态')
                self._step('data', 'completed' if data_complete else 'partial',
                           '代码评测数据检查通过' if data_complete else '部分数据仍需准备，基础隔离环境可用')
            with self.lock:
                self.job.update(status='completed', error=None, finished_at=_now())
                self._save()
            self._log('环境任务已完成')
        except Exception as error:
            with self.lock:
                for step in self.job['steps']:
                    if step['status'] == 'running':
                        step['status'] = 'failed'
                self.job.update(status='failed', error=str(error) if isinstance(error, InstallationError) else
                                '环境任务发生错误，请检查容器运行状态后重试', finished_at=_now())
                self._save()
            self._log(self.job['error'])
        finally:
            with self.lock:
                handle = self.process_lock
                self.process_lock = None
                self.busy = False
                self._release_process_lock(handle)

    def recover_interrupted(self):
        with self.lock:
            if self.busy:
                return
            handle = self._acquire_process_lock()
            if handle is None:
                return
            try:
                self.job = self._read_job()
                if self.job['status'] == 'running':
                    for step in self.job['steps']:
                        if step.get('status') == 'running':
                            step['status'] = 'interrupted'
                    self.job.update(status='interrupted', error='服务已重启，环境任务中断；可重新检查或安装/修复', finished_at=_now())
                    self._save()
                    self._log('已恢复上次任务记录；没有把未完成的安装标记为可用')
            finally:
                self._release_process_lock(handle)
