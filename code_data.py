"""Prepare fixed public code benchmark data without executing dataset contents.

MBPP comes from OpenCompass's official dataset archive. Its archive checksum is
also published in the installed OpenCompass dataset URL mapping. HumanEval CN
has no independently verified public download in that mapping: a genuine local
Chinese JSONL is accepted, but English HumanEval is never substituted for it.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
import time
import urllib.error
import urllib.request
import zipfile

MBPP_URL = 'https://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/mbpp.zip'
MBPP_ARCHIVE_SHA256 = 'd753de7da78cc92b7c3dc07f2a1ae3a7ef6933024c86560e3d6326651650a744'
MBPP_JSONL_SHA256 = '0b5e07b5b40fedd01c57b7c9177b79f8b2eebd72c304c928605d91a7237d2f33'
MBPP_MEMBER = 'mbpp/mbpp.jsonl'
MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024
MAX_JSONL_BYTES = 4 * 1024 * 1024
DOWNLOAD_TIMEOUT = 30
DATA_FILES = {
    'humaneval_cn': 'data/humaneval_cn/human-eval-cn-v2-20210705.jsonl',
    'mbpp': 'data/mbpp/mbpp.jsonl',
}
CN_MISSING_REASON = ('缺少真实 HumanEval CN 数据；尚无已核验的官方中文下载源，'
                     '请将含中文 prompt、task_id、test、entry_point 的 164 题 JSONL '
                     '放入 data/humaneval_cn/human-eval-cn-v2-20210705.jsonl')


class CodeDataError(ValueError):
    """A bounded, user-safe dataset preparation failure."""


def _fixed_path(root, kind):
    base = Path(root).resolve()
    path = base / DATA_FILES[kind]
    # Reject symlinks even when they lead back inside the project. This keeps the
    # installer from silently replacing an alias or traversing outside root.
    current = base
    for component in Path(DATA_FILES[kind]).parts:
        current = current / component
        if current.is_symlink():
            raise CodeDataError('数据路径包含符号链接，请使用项目内的普通目录和文件')
    if not path.resolve().is_relative_to(base):
        raise CodeDataError('数据路径必须位于当前项目内')
    return path


def _nonempty_string(value):
    return isinstance(value, str) and bool(value.strip())


def _jsonl_records(data):
    if len(data) > MAX_JSONL_BYTES:
        raise CodeDataError('数据文件超过大小上限')
    try:
        text = data.decode('utf-8-sig')
    except UnicodeDecodeError:
        raise CodeDataError('数据文件必须使用 UTF-8 编码') from None
    records = []
    try:
        for line in text.splitlines():
            if line.strip():
                if len(line) > 256 * 1024:
                    raise CodeDataError('数据记录超过大小上限')
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise CodeDataError('JSONL 的每条记录必须是对象')
                records.append(row)
                if len(records) > 2000:
                    raise CodeDataError('数据记录数超过上限')
    except (json.JSONDecodeError, RecursionError):
        raise CodeDataError('数据文件不是有效的逐行 JSON') from None
    return records


def _validate_data(kind, data):
    rows = _jsonl_records(data)
    if kind == 'mbpp':
        if len(rows) != 974:
            raise CodeDataError('MBPP 应包含完整的 974 条原始题目')
        ids = []
        for row in rows:
            task_id = row.get('task_id')
            tests = row.get('test_list')
            if (type(task_id) is not int
                    or not _nonempty_string(row.get('text'))
                    or not _nonempty_string(row.get('code'))
                    or not isinstance(row.get('test_setup_code', ''), str)
                    or not isinstance(tests, list) or not tests
                    or not all(_nonempty_string(test) for test in tests)):
                raise CodeDataError('MBPP 缺少题目、代码或测试用例字段')
            ids.append(task_id)
        if set(ids) != set(range(1, 975)):
            raise CodeDataError('MBPP 题目编号缺失或重复')
        # Pin the actual JSONL as well as its archive. This also prevents a
        # modified local file from being presented as the official benchmark.
        if hashlib.sha256(data).hexdigest() != MBPP_JSONL_SHA256:
            raise CodeDataError('MBPP 内容校验失败；请重新安装官方数据')
    else:
        if len(rows) != 164:
            raise CodeDataError('HumanEval CN 应包含完整的 164 条题目')
        ids = []
        chinese_prompts = 0
        for row in rows:
            if not all(_nonempty_string(row.get(key))
                       for key in ('task_id', 'prompt', 'test', 'entry_point')):
                raise CodeDataError('HumanEval CN 缺少 task_id、prompt、test 或 entry_point')
            if not re.fullmatch(r'[A-Za-z_]\w*', row['entry_point'], flags=re.ASCII):
                raise CodeDataError('HumanEval CN 的 entry_point 必须是 Python 函数名')
            ids.append(row['task_id'])
            chinese_prompts += bool(re.search(r'[\u3400-\u9fff]', row['prompt']))
        if set(ids) != {f'HumanEval/{index}' for index in range(164)}:
            raise CodeDataError('HumanEval CN 题目编号缺失或重复')
        if not chinese_prompts:
            raise CodeDataError('数据中的 prompt 不含中文；不能将英文 HumanEval 作为 HumanEval CN')
    return len(rows)


def _read_data(path):
    # The fixed paths above reject symlinks. O_NOFOLLOW closes the final-file
    # replacement race on platforms that support it; reading is still bounded.
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0)
    fd = os.open(path, flags)
    with os.fdopen(fd, 'rb') as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise CodeDataError('数据路径必须是普通文件')
        data = stream.read(MAX_JSONL_BYTES + 1)
    if len(data) > MAX_JSONL_BYTES:
        raise CodeDataError('数据文件超过大小上限')
    return data


def code_data_status(root):
    """Return local schema/content readiness, without imports or code execution."""
    status = {}
    for kind, relative in DATA_FILES.items():
        state = dict(ready=False, path=str(Path(root).resolve() / relative),
                     count=0, reason='', source=MBPP_URL if kind == 'mbpp' else 'local')
        try:
            path = _fixed_path(root, kind)
            if not path.exists():
                state['reason'] = CN_MISSING_REASON if kind == 'humaneval_cn' else '缺少 MBPP 数据；可点击安装代码评测环境下载'
            else:
                data = _read_data(path)
                state.update(ready=True, count=_validate_data(kind, data),
                             sha256=hashlib.sha256(data).hexdigest())
        except CodeDataError as error:
            state['reason'] = str(error)
        except OSError:
            state['reason'] = '无法读取数据文件，请检查目录和文件权限'
        status[kind] = state
    return status


def _download_mbpp():
    deadline = time.monotonic() + DOWNLOAD_TIMEOUT
    request = urllib.request.Request(MBPP_URL, headers={'User-Agent': 'OpenCompass-Evaluator/1'})
    try:
        with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT) as response:
            length = response.headers.get('Content-Length')
            if length and (not length.isdigit() or int(length) > MAX_DOWNLOAD_BYTES):
                raise CodeDataError('官方 MBPP 压缩包超过下载大小上限')
            chunks, size = [], 0
            while True:
                if time.monotonic() > deadline:
                    raise CodeDataError('下载 MBPP 超时，请稍后重试')
                chunk = response.read(min(65536, MAX_DOWNLOAD_BYTES + 1 - size))
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
                if size > MAX_DOWNLOAD_BYTES:
                    raise CodeDataError('官方 MBPP 压缩包超过下载大小上限')
    except (urllib.error.URLError, TimeoutError, OSError):
        raise CodeDataError('无法下载官方 MBPP 数据，请检查网络后重试') from None
    archive = b''.join(chunks)
    if hashlib.sha256(archive).hexdigest() != MBPP_ARCHIVE_SHA256:
        raise CodeDataError('官方 MBPP 压缩包 SHA256 校验失败，请重试')
    return _mbpp_from_archive(archive)


def _mbpp_from_archive(data):
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
            if len(entries) > 32:
                raise CodeDataError('MBPP 压缩包目录异常')
            names = set()
            for entry in entries:
                path = PurePosixPath(entry.filename)
                mode = entry.external_attr >> 16
                if (entry.filename in names or '\\' in entry.filename
                        or path.is_absolute() or '..' in path.parts
                        or (path.parts and ':' in path.parts[0])
                        or stat.S_ISLNK(mode) or entry.flag_bits & 1):
                    raise CodeDataError('MBPP 压缩包包含不安全或重复的路径')
                names.add(entry.filename)
                if entry.file_size > MAX_JSONL_BYTES:
                    raise CodeDataError('MBPP 压缩包成员超过大小上限')
            if MBPP_MEMBER not in names:
                raise CodeDataError('MBPP 压缩包缺少原始 JSONL 文件')
            # Read exactly one known member. No archive paths are extracted.
            with archive.open(MBPP_MEMBER) as stream:
                content = stream.read(MAX_JSONL_BYTES + 1)
            _validate_data('mbpp', content)
            return content
    except (zipfile.BadZipFile, KeyError, RuntimeError, NotImplementedError, EOFError):
        raise CodeDataError('官方 MBPP 压缩包格式无效') from None


def _atomic_write(root, kind, data):
    path = _fixed_path(root, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    path = _fixed_path(root, kind)
    fd, temporary = tempfile.mkstemp(prefix='.code-data-', dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, 'wb') as stream:
            fd = None
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        _fixed_path(root, kind)
        os.replace(temporary, path)
    finally:
        if fd is not None:
            os.close(fd)
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def prepare_code_data(root, progress=None):
    """Install verified MBPP; retain an honest missing-data state for CN.

    Calls never accept a URL or destination from the browser. Existing valid
    data is reused. Failures preserve old files and return user-safe reasons.
    """
    report = progress or (lambda message: None)
    state = code_data_status(root)
    if not state['mbpp']['ready']:
        report('正在下载并校验官方 MBPP 数据（974 题）')
        try:
            data = _download_mbpp()
            _atomic_write(root, 'mbpp', data)
            report('MBPP 数据校验完成，已保存至项目 data/mbpp')
        except CodeDataError as error:
            state['mbpp']['reason'] = str(error)
            report(str(error))
            return state
        except OSError:
            state['mbpp']['reason'] = '无法保存 MBPP 数据，请检查项目目录权限后重试'
            report(state['mbpp']['reason'])
            return state
    else:
        report('已复用校验通过的 MBPP 数据（974 题）')
    state = code_data_status(root)
    if not state['humaneval_cn']['ready']:
        report(state['humaneval_cn']['reason'])
    return state
