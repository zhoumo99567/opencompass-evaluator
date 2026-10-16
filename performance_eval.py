"""End-to-end streaming benchmarks for OpenAI-compatible chat APIs.

Each matrix cell is a closed-loop workload: a fixed number of requests is
scheduled by N workers. Prompt size is an estimate; reported server usage is
kept separately. No model text, API credentials, or HTTP error bodies are saved.
"""
import copy
import csv
import http.client
import io
import json
import math
import queue
import secrets
import socket
import statistics
import threading
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

MAX_RESPONSE_BYTES = 16_000_000
MAX_LINE_BYTES = 1_000_000
METRIC_NOTES = {
    'workload': 'closed_loop_fixed_concurrency',
    'input_size': 'approximate_chars_divided_by_4; use actual_prompt_tokens for realized input sizes',
    'ttft_ms': 'first generated nonempty text, including reasoning; available only for SSE',
    'ttfc_ms': 'first nonempty final answer content; available only for SSE',
    'tpot_ms': '(last generated text time - first generated text time) / (completion_tokens - 1); estimate including reasoning',
    'chunk_interval_ms': 'interval between SSE events containing text; events are not tokens',
    'generation_tps': 'completion_tokens / (last generated text time - first generated text time); estimate including reasoning; unavailable for one text event',
    'slo': 'user-defined optional TTFT/E2E limits; missing required timing or failed request does not pass; pass rate denominator is all measured requests',
    'throughput': 'successful request token usage divided by full measured cell wall time; missing usage excluded from token totals',
    'e2e_ms': 'request start through response completion; successful generated responses only',
    'warmup': 'warm-up requests excluded from results',
    'cache': 'unique prefix for each request reduces shared-prefix caching; server cache policy is not controlled',
}


def now_iso():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def distribution(values):
    """Linear-interpolated percentiles, retaining null when nothing is observed."""
    values = sorted(float(value) for value in values if value is not None and math.isfinite(value))
    if not values:
        return {'mean': None, 'p50': None, 'p95': None, 'p99': None}

    def percentile(fraction):
        position = (len(values) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        return round(values[lower] + (values[upper] - values[lower]) * (position - lower), 3)

    return {'mean': round(statistics.mean(values), 3), 'p50': percentile(.5),
            'p95': percentile(.95), 'p99': percentile(.99)}


def validate_config(payload):
    if not isinstance(payload, dict):
        raise ValueError('性能测试配置必须是对象')

    def integer(name, default, lower, upper):
        value = payload.get(name, default)
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f'{name} 必须是 {lower}–{upper} 之间的整数')
        return value

    def integer_list(name, default, lower, upper):
        values = payload.get(name, default)
        if not isinstance(values, list) or not values or len(values) > 36:
            raise ValueError(f'{name} 必须是非空整数列表，最多 36 项')
        if any(type(value) is not int or not lower <= value <= upper for value in values):
            raise ValueError(f'{name} 的值必须是 {lower}–{upper} 之间的整数')
        return list(dict.fromkeys(values))

    config = {
        'concurrency': integer_list('concurrency', [1, 2, 4], 1, 64),
        'input_tokens': integer_list('input_tokens', [1024, 4096, 8192], 64, 262144),
        'requests_per_cell': integer('requests_per_cell', 8, 1, 200),
        'output_tokens': integer('output_tokens', 256, 1, 8192),
        'warmup': integer('warmup', 1, 0, 10),
        'timeout_seconds': integer('timeout_seconds', 120, 1, 600),
        'prompt_token_estimation': 'approximate_chars_divided_by_4',
        'cache_policy': 'unique_prefix_per_request', 'stream': True,
    }
    for name in ('slo_ttft_ms', 'slo_e2e_ms'):
        value = payload.get(name)
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 600000):
            raise ValueError(f'{name} 必须是大于 0 且不超过 600000 的数值，或留空')
        config[name] = value
    if len(config['concurrency']) * len(config['input_tokens']) > 36:
        raise ValueError('并发数与上下文组合最多 36 组')
    if config['requests_per_cell'] < max(config['concurrency']):
        raise ValueError('每组请求数不能小于最大并发数')
    return config


def synthetic_prompt(target_tokens):
    """No tokenizer downloads: calibrate by approximate characters, record usage."""
    prefix = secrets.token_hex(16) + '\nPerformance benchmark. Treat the reference below as background information.\n'
    ending = ('\nWrite a detailed numbered explanation of how to plan and verify a software project. '
              'Use the reference as background. Continue adding useful steps until your output budget is reached.')
    phrase = 'The project needs clear goals, careful design, useful examples, reliable checks, and measured progress. '
    desired = target_tokens * 4
    remaining = max(0, desired - len(prefix) - len(ending))
    body = (phrase * (remaining // len(phrase) + 1))[:remaining]
    return prefix + body + ending


def _text_counts(message):
    """Count final/reasoning text without retaining the generated text itself."""
    final = reasoning = 0
    content = message.get('content')
    if isinstance(content, str):
        final += len(content) if content.strip() else 0
    elif isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            value = part.get('text')
            if isinstance(value, dict):
                value = value.get('value')
            if not isinstance(value, str) or not value.strip():
                continue
            if part.get('type') in ('reasoning', 'reasoning_text', 'thinking'):
                reasoning += len(value)
            elif part.get('type') in ('text', 'output_text'):
                final += len(value)
    # Different backends use one of these aliases, occasionally both.
    for field in ('reasoning_content', 'reasoning'):
        value = message.get(field)
        if isinstance(value, str) and value.strip():
            reasoning += len(value)
        elif isinstance(value, list):
            for part in value:
                if isinstance(part, dict) and isinstance(part.get('text'), str) and part['text'].strip():
                    reasoning += len(part['text'])
    return final, reasoning


def _usage(data):
    result = {}
    if not isinstance(data, dict):
        return result
    for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
        value = data.get(key)
        if type(value) is int and value >= 0:
            result[key] = value
    details = data.get('completion_tokens_details')
    if isinstance(details, dict) and type(details.get('reasoning_tokens')) is int and details['reasoning_tokens'] >= 0:
        result['reasoning_tokens'] = details['reasoning_tokens']
    if 'total_tokens' not in result and all(key in result for key in ('prompt_tokens', 'completion_tokens')):
        result['total_tokens'] = result['prompt_tokens'] + result['completion_tokens']
    return result


def summarize_cell(cell, config=None):
    samples = cell['samples']
    successful = [sample for sample in samples if sample['success']]
    wall = max(cell.get('elapsed_seconds') or 0, 1e-9)
    prompt_tokens = [sample['usage']['prompt_tokens'] for sample in successful if 'prompt_tokens' in sample['usage']]
    output_lengths = [sample['usage']['completion_tokens'] for sample in successful if 'completion_tokens' in sample['usage']]
    output_tokens = sum(output_lengths)
    total_tokens = sum(sample['usage'].get('total_tokens', 0) for sample in successful)
    final_tokens = sum(max(0, sample['usage']['completion_tokens'] - sample['usage']['reasoning_tokens'])
                      for sample in successful if all(key in sample['usage'] for key in ('completion_tokens', 'reasoning_tokens')))
    metrics = {
        'success_count': len(successful), 'content_success_count': sum(bool(sample['final_content_chars']) for sample in successful),
        'failure_count': len(samples) - len(successful),
        'error_rate': round((len(samples) - len(successful)) / len(samples), 6) if samples else None,
        'reasoning_only_count': sum(not sample['final_content_chars'] and bool(sample['reasoning_chars']) for sample in successful),
        'truncated_count': sum(sample.get('finish_reason') == 'length' for sample in samples),
        'missing_usage_count': sum(not all(key in sample['usage'] for key in ('prompt_tokens', 'completion_tokens')) for sample in successful),
        'nonstream_count': sum(not sample['streamed'] for sample in successful),
        'e2e_ms': distribution([sample['e2e_ms'] for sample in successful]),
        'ttft_ms': distribution([sample['ttft_ms'] for sample in successful]),
        'ttfc_ms': distribution([sample['ttfc_ms'] for sample in successful]),
        'tpot_ms': distribution([sample['tpot_ms'] for sample in successful]),
        'chunk_interval_ms': distribution([value for sample in successful for value in sample['chunk_intervals_ms']]),
        'generation_tps': distribution([sample.get('generation_tps') for sample in successful]),
        'request_throughput_rps': round(len(successful) / wall, 6),
        'output_throughput_tps': round(output_tokens / wall, 6) if any('completion_tokens' in sample['usage'] for sample in successful) else None,
        'total_throughput_tps': round(total_tokens / wall, 6) if any('total_tokens' in sample['usage'] for sample in successful) else None,
        'output_tokens': output_tokens, 'total_tokens': total_tokens,
        'known_final_content_tokens': final_tokens,
        'token_usage_coverage': round(sum('completion_tokens' in sample['usage'] for sample in successful) / len(successful), 6) if successful else None,
    }
    config = config or {}
    thresholds = {key: config.get(key) for key in ('slo_ttft_ms', 'slo_e2e_ms') if config.get(key) is not None}
    passing = 0
    if thresholds:
        for sample in successful:
            if all(sample.get('ttft_ms' if key == 'slo_ttft_ms' else 'e2e_ms') is not None
                   and sample['ttft_ms' if key == 'slo_ttft_ms' else 'e2e_ms'] <= value
                   for key, value in thresholds.items()):
                passing += 1
    metrics.update(slo_pass_count=passing if thresholds else None,
                   slo_pass_rate=round(passing / len(samples), 6) if thresholds and samples else None,
                   goodput_rps=round(passing / wall, 6) if thresholds else None)
    cell['completed'] = len(samples)
    cell['metrics'] = metrics
    cell['actual_prompt_tokens'] = {'min': min(prompt_tokens) if prompt_tokens else None,
                                    'median': statistics.median(prompt_tokens) if prompt_tokens else None,
                                    'max': max(prompt_tokens) if prompt_tokens else None}
    cell['actual_output_tokens'] = {'min': min(output_lengths) if output_lengths else None,
                                    'median': statistics.median(output_lengths) if output_lengths else None,
                                    'max': max(output_lengths) if output_lengths else None}
    return cell


class _Cancelled(Exception):
    pass


class _BadResponse(Exception):
    pass


class _Handle:
    def __init__(self, connection):
        self.connection = connection
        self.socket = None

    def abort(self):
        sock = self.socket or self.connection.sock
        if sock:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass
        # shutdown first wakes another thread blocked inside HTTPResponse.read.
        try:
            self.connection.close()
        except OSError:
            pass


class _Run:
    def __init__(self, report):
        self.report = report
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.connections = set()
        self.started = time.monotonic()
        self.last_saved = 0


class PerformanceEvaluator:
    def __init__(self, reports):
        self.reports = Path(reports)
        self.lock = threading.Lock()
        self.active = None
        self.last_id = None

    def _save(self, report):
        self.reports.mkdir(parents=True, exist_ok=True)
        target = self.reports / (report['id'] + '.json')
        temporary = target.with_suffix('.' + secrets.token_hex(4) + '.tmp')
        try:
            with temporary.open('x', encoding='utf-8') as output:
                temporary.chmod(0o600)
                json.dump(report, output, ensure_ascii=False, indent=2)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)

    def start(self, settings, payload=None):
        config = validate_config(payload if payload is not None else {})
        if not isinstance(settings, dict) or not all(isinstance(settings.get(key), str) and settings[key].strip()
                                                   for key in ('model', 'base_url', 'api_key')):
            raise ValueError('请先保存模型接口、模型名和密钥')
        parsed = urlsplit(settings['base_url'])
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('模型接口地址必须是有效的 HTTP 或 HTTPS 地址')
        model = settings['model'].replace(settings['api_key'], '[REDACTED]')
        report = {
            'id': datetime.now().strftime('%Y%m%d_%H%M%S') + '_performance_' + secrets.token_hex(4),
            'kind': 'performance', 'status': 'running', 'model': model, 'title': model + ' · 端到端性能测试',
            'created_at': now_iso(), 'finished_at': None, 'elapsed_seconds': 0, 'config': config,
            'progress': {'completed': 0, 'total': len(config['concurrency']) * len(config['input_tokens']) * config['requests_per_cell'], 'current': None},
            'cells': [], 'metric_notes': dict(METRIC_NOTES), 'error': None,
        }
        with self.lock:
            if self.active is not None:
                raise ValueError('性能测试正在运行，请等待完成或先停止')
            self._save(report)
            initial_report = copy.deepcopy(report)
            run = _Run(report)
            self.active = run
            self.last_id = report['id']
        try:
            threading.Thread(target=self._run, args=(run, dict(settings)), daemon=True, name='performance-eval').start()
        except Exception:
            with run.lock:
                report.update(status='failed', error='无法启动性能测试', finished_at=now_iso())
                self._save(report)
            self._clear_active(run)
            raise
        return initial_report

    def _clear_active(self, run):
        with self.lock:
            if self.active is run:
                self.active = None

    def status(self):
        with self.lock:
            run, last_id = self.active, self.last_id
        if run:
            with run.lock:
                report = copy.deepcopy(run.report)
                observed_at = time.monotonic()
                report['elapsed_seconds'] = round(observed_at - run.started, 3)
                for cell in report['cells']:
                    if cell.get('status') == 'running' and '_started' in cell:
                        cell['elapsed_seconds'] = round(observed_at - cell['_started'], 6)
                        summarize_cell(cell, report['config'])
                return report
        if last_id:
            try:
                return json.loads((self.reports / (last_id + '.json')).read_text(encoding='utf-8'))
            except (OSError, ValueError):
                pass
        return {'status': 'idle'}

    def stop(self):
        with self.lock:
            run = self.active
        if not run:
            return self.status()
        with run.lock:
            if run.report['status'] != 'running':
                return copy.deepcopy(run.report)
            run.cancel.set()
            handles = list(run.connections)
            run.report.update(status='cancelled', finished_at=now_iso(), elapsed_seconds=round(time.monotonic() - run.started, 3))
            run.report['progress']['current'] = None
            for cell in run.report['cells']:
                if cell['status'] == 'running':
                    cell.update(status='cancelled', elapsed_seconds=round(time.monotonic() - cell.pop('_started'), 6))
                    summarize_cell(cell, run.report['config'])
            self._save(run.report)
            result = copy.deepcopy(run.report)
        for handle in handles:
            handle.abort()
        self._clear_active(run)
        return result

    def recover_interrupted(self):
        latest = None
        for path in self.reports.glob('*_performance_*.json'):
            try:
                report = json.loads(path.read_text(encoding='utf-8'))
                if report.get('kind') != 'performance':
                    continue
                if report.get('status') == 'running':
                    report.update(status='failed', error='服务已重启，性能测试中断，请重新测试', finished_at=now_iso())
                    report.get('progress', {})['current'] = None
                    for cell in report.get('cells', []):
                        cell.pop('_started', None)
                        if cell.get('status') == 'running':
                            cell['status'] = 'failed'
                    self._save(report)
                if latest is None or report['id'] > latest:
                    latest = report['id']
            except (OSError, ValueError, KeyError, TypeError):
                continue
        self.last_id = latest

    def _request(self, run, settings, config, target, index, warmup=False):
        started = time.monotonic()
        sample = {
            'index': index, 'success': False, 'streamed': False, 'e2e_ms': None,
            'ttft_ms': None, 'ttfc_ms': None, 'tpot_ms': None, 'generation_tps': None,
            'chunk_intervals_ms': [], 'text_events': 0, 'final_content_chars': 0,
            'reasoning_chars': 0, 'usage': {}, 'finish_reason': None,
            'error_kind': None, 'error': None,
        }
        connection = response = handle = None
        first_generated = last_generated = first_final = None
        total_bytes = 0
        try:
            if run.cancel.is_set():
                raise _Cancelled()
            parsed = urlsplit(settings['base_url'].rstrip('/'))
            url = parsed.path if parsed.path.endswith('/chat/completions') else parsed.path + '/chat/completions'
            prompt = synthetic_prompt(target)
            sample['prompt_chars'] = len(prompt)
            sample['estimated_prompt_tokens'] = round(len(prompt) / 4)
            request = json.dumps({'model': settings['model'], 'messages': [{'role': 'user', 'content': prompt}],
                                  'temperature': 0, 'max_tokens': config['output_tokens'], 'stream': True,
                                  'stream_options': {'include_usage': True}}).encode('utf-8')
            connection_class = http.client.HTTPSConnection if parsed.scheme == 'https' else http.client.HTTPConnection
            connection = connection_class(parsed.hostname, parsed.port, timeout=config['timeout_seconds'])
            handle = _Handle(connection)
            with run.lock:
                if run.cancel.is_set():
                    raise _Cancelled()
                run.connections.add(handle)
            connection.request('POST', url, body=request,
                               headers={'Content-Type': 'application/json', 'Accept': 'text/event-stream',
                                        'Authorization': 'Bearer ' + settings['api_key']})
            handle.socket = connection.sock
            if run.cancel.is_set():
                raise _Cancelled()
            response = connection.getresponse()
            if response.status < 200 or response.status >= 300:
                sample.update(error_kind='http_error', error=f'模型接口返回 HTTP {response.status}')
                return sample
            is_sse = 'text/event-stream' in (response.getheader('Content-Type') or '').lower()
            sample['streamed'] = is_sse

            def deadline():
                if run.cancel.is_set():
                    raise _Cancelled()
                remaining = config['timeout_seconds'] - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError()
                if handle.socket:
                    handle.socket.settimeout(remaining)

            def process_event(data):
                nonlocal first_generated, last_generated, first_final
                if not isinstance(data, dict) or data.get('error'):
                    raise _BadResponse()
                usage = _usage(data.get('usage'))
                if usage:
                    sample['usage'].update(usage)
                choices = data.get('choices')
                if not isinstance(choices, list):
                    raise _BadResponse()
                if not choices:
                    return
                choice = choices[0]
                if not isinstance(choice, dict):
                    raise _BadResponse()
                # Keep only known finish markers; never retain arbitrary server text.
                finish = choice.get('finish_reason')
                if finish in ('stop', 'length', 'content_filter', 'tool_calls', 'function_call'):
                    sample['finish_reason'] = finish
                message = choice.get('delta') if is_sse else choice.get('message')
                if not isinstance(message, dict):
                    return
                final, reasoning = _text_counts(message)
                sample['final_content_chars'] += final
                sample['reasoning_chars'] += reasoning
                if final or reasoning:
                    at = time.monotonic()
                    sample['text_events'] += 1
                    if is_sse:
                        if first_generated is None:
                            first_generated = at
                            sample['ttft_ms'] = round((at - started) * 1000, 3)
                        elif last_generated is not None:
                            sample['chunk_intervals_ms'].append(round((at - last_generated) * 1000, 3))
                        last_generated = at
                        if final and first_final is None:
                            first_final = at
                            sample['ttfc_ms'] = round((at - started) * 1000, 3)

            if is_sse:
                data_lines = []
                done = False
                while not done:
                    deadline()
                    line = response.readline(MAX_LINE_BYTES + 1)
                    total_bytes += len(line)
                    if len(line) > MAX_LINE_BYTES or total_bytes > MAX_RESPONSE_BYTES:
                        raise _BadResponse()
                    if not line:
                        # Accept a final unterminated event, but require a finish marker.
                        if data_lines:
                            event = '\n'.join(data_lines)
                            if event.strip() != '[DONE]':
                                process_event(json.loads(event))
                        break
                    line = line.decode('utf-8').rstrip('\r\n')
                    if not line:
                        if data_lines:
                            event = '\n'.join(data_lines)
                            data_lines = []
                            if event.strip() == '[DONE]':
                                done = True
                            else:
                                process_event(json.loads(event))
                    elif line.startswith('data:'):
                        data_lines.append(line[5:].lstrip(' '))
                if not done and sample['finish_reason'] is None:
                    raise _BadResponse()
            else:
                deadline()
                body = response.read(MAX_RESPONSE_BYTES + 1)
                if len(body) > MAX_RESPONSE_BYTES:
                    raise _BadResponse()
                process_event(json.loads(body))
            if run.cancel.is_set():
                raise _Cancelled()
            sample['success'] = bool(sample['final_content_chars'] or sample['reasoning_chars'])
            if not sample['success']:
                sample.update(error_kind='empty_output', error='模型没有生成文本（含思考文本）')
            count = sample['usage'].get('completion_tokens')
            if count is not None and count > 1 and first_generated is not None and last_generated is not None and last_generated > first_generated:
                sample['tpot_ms'] = round((last_generated - first_generated) * 1000 / (count - 1), 3)
                sample['generation_tps'] = round(count / (last_generated - first_generated), 6)
        except _Cancelled:
            sample.update(error_kind='cancelled', error='请求已停止')
        except (TimeoutError, socket.timeout):
            sample.update(error_kind='timeout', error='请求超时')
        except (ValueError, TypeError, KeyError, UnicodeError, _BadResponse):
            sample.update(error_kind='cancelled' if run.cancel.is_set() else 'invalid_response',
                          error='请求已停止' if run.cancel.is_set() else '模型响应格式无效、输出过大或流式响应中断')
        except (OSError, http.client.HTTPException):
            sample.update(error_kind='cancelled' if run.cancel.is_set() else 'network_error',
                          error='请求已停止' if run.cancel.is_set() else '连接失败或流式响应中断')
        except Exception:
            sample.update(error_kind='request_error', error='请求异常')
        finally:
            sample['e2e_ms'] = round((time.monotonic() - started) * 1000, 3)
            if handle:
                with run.lock:
                    run.connections.discard(handle)
                handle.abort()
            if response:
                response.close()
        return sample

    def _run(self, run, settings):
        report, config = run.report, run.report['config']
        try:
            for target in config['input_tokens']:
                for concurrency in config['concurrency']:
                    if run.cancel.is_set():
                        return
                    with run.lock:
                        if run.cancel.is_set():
                            return
                        cell = {'id': f'input{target}_concurrency{concurrency}', 'concurrency': concurrency,
                                'input_tokens_target': target, 'requests': config['requests_per_cell'],
                                'completed': 0, 'achieved_concurrency': 0, 'status': 'running',
                                'elapsed_seconds': 0, 'warmup_errors': 0, 'samples': []}
                        cell['_started'] = time.monotonic()
                        report['cells'].append(cell)
                        report['progress']['current'] = f'输入约 {target} tokens · 并发 {concurrency} · 预热'
                        self._save(report)
                    for warmup_index in range(config['warmup']):
                        if run.cancel.is_set():
                            return
                        sample = self._request(run, settings, config, target, warmup_index, warmup=True)
                        if not sample['success']:
                            cell['warmup_errors'] += 1
                    if run.cancel.is_set():
                        return
                    measured_started = time.monotonic()
                    with run.lock:
                        if run.cancel.is_set():
                            return
                        cell['_started'] = measured_started
                        report['progress']['current'] = f'输入约 {target} tokens · 并发 {concurrency} · 测量'
                    pending = queue.Queue()
                    for index in range(config['requests_per_cell']):
                        pending.put(index)
                    inflight = [0]

                    def worker():
                        while not run.cancel.is_set():
                            try:
                                index = pending.get_nowait()
                            except queue.Empty:
                                return
                            with run.lock:
                                if run.cancel.is_set():
                                    return
                                inflight[0] += 1
                                cell['achieved_concurrency'] = max(cell['achieved_concurrency'], inflight[0])
                            sample = self._request(run, settings, config, target, index)
                            with run.lock:
                                inflight[0] -= 1
                                cell['samples'].append(sample)
                                cell['completed'] = len(cell['samples'])
                                report['progress']['completed'] += 1
                                cell['elapsed_seconds'] = round(time.monotonic() - measured_started, 6)
                                summarize_cell(cell, config)
                                # A stopped task may still collect its final interrupted requests.
                                if time.monotonic() - run.last_saved > .5 or run.cancel.is_set():
                                    self._save(report)
                                    run.last_saved = time.monotonic()

                    workers = [threading.Thread(target=worker, daemon=True) for _ in range(concurrency)]
                    for thread in workers:
                        thread.start()
                    for thread in workers:
                        while thread.is_alive() and not run.cancel.is_set():
                            thread.join(.1)
                    with run.lock:
                        cell.pop('_started', None)
                        cell['elapsed_seconds'] = round(time.monotonic() - measured_started, 6)
                        cell['status'] = 'cancelled' if run.cancel.is_set() else 'completed'
                        summarize_cell(cell, config)
                        self._save(report)
                    if run.cancel.is_set():
                        return
            with run.lock:
                successful = sum(cell['metrics']['success_count'] for cell in report['cells'])
                report.update(status='completed' if successful else 'failed',
                              error=None if successful else '所有请求均失败，请查看分组结果并检查接口配置')
        except Exception:
            with run.lock:
                if not run.cancel.is_set():
                    report.update(status='failed', error='性能测试发生异常，请查看已完成结果后重试')
        finally:
            with run.lock:
                for cell in report['cells']:
                    started = cell.pop('_started', None)
                    if cell.get('status') == 'running':
                        cell['status'] = 'cancelled' if run.cancel.is_set() else 'failed'
                        if started is not None:
                            cell['elapsed_seconds'] = round(time.monotonic() - started, 6)
                        summarize_cell(cell, config)
                if run.cancel.is_set():
                    report['status'] = 'cancelled'
                report.update(finished_at=report['finished_at'] or now_iso(), elapsed_seconds=round(time.monotonic() - run.started, 3))
                report['progress']['current'] = None
                self._save(report)
            self._clear_active(run)


def report_to_csv(report):
    output = io.StringIO(newline='')
    fields = ['cell_id', 'concurrency', 'input_tokens_target', 'index', 'success', 'streamed',
              'e2e_ms', 'ttft_ms', 'ttfc_ms', 'tpot_ms', 'generation_tps', 'prompt_tokens', 'completion_tokens',
              'total_tokens', 'reasoning_tokens', 'final_content_chars', 'reasoning_chars',
              'finish_reason', 'error_kind', 'error']
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for cell in report.get('cells', []):
        for sample in cell.get('samples', []):
            row = {key: sample.get(key) for key in fields if key not in ('cell_id', 'concurrency', 'input_tokens_target')}
            row.update(cell_id=cell['id'], concurrency=cell['concurrency'], input_tokens_target=cell['input_tokens_target'])
            row.update(sample.get('usage') or {})
            writer.writerow({key: row.get(key) for key in fields})
    return output.getvalue()
