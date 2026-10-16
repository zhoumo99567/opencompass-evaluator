"""Single-turn pelican SVG probe, with durable results and inert previews."""
import copy
import http.client
import json
import re
import secrets
import socket
import threading
import time
import xml.etree.ElementTree as ET
import tinycss2
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

PROMPT = ('Generate an animated SVG of a pelican riding a bicycle. '
          'Animate the rotating wheels, pedaling legs, and gently moving wings using embedded CSS @keyframes. '
          'Return a complete, self-contained SVG with no JavaScript or external resources.')
SOURCE = 'https://simonwillison.net/2024/Oct/25/pelicans-on-a-bicycle/'
NS = 'http://www.w3.org/2000/svg'
MAX_RESPONSE = 2_000_000
MAX_TOKENS = 32768
ALLOWED_TAGS = set('svg g defs title desc path rect circle ellipse line polyline polygon text tspan linearGradient radialGradient stop clipPath mask pattern use symbol'.split())
ALLOWED_ATTRS = set('id x y x1 x2 y1 y2 cx cy r rx ry width height viewBox preserveAspectRatio d points transform fill fill-rule fill-opacity stroke stroke-width stroke-linecap stroke-linejoin stroke-miterlimit stroke-dasharray stroke-dashoffset stroke-opacity opacity font-family font-size font-weight font-style text-anchor dominant-baseline dx dy rotate letter-spacing offset stop-color stop-opacity gradientUnits gradientTransform spreadMethod fx fy fr clip-path clip-rule clipPathUnits mask maskUnits maskContentUnits patternUnits patternContentUnits patternTransform href'.split())
STYLE_ATTRS = set('fill fill-rule fill-opacity stroke stroke-width stroke-linecap stroke-linejoin stroke-dasharray stroke-dashoffset stroke-opacity opacity font-family font-size font-weight font-style text-anchor dominant-baseline letter-spacing stop-color stop-opacity'.split())
STYLE_ATTRS.update('transform transform-origin transform-box animation animation-name animation-duration animation-delay animation-timing-function animation-iteration-count animation-direction animation-fill-mode animation-play-state'.split())


def safe_style_value(value):
    if re.search(r'[\\<>]|(?:https?:|data:|javascript:|@|expression)', value, re.I):
        return False
    if 'url' in value.lower():
        return bool(re.fullmatch(r'url\(#[\w.-]+\)', value.strip(), re.I))
    return True


def sanitize_css(source):
    """Retain drawing/animation rules, never imports, fonts, or external resources."""
    removed = 0

    def declarations(tokens):
        nonlocal removed
        result = []
        for decl in tinycss2.parse_declaration_list(tokens, skip_comments=True, skip_whitespace=True):
            if decl.type == 'declaration' and decl.lower_name in STYLE_ATTRS:
                value = tinycss2.serialize(decl.value).strip()
                if safe_style_value(value):
                    result.append(f'{decl.lower_name}:{value}' + ('!important' if decl.important else '') + ';')
                    continue
            removed += 1
        return ''.join(result)

    def rules(items, keyframes=False):
        nonlocal removed
        result = []
        for rule in items:
            if rule.type == 'qualified-rule':
                selector = tinycss2.serialize(rule.prelude).strip()
                pattern = r'(?:from|to|\d+(?:\.\d+)?%)(?:\s*,\s*(?:from|to|\d+(?:\.\d+)?%))*' if keyframes else r'[\w\s.#*,>+~-]+'
                if re.fullmatch(pattern, selector):
                    result.append(selector + '{' + declarations(rule.content) + '}')
                    continue
            elif rule.type == 'at-rule' and rule.content is not None and not keyframes:
                prelude = tinycss2.serialize(rule.prelude).strip()
                if rule.lower_at_keyword == 'keyframes' and re.fullmatch(r'[a-zA-Z_][\w-]*', prelude):
                    children = tinycss2.parse_rule_list(rule.content, skip_comments=True, skip_whitespace=True)
                    result.append('@keyframes ' + prelude + '{' + rules(children, keyframes=True) + '}')
                    continue
                if rule.lower_at_keyword == 'media' and re.fullmatch(r'\(\s*prefers-reduced-motion\s*:\s*(?:reduce|no-preference)\s*\)', prelude):
                    children = tinycss2.parse_rule_list(rule.content, skip_comments=True, skip_whitespace=True)
                    # Nested media blocks are unnecessary for this preview.
                    children = [child for child in children if child.type == 'qualified-rule']
                    result.append('@media ' + prelude + '{' + rules(children) + '}')
                    continue
            removed += 1
        return ''.join(result)

    css = rules(tinycss2.parse_stylesheet(source, skip_comments=True, skip_whitespace=True))
    return css, removed


def sanitize_svg(content):
    """Never insert raw model markup into HTML. Return an inert, allowlisted SVG."""
    if not isinstance(content, str) or len(content) > MAX_RESPONSE:
        raise ValueError('模型输出为空或过长')
    if re.search(r'<!\s*(DOCTYPE|ENTITY)', content, re.I):
        raise ValueError('SVG 包含不支持的 XML 声明')
    match = re.search(r'<svg\b[\s\S]*?</svg\s*>', content, re.I)
    if not match:
        raise ValueError('模型没有返回完整 SVG；可查看原始输出后重试')
    raw = match.group(0)
    try:
        tree = ET.fromstring(raw)
    except ET.ParseError as error:
        raise ValueError('SVG 格式无效或被截断，请查看原始输出') from error
    if tree.tag not in ('svg', '{' + NS + '}svg'):
        raise ValueError('模型输出不是 SVG')
    removed = 0

    def safe_value(name, value):
        if name == 'href':
            return bool(re.fullmatch(r'#[\w.-]+', value))
        # No external resources, escaped CSS, or expression-based values.
        if re.search(r'[\\<>]|(?:https?:|data:|javascript:|@import|expression)', value, re.I):
            return False
        if 'url' in value.lower():
            return bool(re.fullmatch(r'url\(#[\w.-]+\)', value.strip(), re.I))
        return True

    def clean(element, depth=0):
        nonlocal removed
        if depth > 64:
            raise ValueError('SVG 嵌套层级过多')
        local = element.tag.removeprefix('{' + NS + '}')
        if local == 'style':
            css, dropped = sanitize_css(element.text or '')
            removed += dropped + len(element.attrib) + len(element)
            result = ET.Element('{' + NS + '}style')
            result.text = css
            return result
        if local not in ALLOWED_TAGS:
            removed += 1
            return None
        result = ET.Element('{' + NS + '}' + local)
        result.text = element.text
        result.tail = element.tail
        for name, value in element.attrib.items():
            if name == '{http://www.w3.org/1999/xlink}href':
                name = 'href'
            if name == 'class' and re.fullmatch(r'[\w\s-]+', value):
                result.set('class', value)
            elif name == 'style':
                css, dropped = sanitize_css('*{' + value + '}')
                removed += dropped
                if css:
                    result.set('style', css[2:-1])
            elif name in ALLOWED_ATTRS and safe_value(name, value):
                result.set(name, value)
            else:
                removed += 1
        for child in element:
            node = clean(child, depth + 1)
            if node is not None:
                result.append(node)
        return result

    cleaned = clean(tree)
    ET.register_namespace('', NS)
    svg = ET.tostring(cleaned, encoding='unicode')
    return svg, removed


def svg_has_animation(svg):
    return bool(re.search(r'@keyframes\s', svg) and re.search(r'animation(?:-name)?\s*:', svg))



# The wire budget includes SSE framing and reasoning deltas; SVG itself retains
# the stricter MAX_RESPONSE limit used by the preview sanitizer.
MAX_WIRE_BYTES = 64_000_000
MAX_EVENT_BYTES = 1_000_000
REQUEST_TIMEOUT_SECONDS = 600


def http_error_message(status, thinking_mode='default'):
    prefix = f'模型接口返回 HTTP {status}：'
    if status == 504:
        return prefix + '网关等待上游模型超时；提高生成额度不能延长网关时限。可使用流式生成、降低模型思考预算，或调整服务端网关超时。流式请求仍可能受到网关总时限限制。'
    if status in (401, 403):
        return prefix + '接口认证或访问权限失败，请检查密钥和服务权限。'
    if status == 400:
        message = '服务端拒绝请求参数，请核对生成额度与上下文窗口的可用范围。'
        if thinking_mode == 'disabled':
            message += '本次还使用了关闭思考的扩展参数；若接口不支持，请选择默认思考模式。'
        return prefix + message
    if status == 429:
        return prefix + '服务限流或当前负载过高，请降低并发或稍后重试。'
    if 500 <= status <= 599:
        return prefix + '模型服务或上游网关出现错误，请检查服务状态与服务端日志。'
    if status == 404:
        return prefix + '接口路径或模型不存在，请核对地址和模型名。'
    return prefix + '请求未被服务接受，请检查接口配置与服务状态。'


def _message_text(message):
    """Final output is retained; reasoning is counted and never persisted."""
    text = ''
    reasoning_chars = 0
    content = message.get('content')
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts = []
        for part in content:
            if not isinstance(part, dict):
                continue
            value = part.get('text')
            if isinstance(value, dict):
                value = value.get('value')
            if not isinstance(value, str):
                continue
            if part.get('type') in ('text', 'output_text'):
                parts.append(value)
            elif part.get('type') in ('reasoning', 'reasoning_text', 'thinking'):
                reasoning_chars += len(value)
        text = ''.join(parts)
    for field in ('reasoning_content', 'reasoning'):
        value = message.get(field)
        if isinstance(value, str):
            reasoning_chars += len(value)
    return text, reasoning_chars, bool(message.get('refusal'))


def _safe_usage(usage):
    if not isinstance(usage, dict):
        return {}
    result = {key: usage[key] for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')
              if type(usage.get(key)) is int and usage[key] >= 0}
    details = usage.get('completion_tokens_details')
    if isinstance(details, dict) and type(details.get('reasoning_tokens')) is int and details['reasoning_tokens'] >= 0:
        result['reasoning_tokens'] = details['reasoning_tokens']
    return result


class _PelicanCancelled(Exception):
    pass


class _PelicanProtocolError(Exception):
    pass


class _PelicanHTTPError(Exception):
    def __init__(self, status):
        self.status = status


class _PelicanRun:
    def __init__(self, report, key):
        self.report = report
        self.key = key
        self.started = time.monotonic()
        self.timeout_seconds = REQUEST_TIMEOUT_SECONDS
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.timed_out = threading.Event()
        self.done = threading.Event()
        self.connection = None
        self.socket = None

    def abort(self):
        sock = self.socket or (self.connection.sock if self.connection else None)
        if sock:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass
        if self.connection:
            try:
                self.connection.close()
            except OSError:
                pass


class PelicanEvaluator:
    def __init__(self, reports):
        self.reports = Path(reports)
        self.lock = threading.Lock()
        self.active_id = None
        self.active_run = None
        self.last_id = None

    def _save(self, report):
        self.reports.mkdir(parents=True, exist_ok=True)
        path = self.reports / (report['id'] + '.json')
        temporary = path.with_suffix('.' + secrets.token_hex(4) + '.tmp')
        try:
            with temporary.open('x', encoding='utf-8') as output:
                temporary.chmod(0o600)
                json.dump(report, output, ensure_ascii=False, indent=2)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _public_report(run):
        report = copy.deepcopy(run.report)
        if run.key:
            for field in ('model', 'title', 'prompt', 'content', 'error', 'svg'):
                if isinstance(report.get(field), str):
                    report[field] = report[field].replace(run.key, '[REDACTED]')
        return report

    def _persist(self, run):
        self._save(self._public_report(run))

    def start(self, settings, max_tokens=MAX_TOKENS, thinking_mode='default'):
        if type(max_tokens) is not int or not 1024 <= max_tokens <= 262144:
            raise ValueError('生成额度必须是 1024–262144 之间的整数')
        if thinking_mode not in ('default', 'disabled'):
            raise ValueError('思考模式必须是默认或关闭思考')
        if not isinstance(settings, dict) or not all(isinstance(settings.get(k), str) and settings[k].strip()
                                                   for k in ('model', 'base_url', 'api_key')):
            raise ValueError('请先保存模型接口、模型名和密钥')
        parsed = urlsplit(settings['base_url'])
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('模型接口地址必须是有效的 HTTP 或 HTTPS 地址')
        report = dict(
            id=datetime.now().strftime('%Y%m%d_%H%M%S') + '_pelican_' + secrets.token_hex(4),
            kind='pelican', status='running', model=settings['model'],
            title=settings['model'] + ' · 鹈鹕骑自行车',
            created_at=datetime.now().astimezone().isoformat(timespec='seconds'),
            finished_at=None, prompt=PROMPT, source=SOURCE, temperature=0.2,
            max_tokens=max_tokens, thinking_mode=thinking_mode,
            request_stream=True, streamed=False, phase='connecting',
            content='', svg='', reasoning_chars=0, received_events=0,
            first_response_seconds=None, first_content_seconds=None,
            elapsed_seconds=0, usage=None, finish_reason=None,
            error=None, error_kind=None, http_status=None,
        )
        run = _PelicanRun(report, settings['api_key'])
        with self.lock:
            if self.active_id:
                raise ValueError('鹈鹕测试正在生成或停止，请等待任务结束')
            self._persist(run)
            self.active_id = report['id']
            self.active_run = run
            self.last_id = report['id']
        initial = self._public_report(run)
        try:
            threading.Thread(target=self._run, args=(report, dict(settings)), daemon=True, name='pelican-eval').start()
        except Exception:
            with run.lock:
                report.update(status='failed', phase='failed', error_kind='task_start_error',
                              error='无法启动生成任务，请重试', finished_at=datetime.now().astimezone().isoformat(timespec='seconds'))
                self._persist(run)
            with self.lock:
                if self.active_run is run:
                    self.active_id = self.active_run = None
            raise
        return initial

    def stop(self):
        with self.lock:
            run = self.active_run
            last_id = self.last_id
        if run is None:
            if last_id:
                try:
                    return json.loads((self.reports / (last_id + '.json')).read_text(encoding='utf-8'))
                except (ValueError, OSError):
                    pass
            return {'status': 'idle'}
        with run.lock:
            if run.report['status'] == 'running':
                run.cancel.set()
                run.report.update(status='cancelled', phase='cancelled', error_kind='cancelled',
                                  error=None, elapsed_seconds=round(time.monotonic() - run.started, 2),
                                  finished_at=datetime.now().astimezone().isoformat(timespec='seconds'))
                self._persist(run)
            result = self._public_report(run)
        run.abort()
        # The worker, rather than this method, releases the slot after cleanup.
        return result

    @staticmethod
    def _check_deadline(run):
        if run.cancel.is_set():
            raise _PelicanCancelled()
        remaining = run.timeout_seconds - (time.monotonic() - run.started)
        if run.timed_out.is_set() or remaining <= 0:
            run.timed_out.set()
            raise TimeoutError()
        sock = run.socket or (run.connection.sock if run.connection else None)
        # HTTPResponse can close the socket while returning its final EOF.
        # Let the parser inspect EOF rather than turning it into EBADF.
        if sock and sock.fileno() != -1:
            sock.settimeout(remaining)
        return remaining

    def _monitor(self, run):
        while not run.done.is_set():
            remaining = run.timeout_seconds - (time.monotonic() - run.started)
            if remaining <= 0:
                run.timed_out.set()
                run.abort()
                return
            if run.done.wait(min(1, remaining)):
                return
            with run.lock:
                if run.report['status'] == 'running':
                    run.report['elapsed_seconds'] = round(time.monotonic() - run.started, 2)
                    self._persist(run)

    def _run(self, report, settings):
        with self.lock:
            run = self.active_run if self.active_run and self.active_run.report['id'] == report['id'] else None
        if run is None:
            run = _PelicanRun(report, settings.get('api_key', ''))
        else:
            # Direct callers/tests can pass the public initial report.
            report = run.report
        response = None
        refused = False
        try:
            threading.Thread(target=self._monitor, args=(run,), daemon=True, name='pelican-progress').start()
            parsed = urlsplit(settings['base_url'].rstrip('/'))
            url = parsed.path if parsed.path.endswith('/chat/completions') else parsed.path + '/chat/completions'
            payload = {'model': settings['model'], 'messages': [{'role': 'user', 'content': report['prompt']}],
                       'temperature': report['temperature'], 'max_tokens': report['max_tokens'],
                       'stream': True, 'stream_options': {'include_usage': True}}
            if report['thinking_mode'] == 'disabled':
                payload['chat_template_kwargs'] = {'enable_thinking': False}
            connection_class = http.client.HTTPSConnection if parsed.scheme == 'https' else http.client.HTTPConnection
            with run.lock:
                run.connection = connection_class(parsed.hostname, parsed.port, timeout=self._check_deadline(run))
            run.connection.request('POST', url, body=json.dumps(payload).encode(),
                                   headers={'Content-Type': 'application/json', 'Accept': 'text/event-stream',
                                            'Authorization': 'Bearer ' + settings['api_key']})
            run.socket = run.connection.sock
            self._check_deadline(run)
            response = run.connection.getresponse()
            report['http_status'] = response.status
            if not 200 <= response.status <= 299:
                raise _PelicanHTTPError(response.status)
            content_type = response.getheader('Content-Type') or ''
            streamed = isinstance(content_type, str) and 'text/event-stream' in content_type.lower()
            with run.lock:
                report['streamed'] = streamed

            def process_event(data):
                nonlocal refused
                if not isinstance(data, dict) or data.get('error'):
                    raise _PelicanProtocolError()
                choices = data.get('choices')
                if not isinstance(choices, list):
                    raise _PelicanProtocolError()
                at = time.monotonic()
                with run.lock:
                    self._check_deadline(run)
                    usage = _safe_usage(data.get('usage'))
                    if usage:
                        report['usage'] = dict(report.get('usage') or {}, **usage)
                    report['received_events'] += 1
                    if not choices:
                        return
                    choice = choices[0]
                    if not isinstance(choice, dict):
                        raise _PelicanProtocolError()
                    finish = choice.get('finish_reason')
                    if finish in ('stop', 'length', 'content_filter', 'tool_calls', 'function_call'):
                        report['finish_reason'] = finish
                    message = choice.get('delta') if streamed else choice.get('message')
                    if not isinstance(message, dict):
                        return
                    content, reasoning, rejected = _message_text(message)
                    refused = refused or rejected
                    if len(report['content']) + len(content) > MAX_RESPONSE:
                        raise ValueError('模型最终文本超过允许长度，已保留此前收到的部分内容')
                    report['content'] += content
                    report['reasoning_chars'] += reasoning
                    if content or reasoning:
                        if report['first_response_seconds'] is None:
                            report['first_response_seconds'] = round(at - run.started, 3)
                        if content and report['first_content_seconds'] is None:
                            report['first_content_seconds'] = round(at - run.started, 3)
                        report['phase'] = 'writing' if report['content'] else 'thinking'
                        report['elapsed_seconds'] = round(at - run.started, 2)

            if streamed:
                wire_bytes = 0
                data_lines = []
                event_bytes = 0
                ended = False
                while not ended:
                    self._check_deadline(run)
                    line = response.readline(MAX_EVENT_BYTES + 1)
                    wire_bytes += len(line)
                    event_bytes += len(line)
                    if wire_bytes > MAX_WIRE_BYTES or len(line) > MAX_EVENT_BYTES or event_bytes > MAX_EVENT_BYTES:
                        raise ValueError('模型流式响应超过允许长度，已保留此前收到的部分内容')
                    if not line:
                        if data_lines:
                            event = '\n'.join(data_lines)
                            if event.strip() == '[DONE]':
                                ended = True
                            else:
                                process_event(json.loads(event))
                        break
                    line = line.decode('utf-8').rstrip('\r\n')
                    if not line:
                        if data_lines:
                            event = '\n'.join(data_lines)
                            data_lines = []
                            if event.strip() == '[DONE]':
                                ended = True
                            else:
                                process_event(json.loads(event))
                        event_bytes = 0
                    elif line.startswith('data:'):
                        data_lines.append(line[5:].lstrip(' '))
                self._check_deadline(run)
                if not ended and report['finish_reason'] is None:
                    raise _PelicanProtocolError()
            else:
                self._check_deadline(run)
                body = response.read(MAX_WIRE_BYTES + 1)
                self._check_deadline(run)
                if len(body) > MAX_WIRE_BYTES:
                    raise ValueError('模型响应过大')
                process_event(json.loads(body))
            with run.lock:
                self._check_deadline(run)
                if not report['content'].strip():
                    if report['finish_reason'] == 'length':
                        report['error_kind'] = 'budget_exhausted'
                        raise ValueError(f"模型已用完 {report['max_tokens']} tokens 的生成额度，尚未返回最终 SVG；请降低模型服务的思考预算或提高生成额度后重试")
                    if refused or report['finish_reason'] == 'content_filter':
                        report['error_kind'] = 'refusal'
                        raise ValueError('模型拒绝了本次绘图请求，未返回 SVG')
                    if report['reasoning_chars']:
                        report['error_kind'] = 'reasoning_only'
                        raise ValueError('接口只返回了思考内容，没有最终回答；请检查模型服务的思考预算和输出配置')
                    report['error_kind'] = 'empty_output'
                    raise ValueError('模型接口返回空文本，未生成 SVG；请检查模型服务后重试')
                report['error_kind'] = 'invalid_svg'
                svg, removed = sanitize_svg(report['content'])
                report.update(svg=svg, animated=svg_has_animation(svg), preview_modified=bool(removed),
                              status='completed', phase='completed', error=None, error_kind=None)
        except _PelicanHTTPError as error:
            with run.lock:
                if run.cancel.is_set():
                    report.update(status='cancelled', phase='cancelled', error_kind='cancelled', error=None)
                else:
                    report.update(status='failed', phase='failed', error_kind='http_error', http_status=error.status,
                                  error=http_error_message(error.status, report['thinking_mode']))
        except _PelicanCancelled:
            with run.lock:
                report.update(status='cancelled', phase='cancelled', error_kind='cancelled', error=None)
        except Exception as error:
            with run.lock:
                if run.cancel.is_set():
                    report.update(status='cancelled', phase='cancelled', error_kind='cancelled', error=None)
                elif run.timed_out.is_set() or isinstance(error, (TimeoutError, socket.timeout)):
                    report.update(status='failed', phase='failed', error_kind='timeout',
                                  error=f'生成超过 {run.timeout_seconds:g} 秒的总时限；已保留收到的正文，请降低思考预算或生成额度后重试')
                elif isinstance(error, (_PelicanProtocolError, json.JSONDecodeError, UnicodeError, KeyError, TypeError)):
                    report.update(status='failed', phase='failed', error_kind='invalid_response',
                                  error='模型流式响应格式错误或在结束前中断；已保留收到的正文，请检查服务端超时和日志')
                elif isinstance(error, ValueError):
                    report.update(status='failed', phase='failed', error_kind=report['error_kind'] or 'invalid_output', error=str(error))
                elif isinstance(error, (OSError, http.client.HTTPException)):
                    report.update(status='failed', phase='failed', error_kind='network_error',
                                  error='连接失败或生成流中断；已保留收到的正文，请检查模型服务与网关日志')
                else:
                    report.update(status='failed', phase='failed', error_kind='generation_error', error='生成出现异常，请重试')
        finally:
            run.done.set()
            run.abort()
            if response:
                try:
                    response.close()
                except OSError:
                    pass
            try:
                with run.lock:
                    report['elapsed_seconds'] = round(time.monotonic() - run.started, 2)
                    report['finished_at'] = report['finished_at'] or datetime.now().astimezone().isoformat(timespec='seconds')
                    self._persist(run)
            finally:
                with self.lock:
                    if self.active_run is run:
                        self.active_id = self.active_run = None

    def recover_interrupted(self):
        latest = None
        for path in self.reports.glob('*_pelican_*.json'):
            try:
                report = json.loads(path.read_text(encoding='utf-8'))
                if report.get('status') == 'running':
                    report.update(status='failed', phase='failed', error_kind='interrupted',
                                  error='服务已重启，生成中断，已保留收到的正文，请重新测试',
                                  finished_at=datetime.now().astimezone().isoformat(timespec='seconds'))
                    self._save(report)
                if latest is None or report['id'] > latest:
                    latest = report['id']
            except (ValueError, OSError, KeyError, TypeError):
                continue
        self.last_id = latest
