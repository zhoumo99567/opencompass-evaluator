#!/usr/bin/env python3
"""CLI for the local evaluator API; no browser or model credentials required.

The web service owns settings, tasks and result parsing. This client deliberately
imports no web server or model runtime and never reads credentials/config files.
"""
import argparse
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

DEFAULT_SERVER = 'http://127.0.0.1:8766'
MAX_API_BYTES = 64_000_000


class CliError(Exception):
    pass


class WaitTimeout(CliError):
    pass


class ApiClient:
    def __init__(self, server, timeout=30):
        parsed = urlsplit(server)
        try:
            parsed.port
        except ValueError:
            raise CliError('后台地址的端口无效') from None
        if (parsed.scheme not in ('http', 'https') or not parsed.hostname or
                parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise CliError('--server 必须是后台的 HTTP/HTTPS 地址，不应包含密钥、认证信息或查询参数')
        self.server = server.rstrip('/')
        self.timeout = timeout

    def request(self, route, payload=None, timeout=None):
        request = Request(self.server + route,
                          data=json.dumps(payload).encode('utf-8') if payload is not None else None,
                          headers={'Accept': 'application/json', 'Content-Type': 'application/json'},
                          method='POST' if payload is not None else 'GET')
        try:
            with urlopen(request, timeout=min(self.timeout, timeout) if timeout is not None else self.timeout) as response:
                body = response.read(MAX_API_BYTES + 1)
        except HTTPError as error:
            # Error bodies can contain upstream credentials, tracebacks or model
            # text. Print the status and a useful action, never the raw body.
            raise CliError(f'后台返回 HTTP {error.code}；请检查参数和服务状态，具体原因可在网页查看') from None
        except (URLError, TimeoutError, OSError):
            raise CliError('无法连接评测后台；请先启动 run_viewer.sh，并用 --server 指定正确地址') from None
        if len(body) > MAX_API_BYTES:
            raise CliError('后台响应过大，请缩小查询范围')
        try:
            data = json.loads(body)
        except (ValueError, UnicodeError):
            raise CliError('后台返回的内容不是有效 JSON，请核对 --server 地址') from None
        if not isinstance(data, (dict, list)):
            raise CliError('后台返回了不支持的数据格式')
        return data


def positive_number(value):
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError('必须是大于 0 的数值') from None
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('必须是大于 0 的数值')
    return number


def positive_integer(value):
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError('必须是正整数') from None
    if number <= 0:
        raise argparse.ArgumentTypeError('必须是正整数')
    return number


def integer_list(value):
    try:
        numbers = list(dict.fromkeys(int(part.strip()) for part in value.split(',') if part.strip()))
    except ValueError:
        raise argparse.ArgumentTypeError('请使用逗号分隔整数，例如 1,2,4') from None
    if not numbers or any(number <= 0 for number in numbers):
        raise argparse.ArgumentTypeError('请至少提供一个正整数')
    return numbers


def add_output(parser):
    parser.add_argument('--json', action='store_true', help='stdout 输出完整 JSON，便于脚本处理')
    parser.add_argument('--output', type=Path, help='同时保存完整 JSON 到指定文件（权限 600）')


def add_wait(parser):
    parser.add_argument('--wait', action='store_true', help='等待任务结束并返回结果；不指定时立即返回任务信息')
    parser.add_argument('--wait-timeout', type=positive_number, default=3600,
                        help='最多等待秒数，默认 3600；到期后后台任务继续运行')
    parser.add_argument('--poll-interval', type=positive_number, default=2,
                        help='状态查询间隔秒数，默认 2')


def build_parser():
    parser = argparse.ArgumentParser(
        description='通过本地评测后台从命令行启动任务、查看分数和获取报告；无需打开浏览器。',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''示例：
  ./run_cli.sh datasets
  ./run_cli.sh datasets demo_gsm8k_chat_gen
  ./run_cli.sh datasets --category math --scope full --search 推理
  ./run_cli.sh ability run --datasets demo_gsm8k_chat_gen demo_math_chat_gen --wait --json
  ./run_cli.sh results latest --json --output result.json
  ./run_cli.sh --server http://127.0.0.1:8765 reports

先运行 ./run_viewer.sh --host 127.0.0.1 --port 8766。
后台复用网页保存的模型设置与 .env；本工具不需要输入 API Key。
results/runs 指 OpenCompass 原始能力结果；reports 指已保存的 AI 分析/鹈鹕/性能报告。''')
    parser.add_argument('--server', default=os.environ.get('OPENCOMPASS_VIEWER_URL', DEFAULT_SERVER),
                        help='评测后台地址（全局参数，放在命令前）；默认 127.0.0.1:8766 或 OPENCOMPASS_VIEWER_URL')
    commands = parser.add_subparsers(dest='command', required=True)
    datasets = commands.add_parser('datasets', aliases=['list-datasets'], help='列出评测集，指定 ID 可查看用途、评分和低分排查')
    datasets.add_argument('dataset_id', nargs='?', help='可选：查看单个评测集的完整说明')
    datasets.add_argument('--available', action='store_true', help='只列出当前可运行的评测集')
    datasets.add_argument('--category', help='按 category_id 筛选，例如 math；可用分类来自后台元数据')
    datasets.add_argument('--scope', choices=['demo', 'full'], help='按演示/完整配置筛选')
    datasets.add_argument('--search', help='按名称、ID 或用途搜索，不区分英文大小写')
    add_output(datasets)
    runs = commands.add_parser('runs', help='列出原始 OpenCompass 能力评测记录')
    runs.add_argument('--completed', action='store_true', help='仅显示结果完整的记录')
    add_output(runs)
    results = commands.add_parser('results', help='查看原始分数、样本和评测方向；默认取最新完整结果')
    results.add_argument('run_id', nargs='?', default='latest', help='运行 ID，或 latest（最新已完成）')
    add_output(results)
    reports = commands.add_parser('reports', help='列出保存的报告；指定 ID 可查看具体报告，区别于原始能力结果')
    reports.add_argument('report_id', nargs='?')
    reports.add_argument('--kind', choices=['analysis', 'pelican', 'performance'], help='列表按报告类型筛选')
    add_output(reports)
    ability = commands.add_parser('ability', help='启动或查询标准 OpenCompass 能力评测')
    actions = ability.add_subparsers(dest='action', required=True)
    run = actions.add_parser('run', help='按网页相同逻辑运行一个或多个评测集')
    run.add_argument('--datasets', nargs='+', required=True, help='一个或多个评测集 ID，以空格分隔')
    run.add_argument('--max-seq-len', type=positive_integer, help='上下文总窗口 tokens，省略时采用后台默认值')
    run.add_argument('--max-out-len', type=positive_integer, help='生成额度 tokens（含思考），省略时采用后台默认值')
    add_wait(run)
    add_output(run)
    add_output(actions.add_parser('status', help='查询后台当前能力评测任务'))
    pelican = commands.add_parser('pelican', help='启动或停止鹈鹕 SVG 测试，结果也是历史报告')
    actions = pelican.add_subparsers(dest='action', required=True)
    run = actions.add_parser('run')
    run.add_argument('--max-tokens', type=positive_integer, default=32768)
    run.add_argument('--thinking-mode', choices=['default', 'disabled'], default='default')
    add_wait(run)
    add_output(run)
    add_output(actions.add_parser('stop', help='停止后台当前的鹈鹕测试'))
    performance = commands.add_parser('performance', help='启动、查询或停止端到端性能测试')
    actions = performance.add_subparsers(dest='action', required=True)
    run = actions.add_parser('run')
    run.add_argument('--concurrency', type=integer_list, default=[1, 2, 4], help='并发数，例如 1,2,4')
    run.add_argument('--input-tokens', type=integer_list, default=[1024, 4096, 8192], help='目标输入长度，例如 1024,4096')
    run.add_argument('--requests-per-cell', type=positive_integer, default=8)
    run.add_argument('--output-tokens', type=positive_integer, default=256)
    run.add_argument('--warmup', type=int, default=1)
    run.add_argument('--timeout-seconds', type=positive_integer, default=120)
    run.add_argument('--slo-ttft-ms', type=positive_number)
    run.add_argument('--slo-e2e-ms', type=positive_number)
    add_wait(run)
    add_output(run)
    add_output(actions.add_parser('status'))
    add_output(actions.add_parser('stop'))
    return parser


def _terminal_exit(data):
    return 1 if isinstance(data, dict) and data.get('status') in ('failed', 'cancelled', 'partial') else 0


def _wait_for(client, initial, route, args, task_field=None):
    deadline = time.monotonic() + args.wait_timeout
    state = initial
    while state.get('status') == 'running':
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise WaitTimeout('等待超过 --wait-timeout；后台任务仍在运行，可继续查询状态和结果')
        # Keep stdout exclusively for the requested final JSON/result.
        progress = state.get('progress') or {}
        count = f" {progress['completed']}/{progress['total']}" if 'total' in progress else ''
        print('任务运行中' + count, file=sys.stderr)
        time.sleep(min(args.poll_interval, remaining))
        if time.monotonic() >= deadline:
            raise WaitTimeout('等待超过 --wait-timeout；后台任务仍在运行，可继续查询状态和结果')
        try:
            state = client.request(route, timeout=max(.001, deadline - time.monotonic()))
        except CliError:
            if time.monotonic() >= deadline:
                raise WaitTimeout('等待超过 --wait-timeout；后台任务仍在运行，可继续查询状态和结果') from None
            raise
        if task_field and initial.get(task_field) and state.get(task_field) != initial[task_field]:
            raise CliError('后台当前任务已变化或服务已重启；请使用 runs/results 或 reports 查询原任务，避免混合结果')
    return state


def execute(args, client):
    command = args.command
    if command in ('datasets', 'list-datasets'):
        data = client.request('/api/datasets')
        if args.dataset_id:
            selected = next((item for item in data if item['id'] == args.dataset_id), None)
            if selected is None:
                raise CliError('未找到此评测集；运行 datasets 查看可用 ID')
            data = [selected]
        if args.category:
            data = [item for item in data if item.get('category_id') == args.category]
        if args.scope:
            data = [item for item in data if item.get('scope') == args.scope]
        if args.search:
            search = args.search.strip().casefold()
            data = [item for item in data if search in ' '.join(str(item.get(key) or '')
                    for key in ('label', 'id', 'purpose')).casefold()]
        if args.available:
            data = [item for item in data if item.get('available')]
        if args.dataset_id:
            if not data:
                raise CliError('此评测集不符合所指定的分类、范围、搜索或可用性筛选；请移除筛选条件后查看单项详情')
            return data[0], 0
        if not data and (args.category or args.scope or args.search or args.available):
            print('没有符合筛选条件的评测集；可移除筛选，或查询 /api/benchmark-metadata 的分类 ID', file=sys.stderr)
        return data, 0
    if command == 'runs':
        data = client.request('/api/runs')
        return ([item for item in data if item.get('status') == 'completed'] if args.completed else data), 0
    if command == 'results':
        run_id = args.run_id
        if run_id == 'latest':
            rows = client.request('/api/runs')
            selected = next((row for row in rows if row.get('status') == 'completed'), None)
            if not selected:
                raise CliError('还没有结果完整的能力评测；运行 runs 查看已有任务')
            run_id = selected['id']
        data = client.request('/api/runs/' + quote(run_id, safe=''))
        return data, _terminal_exit(data)
    if command == 'reports':
        if args.report_id:
            data = client.request('/api/reports/' + quote(args.report_id, safe=''))
            return data, _terminal_exit(data)
        rows = client.request('/api/reports')
        if args.kind:
            rows = [item for item in rows if (item.get('kind') or 'analysis') == args.kind]
        return rows, 0
    if command == 'ability':
        if args.action == 'status':
            data = client.request('/api/evaluation/status')
            return data, _terminal_exit(data)
        payload = {'datasets': list(dict.fromkeys(args.datasets))}
        options = {key: getattr(args, key) for key in ('max_seq_len', 'max_out_len') if getattr(args, key) is not None}
        if options:
            payload['options'] = options
        data = client.request('/api/evaluation/start', payload)
        if not args.wait:
            return data, _terminal_exit(data)
        data = _wait_for(client, data, '/api/evaluation/status', args, task_field='log_path')
        run_ids = data.get('run_ids') or []
        results = [client.request('/api/runs/' + quote(run_id, safe='')) for run_id in run_ids]
        output = {'evaluation': data, 'runs': results}
        if 'run_ids' not in data:
            output['warning'] = '后台版本未提供任务对应的 run_ids；请更新后台，或用 runs/results 按运行 ID 查询'
        code = 0 if data.get('status') == 'completed' else 1
        return output, max([code] + [_terminal_exit(result) for result in results])
    if command == 'pelican':
        if args.action == 'stop':
            data = client.request('/api/pelican/stop', {})
        else:
            data = client.request('/api/pelican/start', {'max_tokens': args.max_tokens, 'thinking_mode': args.thinking_mode})
            if args.wait:
                data = _wait_for(client, data, '/api/reports/' + quote(data['id'], safe=''), args, task_field='id')
        return data, _terminal_exit(data)
    if command == 'performance':
        if args.action == 'run':
            payload = {key: getattr(args, key) for key in ('concurrency', 'input_tokens', 'requests_per_cell',
                       'output_tokens', 'warmup', 'timeout_seconds', 'slo_ttft_ms', 'slo_e2e_ms')}
            data = client.request('/api/performance/start', payload)
            if args.wait:
                data = _wait_for(client, data, '/api/performance/status', args, task_field='id')
        else:
            data = client.request('/api/performance/' + args.action, {} if args.action == 'stop' else None)
        return data, _terminal_exit(data)
    raise CliError('不支持的命令')


def _human_lines(data):
    if isinstance(data, list):
        if not data:
            return ['暂无记录。']
        lines = []
        for item in data:
            line = str(item.get('id', ''))
            label = item.get('label') or item.get('title') or item.get('model') or ''
            if label:
                line += '  ' + str(label)
            if item.get('category_id'):
                line += '  [' + str(item.get('category') or item['category_id']) + ' (' + item['category_id'] + ') / ' + str(item.get('scope') or '') + ']'
            if 'available' in item:
                line += '  ' + ('可运行' if item['available'] else '不可运行')
            elif item.get('status'):
                line += '  ' + str(item['status'])
            lines.append(line)
            if item.get('purpose'):
                lines.append('  测什么：' + item['purpose'])
            elif item.get('focuses'):
                lines.append('  主要测试：' + '、'.join(item['focuses']))
            if item.get('suitable_uses'):
                lines.append('  适合：' + '、'.join(item['suitable_uses']))
        return lines
    if 'evaluation' in data:
        lines = ['能力评测状态：' + str(data['evaluation'].get('status', ''))]
        for result in data.get('runs', []):
            lines.extend(_human_lines(result))
        if data.get('warning'):
            lines.append(data['warning'])
        return lines
    if 'focuses' in data and 'id' in data and 'available' in data:
        lines = [f"{data.get('label', '')} · {data['id']}",
                 '测什么：' + str(data.get('purpose') or data.get('description') or ''),
                 '适合：' + '、'.join(data.get('suitable_uses') or []),
                 '题型示例（示意）：' + str(data.get('task_example') or ''),
                 '主要测试：' + '、'.join(data.get('focuses') or []),
                 '评分含义：' + str(data.get('metric_explanation') or data.get('metric_meaning') or ''),
                 '低分排查候选：' + '；'.join(data.get('low_score_checks') or data.get('low_score_reasons') or []),
                 '可能影响：' + '、'.join(data.get('impacts') or []),
                 '局限：' + str(data.get('limitations') or '')]
        if data.get('category_id'):
            lines.append('分类：' + str(data.get('category') or data['category_id']) + '（' + data['category_id'] + '） / ' + str(data.get('scope') or ''))
        if data.get('unsupported_reason'):
            lines.append('当前不可运行：' + data['unsupported_reason'])
        if data.get('preparation_note'):
            lines.append('数据准备：' + data['preparation_note'])
        return lines
    lines = [str(data.get('id') or '当前任务') + '  ' + str(data.get('status') or data.get('kind') or '')]
    for score in data.get('scores', []):
        lines.append(f"  {score.get('dataset')} / {score.get('metric')}: {score.get('score')} ({score.get('model', '')})")
    for interpretation in data.get('score_interpretations', []):
        lines.append('  ' + str(interpretation.get('dataset', '')) + ' 测什么：' +
                     str(interpretation.get('purpose') or '、'.join(interpretation.get('focuses') or [])))
        if interpretation.get('suitable_uses'):
            lines.append('    适合：' + '、'.join(interpretation['suitable_uses']))
    if data.get('run_ids'):
        lines.append('运行 ID：' + ', '.join(data['run_ids']))
    if data.get('error'):
        lines.append('错误：' + str(data['error']))
    if data.get('kind') in ('pelican', 'performance'):
        lines.append('完整生成内容、逐请求指标请加 --json 或 --output。')
    return lines


def emit(data, args):
    serialized = json.dumps(data, ensure_ascii=False, indent=2)
    if args.output:
        destination = args.output.expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=destination.parent,
                                             prefix='.' + destination.name + '.', delete=False) as output:
                temporary = Path(output.name)
                output.write(serialized + '\n')
            temporary.replace(destination)
        finally:
            if temporary:
                temporary.unlink(missing_ok=True)
    print(serialized if args.json else '\n'.join(_human_lines(data)))
    if args.output and not args.json:
        print('已保存完整 JSON：' + str(args.output.expanduser().resolve()))


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        data, code = execute(args, ApiClient(args.server))
        emit(data, args)
        return code
    except WaitTimeout as error:
        print('等待超时：' + str(error), file=sys.stderr)
        return 124
    except KeyboardInterrupt:
        print('已停止等待；后台任务继续运行。可查询 status/results，或用 pelican/performance stop 停止对应任务。', file=sys.stderr)
        return 130
    except CliError as error:
        print('错误：' + str(error), file=sys.stderr)
        return 2
    except OSError:
        print('错误：无法保存输出文件，请检查路径和文件权限', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
