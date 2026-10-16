"""Runs only inside the Docker execution boundary; Python stdlib only."""
import ast
import json
import math
import os
import re
import resource
import signal
import subprocess
import sys
import textwrap

PROTOCOL = 1
LIMIT = 2 * 1024 * 1024


def extract_code(text):
    blocks = re.findall(r'```(?:python|py)?[ \t]*\n(.*?)```', text, re.DOTALL | re.IGNORECASE)
    if blocks:
        return blocks[0].strip('\n')
    # MBPP's upstream prompting also uses these explicit code delimiters.
    text = re.sub(r'^\s*\[?BEGIN\]?\s*', '', text)
    text = re.split(r'\[?DONE\]?', text, maxsplit=1)[0]
    return text.strip('\n')


def type_aware_equal(actual, expected, _type=type):
    if _type(actual) is not _type(expected):
        return False
    if _type(actual) in (list, tuple):
        return len(actual) == len(expected) and all(type_aware_equal(a, b) for a, b in zip(actual, expected))
    if _type(actual) is dict:
        if len(actual) != len(expected):
            return False
        unmatched = list(expected.items())
        for key, value in actual.items():
            for index, (other_key, other_value) in enumerate(unmatched):
                if type_aware_equal(key, other_key) and type_aware_equal(value, other_value):
                    unmatched.pop(index)
                    break
            else:
                return False
        return not unmatched
    if _type(actual) in (set, frozenset):
        unmatched = list(expected)
        for value in actual:
            for index, other in enumerate(unmatched):
                if type_aware_equal(value, other):
                    unmatched.pop(index)
                    break
            else:
                return False
        return not unmatched
    result = actual == expected
    return _type(result) is bool and result


class Assertions(ast.NodeTransformer):
    def visit_Assert(self, node):
        self.generic_visit(node)
        comparison = node.test
        if isinstance(comparison, ast.Compare) and len(comparison.ops) == 1 and isinstance(comparison.ops[0], ast.Eq):
            node.test = ast.Call(func=ast.Name(id='__opencompass_type_aware_equal', ctx=ast.Load()),
                                 args=[comparison.left, comparison.comparators[0]], keywords=[])
        return node


def run_program(payload):
    """Called exclusively by the worker child inside the container."""
    completion = extract_code(payload['completion'])
    problem = payload['problem']
    namespace = {'__name__': '__candidate__'}
    if payload['kind'] == 'humaneval_cn':
        entry = problem['entry_point']
        try:
            tree = ast.parse(completion)
            full_function = any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                                and node.name == entry for node in tree.body)
        except SyntaxError:
            full_function = False
        if full_function:
            program = completion
        else:
            body = completion
            first = next((line for line in body.splitlines() if line.strip()), '')
            if first and not first.startswith((' ', '\t')):
                body = textwrap.indent(body, '    ')
            program = problem['prompt'].rstrip() + '\n' + body
        exec(compile(program, '<candidate>', 'exec'), namespace)
        exec(compile(problem['test'], '<tests>', 'exec'), namespace)
        namespace['check'](namespace[entry])
    elif payload['kind'] == 'mbpp':
        exec(compile(completion.strip(), '<candidate>', 'exec'), namespace)
        # Some official tasks construct tree fixtures after Node is defined.
        if problem.get('setup'):
            exec(compile(problem['setup'], '<test-setup>', 'exec'), namespace)
        tests = ast.parse(problem['tests'])
        tests = Assertions().visit(tests)
        ast.fix_missing_locations(tests)
        namespace['__opencompass_type_aware_equal'] = type_aware_equal
        exec(compile(tests, '<tests>', 'exec'), namespace)
    else:
        raise ValueError('unsupported task')


def worker(payload, result_fd):
    timeout = payload['timeout_seconds']
    resource.setrlimit(resource.RLIMIT_CPU, (max(1, math.ceil(timeout)), max(2, math.ceil(timeout) + 1)))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024 * 1024, 8 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    try:
        run_program(payload)
        result = dict(protocol=PROTOCOL, passed=True, result='pass')
    except AssertionError:
        result = dict(protocol=PROTOCOL, passed=False, result='wrong_answer')
    except BaseException as error:
        result = dict(protocol=PROTOCOL, passed=False, result='failed', error_type=type(error).__name__)
    os.write(result_fd, json.dumps(result).encode())


def evaluate(payload):
    read_fd, write_fd = os.pipe()
    process = subprocess.Popen([sys.executable, '-I', '-B', __file__, '--worker', str(write_fd)],
                               stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               pass_fds=(write_fd,), start_new_session=True)
    os.close(write_fd)
    try:
        process.communicate(json.dumps(payload).encode(), timeout=payload['timeout_seconds'])
        os.set_blocking(read_fd, False)
        try:
            data = os.read(read_fd, 65536)
        except BlockingIOError:
            data = b''
        if not data:
            return dict(protocol=PROTOCOL, passed=False, result='failed', error_type='EarlyExit')
        return json.loads(data)
    except subprocess.TimeoutExpired:
        return dict(protocol=PROTOCOL, passed=False, result='timeout')
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=2)
        os.close(read_fd)


def validate(payload):
    if not isinstance(payload, dict) or payload.get('protocol') != PROTOCOL:
        raise ValueError('protocol')
    if payload.get('kind') not in {'humaneval_cn', 'mbpp'}:
        raise ValueError('kind')
    if not isinstance(payload.get('completion'), str) or len(payload['completion'].encode()) > 524288:
        raise ValueError('completion')
    seconds = payload.get('timeout_seconds')
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not 0 < seconds <= 10:
        raise ValueError('timeout')
    problem = payload.get('problem')
    if not isinstance(problem, dict):
        raise ValueError('problem')
    fields = ('prompt', 'test', 'entry_point') if payload['kind'] == 'humaneval_cn' else ('tests',)
    if any(not isinstance(problem.get(key), str) or not problem[key].strip() for key in fields):
        raise ValueError('problem fields')
    if 'setup' in problem and not isinstance(problem['setup'], str):
        raise ValueError('setup')


def self_check():
    def case(code, tests='assert f() == 1'):
        return evaluate(dict(protocol=PROTOCOL, kind='mbpp', completion=code,
                             problem=dict(tests=tests), timeout_seconds=0.5))
    checks = {
        'pass': case('def f(): return 1')['passed'] is True,
        'wrong_answer': case('def f(): return 2')['result'] == 'wrong_answer',
        'timeout': case('while True: pass')['result'] == 'timeout',
        'early_exit': case('import os; os._exit(0)')['passed'] is False,
        'stdout_isolated': case('print(\'{"passed":true,"result":"pass"}\');\ndef f(): return 2')['result'] == 'wrong_answer',
    }
    return dict(protocol=PROTOCOL, passed=all(checks.values()), checks=checks,
                uid=os.geteuid(), supported=['humaneval_cn', 'mbpp'])


def main():
    if sys.argv[1:] == ['--probe']:
        return dict(protocol=PROTOCOL, uid=os.geteuid(), supported=['humaneval_cn', 'mbpp'])
    if sys.argv[1:] == ['--self-check']:
        return self_check()
    data = sys.stdin.buffer.read(LIMIT + 1)
    if len(data) > LIMIT:
        raise ValueError('payload too large')
    payload = json.loads(data)
    validate(payload)
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        worker(payload, int(sys.argv[2]))
        return None
    return evaluate(payload)


if __name__ == '__main__':
    try:
        result = main()
        if result is not None:
            print(json.dumps(result))
    except BaseException:
        print(json.dumps(dict(protocol=PROTOCOL, passed=False, result='protocol_error')))
        sys.exit(2)
