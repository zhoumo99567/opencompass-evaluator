#!/usr/bin/env python3
"""Small local web UI for browsing OpenCompass output directories."""

from __future__ import annotations

import argparse
import importlib.util
import ipaddress
import csv
import json
import re
import os
import secrets
import subprocess
import threading
from benchmark_catalog import (BENCHMARK_CATEGORIES, BENCHMARK_PRESETS, DATASET_CATALOG,
                               dataset_preparation, describe_benchmark, unavailable_reason)
from evaluation_config import build_evaluation_config
from environment_manager import EnvironmentManager
from performance_eval import PerformanceEvaluator, report_to_csv
from pelican_eval import MAX_TOKENS, PelicanEvaluator, http_error_message, sanitize_svg, svg_has_animation
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, unquote, urlparse
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
REPORTS = ROOT / "reports"
PELICAN = PelicanEvaluator(REPORTS)
PERFORMANCE = PerformanceEvaluator(REPORTS)
ENVIRONMENT = EnvironmentManager(ROOT)
ACTIVITY_LOCK = threading.Lock()
SETTINGS_PATH = ROOT / ".viewer_settings.json"
COLLECTIONS_PATH = ROOT / ".dataset_collections.json"
EVAL_LOCK = threading.Lock()
EVAL_STATE = {
    "status": "idle",
    "pid": None,
    "datasets": [],
    "started_at": None,
    "finished_at": None,
    "returncode": None,
    "log_path": None,
    "process": None,
}

# Dataset metadata lives alongside verified installed OpenCompass configuration IDs.

def _local_environment_request(client_address, headers, port):
    """Only this local JSON UI may invoke the fixed environment actions."""
    try:
        if not ipaddress.ip_address(client_address[0]).is_loopback:
            return False
        host = urlparse('http://' + headers.get('Host', ''))
        if host.hostname not in ('localhost', '127.0.0.1', '::1') or (host.port or 80) != port:
            return False
        if headers.get('Content-Type', '').split(';', 1)[0].strip().lower() != 'application/json':
            return False
        if not 0 <= int(headers.get('Content-Length', '0')) <= 8192:
            return False
        origin = headers.get('Origin')
        if origin:
            parsed = urlparse(origin)
            if (parsed.scheme, parsed.hostname, parsed.port or 80) != ('http', host.hostname, port):
                return False
        return True
    except (ValueError, TypeError):
        return False


def _start_activity(start):
    with ACTIVITY_LOCK:
        if ENVIRONMENT.status().get('busy'):
            raise ValueError('环境任务正在运行，请完成后再开始评测')
        return start()


def _start_environment(action):
    if action not in ('start_runtime', 'install_code', 'check'):
        raise ValueError('未知环境操作')
    with ACTIVITY_LOCK:
        with PELICAN.lock:
            pelican_running = PELICAN.active_id is not None
        if (_eval_public_state().get('status') == 'running' or
                pelican_running or
                PERFORMANCE.status().get('status') == 'running'):
            raise ValueError('有评测正在运行，请完成或停止后再准备环境')
        return ENVIRONMENT.start(action)

def _env_file_values():
    values = {}
    path = ROOT / ".env"
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def load_settings():
    env = _env_file_values()
    settings = {
        "base_url": env.get("OPENCOMPASS_BASE_URL", os.environ.get("OPENCOMPASS_BASE_URL", "")),
        "model": env.get("OPENCOMPASS_MODEL", os.environ.get("OPENCOMPASS_MODEL", "")),
        "api_key": env.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
    }
    saved = read_json(SETTINGS_PATH)
    if isinstance(saved, dict):
        for key in ("base_url", "model", "api_key"):
            if key in saved:
                settings[key] = str(saved[key] or "")
    return settings


def public_settings(settings=None):
    settings = settings or load_settings()
    key = settings.get("api_key", "")
    return {
        "base_url": settings.get("base_url", ""),
        "model": settings.get("model", ""),
        "api_key_set": bool(key),
        "api_key_hint": f"••••{key[-4:]}" if key else "",
    }


def save_settings(payload):
    current = load_settings()
    for key in ("base_url", "model"):
        if key in payload:
            current[key] = str(payload.get(key) or "").strip()
    if str(payload.get("api_key") or "").strip():
        current["api_key"] = str(payload["api_key"]).strip()
    SETTINGS_PATH.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        SETTINGS_PATH.chmod(0o600)
    except OSError:
        pass
    return public_settings(current)


def dataset_catalog():
    """Return the curated picker entries whose configs exist in this install."""
    config_root = ROOT / ".venv" / "lib"
    # The exact Python minor version is part of the venv path, so search once
    # rather than assuming a particular interpreter directory.
    matches = list(config_root.glob("python*/site-packages/opencompass/configs/datasets")) if config_root.exists() else []
    available = set()
    if matches:
        dataset_root = matches[0]
        for path in dataset_root.rglob("*.py"):
            available.add(path.stem)
    catalog = []
    for item in DATASET_CATALOG:
        entry = dict(item)
        reason = unavailable_reason(entry, ROOT)
        if entry['id'] not in available:
            reason = reason or '当前 OpenCompass 安装未包含此评测配置'
        entry['unsupported_reason'] = reason
        entry['available'] = not reason
        entry.update(dataset_preparation(entry, ROOT))
        catalog.append(entry)
    return catalog


def load_collections():
    payload = read_json(COLLECTIONS_PATH)
    if not isinstance(payload, list):
        return []
    catalog_ids = {item["id"] for item in DATASET_CATALOG}
    result = []
    for item in payload:
        if not isinstance(item, dict) or not str(item.get("name", "")).strip():
            continue
        datasets = [str(value) for value in item.get("datasets", []) if str(value) in catalog_ids]
        result.append({
            "id": str(item.get("id") or secrets.token_hex(4)),
            "name": str(item["name"]).strip(),
            "description": str(item.get("description") or "").strip(),
            "datasets": datasets,
        })
    return result


def save_collection(payload):
    name = str(payload.get("name") or "").strip()
    datasets = [str(value) for value in payload.get("datasets", [])]
    valid_ids = {item["id"] for item in DATASET_CATALOG}
    datasets = list(dict.fromkeys(value for value in datasets if value in valid_ids))
    if not name:
        raise ValueError("集合名称不能为空")
    if not datasets:
        raise ValueError("请至少选择一个评测集")
    collections = load_collections()
    collection = {
        "id": secrets.token_hex(4),
        "name": name[:80],
        "description": str(payload.get("description") or "").strip()[:240],
        "datasets": datasets,
    }
    collections = [item for item in collections if item["name"] != collection["name"]]
    collections.insert(0, collection)
    COLLECTIONS_PATH.write_text(json.dumps(collections, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        COLLECTIONS_PATH.chmod(0o600)
    except OSError:
        pass
    return collection


def _eval_public_state():
    with EVAL_LOCK:
        process = EVAL_STATE.get("process")
        if process is not None and process.poll() is not None:
            if EVAL_STATE["status"] == "running":
                EVAL_STATE["returncode"] = process.returncode
                EVAL_STATE["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
                EVAL_STATE["status"] = "completed" if process.returncode == 0 else "failed"
                work_dir = Path(EVAL_STATE['work_dir'])
                manifest = read_json(work_dir / 'evaluation_manifest.json') or []
                expected = {abbr for item in manifest for abbr in item['abbrs']}
                finished = {path.stem for path in work_dir.glob('*/results/**/*.json')
                            if isinstance(read_json(path), dict) and not read_json(path).get('error')}
                EVAL_STATE['completed_count'] = len(expected & finished)
                if process.returncode == 0 and expected - finished:
                    EVAL_STATE['status'] = 'failed'
                    EVAL_STATE['error'] = f"仅完成 {len(expected & finished)}/{len(expected)} 个子任务，请查看日志"
            EVAL_STATE["process"] = None
        public = {key: value for key, value in EVAL_STATE.items()
                  if key not in ("process", "work_dir")}
        work_dir = EVAL_STATE.get('work_dir')
        public['run_ids'] = sorted(
            (path.name for path in Path(work_dir).glob('*')
             if path.is_dir() and re.fullmatch(r'\d{8}_\d{6}', path.name)),
            reverse=True,
        ) if work_dir else []
        return public


def start_evaluation(dataset_ids, options=None):
    if not isinstance(dataset_ids, list) or any(not isinstance(x, str) for x in dataset_ids):
        raise ValueError("评测集必须是名称列表")
    options = {'max_seq_len': 262144, 'max_out_len': 32768} if options is None else options
    if not isinstance(options, dict):
        raise ValueError('能力评测参数必须为对象')
    options = {key: options.get(key, default) for key, default in
               (('max_seq_len', 262144), ('max_out_len', 32768))}
    if (type(options['max_seq_len']) is not int or not 128 <= options['max_seq_len'] <= 262144 or
            type(options['max_out_len']) is not int or not 64 <= options['max_out_len'] <= 65536 or
            options['max_out_len'] >= options['max_seq_len']):
        raise ValueError('输入窗口需为 128–262144，生成额度需为 64–65536，且生成额度小于窗口')
    selected = list(dict.fromkeys(dataset_ids))
    catalog = {item["id"]: item for item in dataset_catalog()}
    for value in selected:
        if value not in catalog:
            raise ValueError(f"未知评测集：{value}")
        if not catalog[value]['available']:
            raise ValueError(catalog[value].get('unsupported_reason') or f"评测集未安装：{value}")
    if not selected:
        raise ValueError("请至少选择一个可用评测集")
    with EVAL_LOCK:
        current = EVAL_STATE.get("process")
        if current is not None and current.poll() is None:
            raise ValueError("已有评测正在运行，请等待完成")
        settings = load_settings()
        if not settings["base_url"] or not settings["model"]:
            raise ValueError("请先在模型设置中填写接口地址和模型名称")
        if not settings["api_key"]:
            raise ValueError("请先在模型设置中填写 API Key")
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(3)
        log_path = OUTPUTS / f"evaluation_{stamp}.log"
        work_dir = OUTPUTS / 'web' / f'job_{stamp}'
        config_path = work_dir / 'evaluation_config.py'
        manifest = build_evaluation_config(ROOT, selected, settings, config_path, options=options)
        (work_dir / 'evaluation_parameters.json').write_text(
            json.dumps(options, ensure_ascii=False, indent=2), encoding='utf-8')
        (work_dir / 'evaluation_manifest.json').write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        command = [
            str(ROOT / '.venv' / 'bin' / 'opencompass'),
            str(config_path), '-w', str(work_dir),
        ]
        environment = os.environ.copy()
        environment.update({
            "OPENCOMPASS_BASE_URL": chat_completions_url(settings["base_url"]),
            "OPENCOMPASS_MODEL": settings["model"],
            "OPENAI_API_KEY": settings["api_key"],
            "PYTHONPATH": os.pathsep.join(filter(None, (str(ROOT), environment.get('PYTHONPATH')))),
        })
        log_handle = log_path.open("w", encoding="utf-8")
        try:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError:
            log_handle.close()
            raise
        # The child owns the file descriptor after Popen; closing it in the
        # server keeps the viewer from holding the log open unnecessarily.
        log_handle.close()
        EVAL_STATE.update({
            "status": "running",
            "pid": process.pid,
            "datasets": selected,
            "generation_parameters": options,
            "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "finished_at": None,
            "returncode": None,
            "error": None,
            "completed_count": 0,
            "log_path": os.path.relpath(log_path, ROOT),
            "work_dir": str(work_dir),
            "dataset_count": sum(len(item["abbrs"]) for item in manifest),
            "process": process,
        })
    return _eval_public_state()


def chat_completions_url(base_url):
    base_url = (base_url or "").strip().rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    return base_url + "/chat/completions"


def generate_report(run_id, instruction=""):
    settings = load_settings()
    if not settings["base_url"] or not settings["model"]:
        raise ValueError("请先在 Settings 中填写接口地址和模型名称")
    if not settings["api_key"]:
        raise ValueError("请先在 Settings 中填写 API Key")
    run = run_detail(run_id)
    if not run:
        raise ValueError("评测任务不存在")
    score_lines = "\n".join(
        f"- {item['dataset']} / {item['metric']}: {item['score']}"
        for item in run.get("scores", [])
    ) or "- 当前只有预测记录，尚未生成 summary 分数"
    sample_lines = "\n".join(
        f"样本 {item['idx']} | 问题：{item['prompt']} | 模型输出：{item['prediction']} | 标准答案：{item['gold']}"
        for item in run.get("samples", [])[:18]
    ) or "暂无预测样本"
    definitions = json.dumps(run.get('score_interpretations', []), ensure_ascii=False)
    extra = instruction.strip() or "请重点说明模型表现、明显错误模式和下一步优化建议。"
    prompt = f"""你是一名 LLM 评测分析师。请根据下面的 OpenCompass 评测数据生成一份中文 Markdown 报告。

评测任务：{run['id']}
模型：{run['model']}
预览样本数：{len(run['samples'])}

分数：
{score_lines}

数据集任务、评分规则与排查候选（候选不等于已证实原因）：
{definitions}

样本摘录：
{sample_lines}

用户补充要求：{extra}

报告请包含：结论摘要、逐数据集评测方向与指标解读、低分排查与证据、可能影响的使用场景、代表性错误样本、局限、可执行的优化建议。区分观察到的错误与待验证的原因，不把不同数据集的均分视作统一能力分。不要编造数据；没有证据时明确写“数据不足”。"""
    payload = {
        "model": settings["model"],
        "messages": [
            {"role": "system", "content": "你输出严谨、简洁、可执行的中文评测报告。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {settings['api_key']}"}
    request = Request(chat_completions_url(settings["base_url"]), data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with urlopen(request, timeout=180) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"模型接口返回 HTTP {error.code}: {detail}") from error
    except (URLError, TimeoutError, ValueError) as error:
        raise RuntimeError(f"请求模型接口失败：{error}") from error
    try:
        content = response_data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError("模型返回格式不是 OpenAI Chat Completions 格式") from error
    created = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    report_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(3)}"
    report = {
        "id": report_id,
        "created_at": created,
        "run_id": run["id"],
        "model": run["model"],
        "title": f"{run['model']} · {run['id']}",
        "content": str(content),
        "scores": run.get("scores", []),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"{report_id}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def discover_reports():
    if not REPORTS.exists():
        return []
    reports = []
    for path in sorted(REPORTS.glob("*.json"), reverse=True):
        report = report_detail(path.stem)
        if isinstance(report, dict) and report.get("id"):
            reports.append({key: report.get(key, "") for key in ("id", "created_at", "run_id", "model", "title", "kind", "status", "elapsed_seconds", "animated")})
    return reports


def report_detail(report_id):
    safe = Path(report_id)
    if safe.name != report_id or ".." in safe.parts:
        return None
    report = read_json(REPORTS / f"{report_id}.json")
    if report and report.get('kind') == 'pelican':
        # Explain old HTTP failures accurately without rewriting saved history.
        match = re.match(r'模型接口返回 HTTP (\d{3})', report.get('error') or '')
        code = report.get('http_status') or (int(match[1]) if match else None)
        if report.get('status') == 'failed' and isinstance(code, int) and code >= 400:
            report = dict(report, http_status=code, error=http_error_message(code, report.get('thinking_mode', 'default')))
    if report and report.get('kind') == 'pelican' and report.get('status') == 'completed':
        # Rebuild older previews from their original output as the sanitizer improves.
        try:
            svg, removed = sanitize_svg(report.get('content') or report.get('svg', ''))
            report = dict(report, svg=svg, animated=svg_has_animation(svg), preview_modified=bool(removed))
        except ValueError:
            pass
    return report


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def trim(value, limit=900):
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    value = str(value)
    return value if len(value) <= limit else value[:limit] + "…"


def prompt_text(value):
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict) and item.get("prompt"):
                parts.append(str(item["prompt"]))
            elif isinstance(item, dict) and item.get("content"):
                parts.append(str(item["content"]))
        return "\n".join(parts)
    return value


def discover_prediction_files(run_dir: Path):
    return sorted([*run_dir.glob("predictions/**/*.jsonl"), *run_dir.glob("predictions/**/*.json")])


def model_name(run_dir: Path):
    files = discover_prediction_files(run_dir)
    if files:
        return files[0].relative_to(run_dir).parts[1]
    configs = sorted((run_dir / "configs").glob("*.py"))
    if configs:
        text = configs[-1].read_text(encoding="utf-8", errors="ignore")
        match = re.search(r"abbr\s*=\s*['\"]([^'\"]+)", text)
        if match:
            return match.group(1)
    return "Unknown model"


def parse_summary(run_dir: Path):
    scores = []
    for path in sorted((run_dir / "summary").glob("*.csv")):
        try:
            with path.open(encoding="utf-8", newline="") as source:
                rows = list(csv.DictReader(source))
        except (OSError, csv.Error):
            continue
        for row in rows:
            dataset = row.get("dataset") or row.get("Dataset") or "Unknown"
            metric = row.get("metric") or row.get("Metric") or "score"
            reserved = {"dataset", "Dataset", "version", "metric", "Metric", "mode"}
            for key, raw in row.items():
                if key in reserved or raw in (None, ""):
                    continue
                try:
                    score = float(str(raw).strip().rstrip("%"))
                except ValueError:
                    continue
                # OpenCompass summary CSVs already contain scores on the
                # evaluator's scale; 0.5% must never be inflated to 50%.
                scores.append({"dataset": dataset, "metric": metric, "score": round(score, 2), "model": key})
    return scores


def prediction_rows(run_dir: Path, limit=250):
    rows = []
    for path in discover_prediction_files(run_dir):
        dataset = path.stem.replace("tmp_", "").replace("_", " ")
        model = path.relative_to(run_dir).parts[1]
        try:
            if path.suffix == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
                items = list(payload.values()) if isinstance(payload, dict) else payload
            else:
                items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        except OSError:
            continue
        except ValueError:
            continue
        for item in items:
            rows.append({
                "idx": item.get("idx", len(rows)),
                "dataset": dataset,
                "model": model,
                "prediction": trim(item.get("prediction", "")),
                "gold": trim(item.get("gold", "")),
                "prompt": trim(prompt_text(item.get("prompt") or item.get("origin_prompt") or ""), 500),
            })
            if len(rows) >= limit:
                return rows
    return rows


def run_summary(run_dir: Path):
    predictions = prediction_rows(run_dir)
    scores = parse_summary(run_dir)
    has_results = any((run_dir / name).exists() and any((run_dir / name).rglob("*")) for name in ("results", "summary"))
    manifest = read_json(run_dir.parent / 'evaluation_manifest.json')
    status = "completed" if has_results else ("in progress" if predictions else "empty")
    coverage = None
    if manifest:
        expected = {abbr for item in manifest for abbr in item['abbrs']}
        finished = {path.stem for path in (run_dir / 'results').rglob('*.json')
                    if isinstance(read_json(path), dict) and not read_json(path).get('error')}
        coverage = {'completed': len(expected & finished), 'total': len(expected),
                    'datasets': [item['id'] for item in manifest]}
        status = 'completed' if expected <= finished else 'partial'
    return {
        "coverage": coverage,
        "id": run_dir.name,
        "timestamp": run_dir.name,
        "model": model_name(run_dir),
        "status": status,
        "samples": len(predictions),
        "scores": scores,
    }


def discover_runs():
    if not OUTPUTS.exists():
        return []
    runs = [p for p in OUTPUTS.rglob("*") if p.is_dir() and re.fullmatch(r"\d{8}_\d{6}", p.name)]
    summaries = (run_summary(p) for p in runs)
    return sorted((item for item in summaries if item["status"] != "empty"),
                  key=lambda item: item["timestamp"], reverse=True)


def run_detail(run_id):
    safe = Path(run_id)
    if safe.name != run_id or ".." in safe.parts:
        return None
    matches = [p for p in OUTPUTS.rglob(run_id) if p.is_dir()]
    if not matches:
        return None
    run_dir = matches[0]
    data = run_summary(run_dir)
    files = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.stat().st_size < 2_000_000:
            files.append(str(path.relative_to(run_dir)))
    data['generation_parameters'] = read_json(run_dir.parent / 'evaluation_parameters.json')
    data["files"] = files
    data["samples"] = prediction_rows(run_dir)
    data['score_interpretations'] = [dict(describe_benchmark(item['dataset']),
                                         score=item['score'], metric=item['metric'])
                                    for item in data['scores']]
    data['preview_diagnostics'] = {
        'sample_count': len(data['samples']),
        'empty_predictions': sum(not str(item['prediction']).strip() for item in data['samples']),
        'note': '这里只统计最多 250 条样本预览；应结合完整预测和日志确认错误原因。',
    }
    return data


PAGE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>OpenCompass Runs</title>
<link rel="icon" href="data:," />
<style>
:root{--ink:#f1f0e9;--muted:#8e969a;--bg:#171b1d;--rail:#121618;--panel:#f5f4ee;--panel2:#ebeae3;--dark:#22282a;--lime:#a6ff44;--lime2:#6ed52c;--line:#d7d6ce;--danger:#ff7180;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);min-height:100vh}button,input,textarea{font:inherit}.shell{display:grid;grid-template-columns:280px 1fr;min-height:100vh}.rail{background:var(--rail);padding:26px 16px 20px;display:flex;flex-direction:column;border-right:1px solid #293032}.brand{font-size:22px;font-weight:760;letter-spacing:-.03em;padding:0 12px 24px}.brand span{color:var(--lime)}.rail-label{color:#687175;font-size:12px;text-transform:uppercase;letter-spacing:.12em;margin:16px 12px 8px}.run-list{display:flex;flex-direction:column;gap:4px;overflow:auto;flex:1;min-height:0}.run{border:1px solid transparent;border-radius:12px;background:transparent;color:var(--ink);text-align:left;padding:12px;cursor:pointer}.run:hover{background:#202729}.run.active{background:#2b3234;border-color:#3b4446}.run-title{font-size:13px;font-weight:650;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.run-meta{font-size:11px;color:var(--muted);margin-top:5px;display:flex;justify-content:space-between;gap:6px}.status{color:var(--lime)}.status.in-progress{color:#f0c76a}.status.failed{color:var(--danger)}.nav{margin-top:auto;border-top:1px solid #293032;padding-top:12px}.nav button{border:0;background:none;color:#a9b0b2;width:100%;text-align:left;padding:11px 12px;border-radius:9px;cursor:pointer}.nav button:hover,.nav button.selected{background:#242a2c;color:var(--ink)}main{padding:34px 42px 56px;min-width:0}.topbar{display:flex;align-items:center;justify-content:space-between;gap:20px;margin:0 auto 18px;max-width:1220px}.crumb{color:#aab2b4;font-size:14px}.crumb b{color:var(--ink)}.actions{display:flex;gap:8px}.refresh{background:transparent;border:1px solid #455052;color:var(--ink);border-radius:8px;padding:9px 13px;cursor:pointer}.secondary{background:transparent;border:1px solid #9aa09a;color:#1f2425;border-radius:8px;padding:9px 13px;cursor:pointer}.refresh:hover{border-color:var(--lime);color:var(--lime)}.secondary:hover{border-color:var(--lime2);color:#1f2425;background:#e4f5cf}.primary{background:var(--dark);border:1px solid var(--dark);color:var(--lime);border-radius:8px;padding:10px 14px;cursor:pointer;font-weight:650}.primary:hover{background:#101516}.primary:disabled{opacity:.55;cursor:wait}.hero,.panel,.stat{background:var(--panel);color:#1f2425;border-radius:16px}.hero{padding:28px 32px;margin:0 auto 16px;max-width:1220px}.hero h1{font-family:Georgia,"Times New Roman",serif;font-size:42px;letter-spacing:-.045em;line-height:1.05;margin:0 0 14px}.hero p{margin:0;line-height:1.65}.hero-meta{display:flex;gap:28px;flex-wrap:wrap}.hero-meta label{display:block;font-size:11px;color:#778083;text-transform:uppercase;letter-spacing:.08em}.hero-meta strong{display:block;margin-top:6px;font-size:14px}.stats{max-width:1220px;margin:0 auto 16px;display:grid;grid-template-columns:repeat(4,1fr);gap:16px}.stat{padding:20px 22px}.stat-label{font-size:12px;color:#6c7476}.stat-value{font-family:Georgia,serif;font-size:35px;letter-spacing:-.04em;margin-top:12px}.stat-sub{font-size:12px;color:#6c7476;margin-top:4px}.content{max-width:1220px;margin:0 auto;display:grid;grid-template-columns:1fr 1fr;gap:16px}.panel{padding:24px}.panel.full{grid-column:1/-1}.panel-head{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:20px}.panel h2{font-family:Georgia,serif;font-size:22px;letter-spacing:-.03em;margin:0}.muted{color:#7c8587;font-size:12px}.empty{color:#7c8587;padding:20px 0;line-height:1.6}.cta{border:1px dashed #b9bab2;background:#eeeee8;border-radius:12px;padding:24px;margin-top:18px}.cta h3{margin:0 0 8px;font-size:17px}.bar-row{display:grid;grid-template-columns:110px 1fr 54px;align-items:center;gap:12px;margin:17px 0}.bar-label{font-size:13px;font-weight:650}.bar-track{height:12px;background:#d8d9d5;border-radius:99px;overflow:hidden}.bar-fill{height:100%;background:linear-gradient(90deg,var(--lime),var(--lime2));border-radius:99px}.bar-score{text-align:right;font-family:Georgia,serif;font-size:18px}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:10px}.samples{border-collapse:collapse;width:100%;min-width:720px;font-size:12px}.samples th,.samples td{padding:12px 11px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}.samples th{color:#737b7d;font-weight:600;background:#eeeee8}.samples tr:last-child td{border-bottom:0}.samples td:nth-child(1){color:#8a9293;width:34px}.samples td:nth-child(4),.samples td:nth-child(5){max-width:280px;white-space:pre-wrap}.sample-prompt{color:#596163;white-space:pre-wrap;max-width:300px}.files{display:flex;flex-wrap:wrap;gap:8px}.file{background:#e6e6df;border-radius:7px;padding:7px 9px;color:#596163;font-size:11px}.search{width:220px;border:1px solid #c9c9c1;background:#faf9f3;border-radius:8px;padding:9px 11px;outline:0}.search:focus,.field input:focus,.field textarea:focus{border-color:var(--lime2)}.form-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.field{display:flex;flex-direction:column;gap:7px}.field.full{grid-column:1/-1}.field label{font-size:12px;color:#6c7476}.field input,.field textarea,.field select{border:1px solid #c9c9c1;background:#faf9f3;color:#1f2425;border-radius:8px;padding:11px 12px;outline:0}.field textarea{min-height:110px;resize:vertical}.form-actions{display:flex;align-items:center;gap:12px;margin-top:20px}.hint{font-size:12px;color:#7c8587;line-height:1.6}.report-list{display:grid;gap:10px}.report-card{display:flex;align-items:center;justify-content:space-between;gap:16px;border:1px solid var(--line);background:#eeeee8;border-radius:10px;padding:14px 16px}.report-card h3{margin:0 0 5px;font-size:14px}.report-card p{margin:0;color:#7c8587;font-size:12px}.report-content{white-space:pre-wrap;line-height:1.7;font-size:14px;color:#2c3334;background:#faf9f3;border-radius:10px;padding:22px;overflow:auto}.error{color:#9d3038;background:#ffe3e4;padding:12px;border-radius:8px}.toast{position:fixed;bottom:20px;right:20px;background:#303a3c;color:var(--ink);padding:10px 14px;border-radius:9px;box-shadow:0 10px 30px #0005;display:none;z-index:4}.dataset-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.dataset-card{border:1px solid var(--line);background:#eeeee8;border-radius:11px;padding:14px;cursor:pointer;transition:.15s}.dataset-card:hover{border-color:#96be6c}.dataset-card.selected{border-color:var(--lime2);background:#e4f5cf;box-shadow:inset 0 0 0 1px var(--lime2)}.dataset-card input{accent-color:#5c9d28;margin-right:8px}.dataset-title{font-weight:700;font-size:14px}.dataset-meta{font-size:11px;color:#788082;margin-top:7px;line-height:1.5}.collection-list{display:grid;gap:9px}.collection-card{display:flex;align-items:center;justify-content:space-between;gap:12px;border:1px solid var(--line);background:#eeeee8;border-radius:10px;padding:13px}.collection-card h3{font-size:14px;margin:0 0 5px}.collection-card p{margin:0;color:#7c8587;font-size:12px}.check-row{display:flex;align-items:center;gap:8px;font-size:12px;color:#596163}.status-box{display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:12px 14px;border-radius:9px;background:#e9e9e2;font-size:13px}.status-dot{width:9px;height:9px;border-radius:50%;background:#8b9495}.status-dot.running{background:#e4b240;box-shadow:0 0 0 4px #e4b24033}.status-dot.completed{background:#66ad31}.status-dot.failed{background:#d85261}@media(max-width:900px){.shell{grid-template-columns:1fr}.rail{position:static;padding:16px}.run-list{display:flex;flex-direction:row;overflow:auto}.run{min-width:180px}.rail-label{display:none}.nav{display:flex;border-top:0;padding-top:0;gap:4px;overflow:auto}.nav button{width:auto;white-space:nowrap}main{padding:22px 16px 40px}.stats{grid-template-columns:repeat(2,1fr)}.content,.form-grid,.dataset-grid{grid-template-columns:1fr}.panel.full,.field.full{grid-column:auto}.hero h1{font-size:32px}}

[hidden]{display:none!important}.nav{margin-top:0;border-top:0;padding-top:0}.nav button{min-height:44px;font-size:14px}.rail-label{margin-top:28px}#runSection{min-width:0}.panel,.hero{max-width:1220px;margin-left:auto;margin-right:auto}.form-actions{flex-wrap:wrap}.dataset-meta,.file,.status-box{overflow-wrap:anywhere}.status-box{min-width:0}.status-box .muted{min-width:0}.dataset-card:focus-within{outline:2px solid #598c25;outline-offset:2px}button:focus-visible{outline:2px solid #6ed52c;outline-offset:3px}.eval-intro{display:flex;align-items:center;justify-content:space-between;gap:20px}.eval-intro h1{font-family:inherit;font-size:28px;letter-spacing:-.03em;margin-bottom:8px}.eval-intro .secondary{flex-shrink:0}.muted,.hint,.dataset-meta{color:#596466}.rail .run-meta{color:#abb4b7}#evaluationStatus:empty{display:none}.content>.panel{width:100%}
@media(max-width:900px){.rail-label{display:block;margin-top:16px}.brand{padding-bottom:12px}.nav button{flex:1;text-align:center;padding:10px 8px}.nav{overflow:visible}.eval-intro{align-items:flex-start;flex-direction:column}.hero{padding:22px}.panel{padding:20px}.panel-head{flex-wrap:wrap}.search{max-width:100%}.collection-card{flex-wrap:wrap}}
.rail{min-width:0}.dataset-card:has(input:disabled){opacity:.65;cursor:not-allowed}.bar-row{grid-template-columns:minmax(120px,2fr) minmax(70px,3fr) 54px}.bar-label{overflow-wrap:anywhere}

.eval-modes{max-width:1220px;margin:0 auto 20px;display:grid;grid-template-columns:1fr 1fr;gap:12px}.eval-modes button{text-align:left;border:1px solid #455052;background:transparent;color:var(--ink);padding:18px 22px;border-radius:12px;cursor:pointer}.eval-modes strong,.eval-modes span{display:block}.eval-modes span{font-size:13px;color:#aab4b7;margin-top:6px}.eval-modes button.chosen{border-color:var(--lime);background:#263020}.pelican-layout{display:grid;grid-template-columns:minmax(230px,1fr) minmax(0,2fr);gap:36px;align-items:start}.pelican-controls h1{font-size:28px;line-height:1.3;margin:0 0 18px}.pelican-controls>p{line-height:1.8;font-size:14px}.prompt-box{padding:18px 0;margin:24px 0;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}.prompt-box span{font-size:12px;color:#596466}.prompt-box blockquote{margin:12px 0 0;font-size:19px;line-height:1.5}.review-guide{margin-top:28px}.review-guide h3{font-size:15px}.review-guide ul{padding-left:20px;font-size:13px;line-height:2}.svg-stage{background:#fff;border:1px solid var(--line);border-radius:10px;aspect-ratio:4/3;display:grid;place-items:center;overflow:hidden}.svg-stage img{display:block;width:100%;height:100%;min-height:0;object-fit:contain}.svg-stage.large{min-height:320px}.svg-stage.placeholder{padding:28px;text-align:center;color:#66716f}.placeholder h2{font-family:inherit;font-size:22px}.placeholder p{font-size:14px;line-height:1.7}.result-metrics{display:flex;gap:22px;flex-wrap:wrap;margin:20px 0;font-size:12px;color:#596466}.result-metrics b{display:block;margin-top:5px;color:#1f2425;font-size:17px}.result-badge{border:1px solid #c4ccb9;border-radius:5px;padding:5px 8px;font-size:12px;white-space:nowrap}.raw-output{border-top:1px solid var(--line);margin-top:18px;padding-top:14px}.raw-output summary{cursor:pointer;font-size:13px}.raw-output pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:400px;overflow:auto;font-size:12px;padding:16px;background:#e9e9e3;border-radius:8px}.preview-toolbar{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;margin-bottom:14px}.svg-download{display:inline-flex;align-items:center;justify-content:center;min-height:44px;text-decoration:none;font-size:13px}.download-link{font-size:13px;color:#2a6318;text-decoration:underline}.gallery,.comparison-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}.gallery-card{border:1px solid var(--line);border-radius:12px;overflow:hidden}.gallery-card .svg-stage{border:0;border-bottom:1px solid var(--line);border-radius:0}.gallery-caption{padding:16px}.gallery-caption h3{margin:0;font-size:16px}.gallery-caption label{font-size:13px}.gallery-caption .form-actions{margin-top:10px}.report-history,.comparison-panel{margin-top:20px}.visual-result{min-width:0}.comparison-grid .svg-stage.large{min-height:240px}.panel h2{font-family:inherit;font-size:21px}.comparison-grid>.visual-result{padding:0 8px}.pelican-controls .form-actions button{min-height:44px}
@media(max-width:1000px){.pelican-layout{grid-template-columns:1fr}.pelican-controls{max-width:none}.review-guide{margin-top:16px}.svg-stage.large{min-height:240px}}@media(max-width:650px){.eval-modes{gap:8px}.eval-modes button{padding:14px 12px}.eval-modes span{font-size:12px}.gallery,.comparison-grid{grid-template-columns:1fr}.pelican-layout{gap:24px}.pelican-controls h1{font-size:25px}.result-metrics{gap:16px}.svg-stage.large{min-height:210px}}
.result-tools,.pagination,.page-controls{display:flex;align-items:center;gap:10px;flex-wrap:wrap}.result-tools,.pagination{justify-content:space-between}.result-tools .search{flex:1;min-width:120px}.result-tools label{font-size:12px;color:#596466}.result-tools select{font:inherit;border:1px solid #9aa09a;border-radius:6px;padding:8px;background:#faf9f3;color:#1f2425}.pagination{border-top:1px solid var(--line);padding-top:14px;margin-top:18px}.pagination button{min-height:44px}.secondary:disabled{opacity:.4;cursor:default}.result-metric{display:block;font-size:11px;font-weight:400;color:#596466;margin-top:4px}.content>.panel{min-width:0}
/* Keep navigation and page content in separate viewport-sized scroll regions. */
body{height:100vh;height:100dvh;overflow:hidden}
.shell{height:100%;min-height:0;grid-template-columns:280px minmax(0,1fr);grid-template-rows:minmax(0,1fr);overflow:hidden}
.rail,main{min-height:0;overflow-y:auto;overscroll-behavior-y:contain;scrollbar-gutter:stable}
.rail>*{flex-shrink:0}
.run-list{overflow:visible;flex:none}
@media(max-width:900px){
  .shell{grid-template-columns:minmax(0,1fr);grid-template-rows:auto minmax(0,1fr)}
  .rail{max-height:40vh;max-height:40dvh}
  .run-list{overflow-x:auto}
}
</style>
<link rel="stylesheet" href="/assets/performance_ui.css" />
<link rel="stylesheet" href="/assets/benchmark_ui.css" />
<link rel="stylesheet" href="/assets/environment_ui.css" />
</head>
<body>
<div class="shell"><aside class="rail"><div class="brand">OpenCompass <span>Runs</span></div><nav class="nav" aria-label="主导航"><button data-view="evaluate" class="selected">开始评测</button><button data-view="performance">性能测试</button><button data-view="reports">历史报告</button><button data-view="settings">模型设置</button><button data-view="environment">环境管理</button></nav><section id="runSection" hidden><div class="rail-label">评测记录</div><div id="runList" class="run-list"></div></section></aside>
<main tabindex="0" aria-label="评测内容"><div class="topbar"><div class="crumb">OpenCompass&nbsp; / &nbsp;<b id="crumb">Loading…</b></div><div class="actions" id="actions"></div></div><div id="app"></div></main></div><div class="toast" id="toast" role="status"></div>
<script src="/assets/performance_ui.js"></script>
<script src="/assets/benchmark_ui.js"></script>
<script src="/assets/environment_ui.js"></script>
<script>
const state={benchmarkMetadata:{categories:[],presets:[]},benchmarkMetadataWarning:'',runs:[],selected:null,detail:null,settings:null,reports:[],report:null,datasets:[],collections:[],selectedDatasets:[],abilityOptions:{max_seq_len:262144,max_out_len:32768},evalStatus:null,view:'evaluate',collectionDraft:{name:'',description:''},starting:false,evalMode:'standard',pelican:null,pelicanStarting:false,pelicanMaxTokens:32768,pelicanThinkingMode:'default',performance:null,perfStarting:false,perfDraft:{concurrency:'1,2,4',contexts:'1024,4096,8192',requests:8,output:256,warmup:1,timeout:120},compareIDs:[],comparison:null};
const resultPaging={runId:null,page:1,size:10,query:'',category:'all'};
const $=s=>document.querySelector(s);
function esc(v){return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function fmtDate(v){if(!/^\d{8}_\d{6}$/.test(v))return v;return `${v.slice(0,4)}-${v.slice(4,6)}-${v.slice(6,8)} ${v.slice(9,11)}:${v.slice(11,13)}`}
async function request(url,options={}){const r=await fetch(url,{headers:{'Content-Type':'application/json',...(options.headers||{})},...options});const data=await r.json().catch(()=>({}));if(!r.ok)throw Error(data.error||'请求失败');return data}
function showToast(text){const el=$('#toast');el.textContent=text;el.style.display='block';setTimeout(()=>el.style.display='none',2200)}
function showError(error){showToast(error.message||String(error))}
function renderRuns(){
 const el=$('#runList'), status=state.evalStatus||{};
 $('#runSection').hidden=!state.runs.length&&status.status!=='running';
 el.innerHTML=(status.status==='running'?'<button class="run" id="activeEvaluation"><div class="run-title">当前评测</div><div class="run-meta"><span>查看进度</span><span class="status">运行中</span></div></button>':'')+state.runs.map(r=>`<button class="run ${state.view==='runs'&&r.id===state.selected?'active':''}" data-id="${esc(r.id)}"><div class="run-title">${esc(r.model)}</div><div class="run-meta"><span>${esc(fmtDate(r.timestamp))}</span><span class="status">${esc(r.status==='completed'?'已完成':r.status==='partial'?'结果未齐':'已有预测')}</span></div></button>`).join('');
 el.querySelectorAll('[data-id]').forEach(x=>x.onclick=()=>selectRun(x.dataset.id));
 if($('#activeEvaluation'))$('#activeEvaluation').onclick=()=>navigate('evaluate');
}
function scoreCards(scores,samples){const vals=scores.map(s=>s.score).filter(Number.isFinite);const avg=vals.length?vals.reduce((a,b)=>a+b,0)/vals.length:null;const best=vals.length?Math.max(...vals):null;const low=vals.length?Math.min(...vals):null;return `<div class="stats"><div class="stat"><div class="stat-label">分项平均</div><div class="stat-value">${avg==null?'—':avg.toFixed(1)}</div><div class="stat-sub">${scores.length||0} 个指标 · 非综合能力分</div></div><div class="stat"><div class="stat-label">最高分</div><div class="stat-value">${best==null?'—':best.toFixed(1)}</div><div class="stat-sub">${best==null?'尚无 summary':esc(scoreInterpretation(scores.find(s=>s.score===best)||{}).label||'')}</div></div><div class="stat"><div class="stat-label">最低分</div><div class="stat-value">${low==null?'—':low.toFixed(1)}</div><div class="stat-sub">${low==null?'仅有预测记录':esc(scoreInterpretation(scores.find(s=>s.score===low)||{}).label||'')}</div></div><div class="stat"><div class="stat-label">样本数</div><div class="stat-value">${samples}</div><div class="stat-sub">预测记录</div></div></div>`}
function renderRunsView(){const d=state.detail;$('#crumb').textContent=d?d.model:'评测运行';$('#actions').innerHTML='<button class="refresh" id="newEvaluation">开始新评测</button><button class="refresh" id="refresh">刷新</button>';if(!d){state.view='evaluate';render();return}const scores=d.scores||[];const samples=d.samples||[];const rows=samples.slice(0,30).map((s,i)=>`<tr><td>${esc(s.idx??i)}</td><td>${esc(s.dataset)}</td><td class="sample-prompt">${esc(s.prompt)}</td><td>${esc(s.prediction)}</td><td>${esc(s.gold)}</td></tr>`).join('');$('#app').innerHTML=`<section class="hero"><h1>${esc(d.model)}</h1>${d.coverage?`<p class="hint">本次选择 ${d.coverage.datasets.length} 个评测配置 · 已完成 ${d.coverage.completed}/${d.coverage.total} 个子任务</p>`:''}<div class="hero-meta"><div><label>状态</label><strong>${esc(d.status==='partial'?'结果未齐':d.status==='completed'?'已完成':d.status)}</strong></div><div><label>运行时间</label><strong>${esc(fmtDate(d.timestamp))}</strong></div><div><label>行数</label><strong>${samples.length}</strong></div><div><label>来源</label><strong>OpenCompass outputs</strong></div></div></section>${scoreCards(scores,samples.length)}<div class="content"><section class="panel full"><div class="panel-head"><h2>Benchmark 结果与用途</h2><span class="muted">分数越高越好</span></div><p class="hint">分类数字为本次指标条数；0 表示本次未包含该类。</p><div id="scoreCategories" class="category-filters" role="group" aria-label="按结果能力筛选"></div><div class="result-tools"><input id="scoreSearch" class="search" type="search" aria-label="筛选评测集" placeholder="搜索评测集、学科或用途" /><label>每页 <select id="scorePageSize" aria-label="每页评测结果数量"><option value="10">10 条</option><option value="20">20 条</option><option value="50">50 条</option></select></label></div><div id="scoreResults"></div><nav id="scorePagination" class="pagination" aria-label="评测结果分页"></nav></section>${benchmarkInterpretations(d)}<section class="panel"><div class="panel-head"><h2>运行文件</h2><span class="muted">${d.files.length} 个文件</span></div><div class="files">${d.files.slice(0,40).map(f=>`<span class="file">${esc(f)}</span>`).join('')||'<span class="empty">暂无文件</span>'}</div></section><section class="panel"><div class="panel-head"><h2>生成分析报告</h2><span class="muted">调用已保存的本地模型接口</span></div><div class="field"><label for="reportInstruction">报告要求（可选）</label><textarea id="reportInstruction" placeholder="例如：重点分析数学题错误，并给出提示词优化建议"></textarea></div><div class="form-actions"><button class="primary" id="generateReport">✦ 请求生成报告</button><span class="hint">报告会保存到本地历史记录。</span></div></section><section class="panel"><div class="panel-head"><h2>最近样本</h2><input id="search" class="search" placeholder="筛选样本" /></div><div class="table-wrap"><table class="samples"><thead><tr><th>#</th><th>数据集</th><th>问题</th><th>模型输出</th><th>标准答案</th></tr></thead><tbody id="sampleRows">${rows||'<tr><td colspan="5" class="empty">暂无预测记录。</td></tr>'}</tbody></table></div></section></div>`;$('#newEvaluation').onclick=()=>navigate('evaluate');$('#refresh').onclick=()=>loadRuns();const search=$('#search');if(search)search.oninput=()=>{const q=search.value.toLowerCase();$('#sampleRows').querySelectorAll('tr').forEach(row=>row.style.display=row.textContent.toLowerCase().includes(q)?'':'none')};$('#generateReport').onclick=generateReport;initScorePagination();}
function initScorePagination(){
 if(resultPaging.runId!==state.selected)Object.assign(resultPaging,{runId:state.selected,page:1,query:'',category:'all'});
 const search=$('#scoreSearch'),size=$('#scorePageSize');
 search.value=resultPaging.query;size.value=String(resultPaging.size);
 search.oninput=()=>{resultPaging.query=search.value;resultPaging.page=1;renderScoreResults()};
 size.onchange=()=>{resultPaging.size=Number(size.value);resultPaging.page=1;renderScoreResults()};
 renderScoreResults();
}
function renderScoreResults(){
 const scores=state.detail?.scores||[],query=resultPaging.query.trim().toLowerCase();
 const filtered=scores.filter(s=>{const info=scoreInterpretation(s);if(resultPaging.category!=='all'&&(info.category_id||'unknown')!==resultPaging.category)return false;return [s.dataset,info.label,info.category,info.purpose,...(info.suitable_uses||[])].join(' ').toLowerCase().includes(query)});
 $('#scoreCategories').innerHTML=categoryFilters({scores:true});document.querySelectorAll('[data-score-category]').forEach(btn=>btn.onclick=()=>{resultPaging.category=btn.dataset.scoreCategory;resultPaging.page=1;renderScoreResults();[...document.querySelectorAll('[data-score-category]')].find(el=>el.dataset.scoreCategory===resultPaging.category)?.focus({preventScroll:true})});
 const pages=Math.max(1,Math.ceil(filtered.length/resultPaging.size));
 resultPaging.page=Math.max(1,Math.min(resultPaging.page,pages));
 const start=(resultPaging.page-1)*resultPaging.size,visible=filtered.slice(start,start+resultPaging.size);
 $('#scoreResults').innerHTML=visible.map(renderBenchmarkScore).join('')||`<div class="empty">${scores.length?'没有匹配的评测集。':'这次运行已有预测，但尚未生成 summary 分数。'}</div>`;
 $('#scorePagination').innerHTML=`<span class="hint" role="status">${filtered.length?`${start+1}–${Math.min(start+resultPaging.size,filtered.length)} / ${filtered.length} 条结果`:'0 条结果'}${query||resultPaging.category!=='all'?`（共 ${scores.length} 条）`:''}</span><div class="page-controls"><button class="secondary" id="scorePrev" ${resultPaging.page===1?'disabled':''}>上一页</button><span class="hint">${resultPaging.page} / ${pages}</span><button class="secondary" id="scoreNext" ${resultPaging.page===pages?'disabled':''}>下一页</button></div>`;
 for(const [id,delta] of [['scorePrev',-1],['scoreNext',1]])$('#'+id).onclick=()=>{
  resultPaging.page+=delta;renderScoreResults();
  const button=$('#'+id);(button.disabled?$('#'+(delta>0?'scorePrev':'scoreNext')):button).focus({preventScroll:true});
 };
}
function collectionNames(c){return c.datasets.map(id=>{const item=state.datasets.find(d=>d.id===id);return item?item.label:id}).join('、')}
function renderEvaluateView(){if(state.evalMode==='pelican'){renderPelicanView();return}renderStandardEvaluation()}
function evaluationStatusMarkup(){
 const status=state.evalStatus||{};if(!status.status||status.status==='idle')return '';
 const labels={running:'评测进行中',completed:'评测已完成',failed:'评测失败'};
 return `<section class="panel" style="margin-bottom:16px"><div class="panel-head"><h2>${esc(labels[status.status]||status.status)}</h2></div><div class="status-box"><span class="status-dot ${esc(status.status)}"></span><span>${esc((status.datasets||[]).map(id=>state.datasets.find(d=>d.id===id)?.label||id).join('、'))}</span></div><p class="hint">${status.status==='running'?'正在收集模型回答，你可以切换页面，进度会持续更新。':status.status==='failed'?'请查看日志，并检查模型设置。':'在下方或左侧选择评测记录，查看分数并生成报告。'}</p>${status.error?`<p class="error">${esc(status.error)}</p>`:''}${status.dataset_count?`<p class="hint">${(status.datasets||[]).length} 个评测配置，共 ${status.dataset_count} 个子任务</p>`:''}${status.log_path?`<details><summary class="hint">运行详情</summary><p class="hint">日志：${esc(status.log_path)}</p></details>`:''}${state.runs.length?'<button class="secondary" id="viewLatestRun">查看最近结果</button>':''}</section>`;
}
function bindStatus(){if($('#viewLatestRun'))$('#viewLatestRun').onclick=()=>selectRun(state.runs[0].id)}
function updateEvaluationStatus(){
 renderRuns();if(state.view!=='evaluate'||state.evalMode!=='standard')return;
 const region=$('#evaluationStatus');if(region){region.innerHTML=evaluationStatusMarkup();bindStatus()}
 if($('#datasetResults'))updatePickerResults();
}
async function loadEvaluate(){try{const [datasets,collections,evalStatus,metadata]=await Promise.all([request('/api/datasets'),request('/api/collections'),request('/api/evaluation/status'),request('/api/benchmark-metadata').catch(()=>null)]);Object.assign(state,{datasets,collections,evalStatus});acceptBenchmarkMetadata(metadata);state.selectedDatasets=state.selectedDatasets.filter(id=>datasets.some(d=>d.id===id&&d.available));if(state.view==='evaluate')render();if(evalStatus.status==='running')pollEvaluation()}catch(e){showToast(e.message)}}
async function saveCollection(){const name=$('#collectionName').value.trim();if(!name){showToast('请填写集合名称');return}try{await request('/api/collections',{method:'POST',body:JSON.stringify({name,description:$('#collectionDescription').value,datasets:state.selectedDatasets})});state.collections=await request('/api/collections');showToast('集合已保存');renderEvaluateView()}catch(e){showError(e)}}
async function startEvaluation(){
 if(!state.selectedDatasets.length||state.starting)return;
 state.starting=true;updateEvaluationStatus();
 try{state.evalStatus=await request('/api/evaluation/start',{method:'POST',body:JSON.stringify({datasets:state.selectedDatasets,options:state.abilityOptions})});showToast('OpenCompass 已启动');pollEvaluation()}
 catch(e){showToast(e.message)}finally{state.starting=false;updateEvaluationStatus()}
}
let pollTimer=null;
function pollEvaluation(){
 if(pollTimer)return;
 pollTimer=setTimeout(async()=>{
  try{
   state.evalStatus=await request('/api/evaluation/status');
   state.runs=await request('/api/runs');updateEvaluationStatus();
   if(state.evalStatus.status!=='running')showToast(state.evalStatus.status==='completed'?'评测已完成，可查看结果':'评测失败，请查看日志');
  }catch(e){showToast('进度更新失败：'+e.message)}
  finally{pollTimer=null;if(state.evalStatus?.status==='running')pollEvaluation()}
 },3000);
}
function renderSettingsView(){state.view='settings';$('#crumb').textContent='模型设置';$('#actions').innerHTML='';const s=state.settings||{};$('#app').innerHTML=`<section class="panel"><div class="panel-head"><h2>本地模型接口</h2><span class="muted">保存到当前项目目录</span></div><div class="form-grid"><div class="field full"><label for="baseUrl">接口地址</label><input id="baseUrl" value="${esc(s.base_url||'')}" placeholder="http://127.0.0.1:8080/v1 或完整 /chat/completions 地址" /></div><div class="field"><label for="model">模型名称</label><input id="model" value="${esc(s.model||'')}" placeholder="Qwen3.8-Flash-Next" /></div><div class="field"><label for="apiKey">API Key</label><input id="apiKey" type="password" placeholder="${s.api_key_set?`已保存 ${esc(s.api_key_hint)}，留空保持不变`:'请输入 API Key'}" /></div></div><div class="form-actions"><button class="primary" id="saveSettings">保存设置</button><span class="hint">密钥保存在 .viewer_settings.json（权限 600），不会显示在页面中。</span></div></section>`;$('#saveSettings').onclick=async()=>{const button=$('#saveSettings');button.disabled=true;try{state.settings=await request('/api/settings',{method:'POST',body:JSON.stringify({base_url:$('#baseUrl').value,model:$('#model').value,api_key:$('#apiKey').value})});showToast('设置已保存');renderSettingsView()}catch(e){showError(e)}finally{button.disabled=false}}}
function evaluationModes(){return `<div class="eval-modes" role="group" aria-label="评测类型"><button class="${state.evalMode==='standard'?'chosen':''}" data-mode="standard"><strong>标准评测</strong><span>数据集、分数与逐题结果</span></button><button class="${state.evalMode==='pelican'?'chosen':''}" data-mode="pelican"><strong>鹈鹕 SVG 测试</strong><span>一次提示，直观看绘图效果</span></button></div>`}
function bindModes(){document.querySelectorAll('[data-mode]').forEach(btn=>btn.onclick=()=>{state.evalMode=btn.dataset.mode;render()})}
function pelicanDownload(r){return r.status==='completed'?`<a class="secondary svg-download" href="/api/pelican/${encodeURIComponent(r.id)}/svg?download=1" download="pelican-${esc(r.id)}.svg">${r.animated?'下载动画 SVG':'下载 SVG'}</a>`:''}
function pelicanPreview(r,large=false){return r.status==='completed'?`<div class="svg-stage ${large?'large':''}"><img loading="lazy" src="/api/pelican/${encodeURIComponent(r.id)}/svg" alt="${esc(r.model)} 生成的鹈鹕骑自行车 SVG" /></div>`:`<div class="svg-stage placeholder"><p>${r.status==='running'?'模型正在绘制…':'未生成可预览的 SVG'}</p></div>`}
function pelicanPhase(r){return ({connecting:'等待模型服务响应',thinking:'模型正在思考',writing:'正在接收 SVG 正文',completed:'生成完成',failed:'生成失败',cancelled:'已停止'})[r.phase]||'正在生成'}
function pelicanResult(r){if(!r)return `<div class="svg-stage large placeholder"><div><h2>让模型用代码画一幅画</h2><p>生成后，鹈鹕与自行车会出现在这里。</p></div></div>`;
 const thinking=r.thinking_mode==='disabled'?'请求关闭（Qwen 兼容）':'服务默认';
 return `<div class="visual-result"><div class="panel-head"><h2>${esc(r.model)}</h2><span class="result-badge">${r.status==='running'?'生成中':r.status==='completed'?'SVG 已生成':r.status==='cancelled'?'已停止':'生成失败'}</span></div>${r.status==='completed'?`<div class="preview-toolbar"><span class="hint">${r.animated?'动画预览 · 自动播放':'静态预览 · 此次输出未包含动画'}</span>${pelicanDownload(r)}</div>`:''}${pelicanPreview(r,true)}${r.status==='running'||r.request_stream?`<p class="hint" role="status">${esc(pelicanPhase(r))} · 已接收思考 ${(r.reasoning_chars||0).toLocaleString()} 字符 · 正文 ${(r.content||'').length.toLocaleString()} 字符${r.first_response_seconds!=null?' · 首生成 '+esc(r.first_response_seconds)+' s':''}</p>`:''}<div class="result-metrics"><span>耗时 <b>${r.elapsed_seconds==null?'—':esc(r.elapsed_seconds)+' s'}</b></span><span>输出 tokens <b>${esc(r.usage?.completion_tokens??'—')}</b></span>${r.usage?.reasoning_tokens!=null?`<span>其中思考 tokens <b>${esc(r.usage.reasoning_tokens)}</b></span>`:''}<span>格式 <b>SVG</b></span></div>${r.error?`<p class="error">${esc(r.error)}</p>`:''}${r.thinking_mode==='disabled'&&r.reasoning_chars>0?'<p class="hint">已请求关闭思考，但接口仍返回思考内容；服务可能未采用该参数，请核对服务端配置。</p>':''}${r.preview_modified?'<p class="hint">预览已过滤不支持的属性、脚本或外部资源，原始输出保留在下方。</p>':''}${r.finish_reason==='length'?'<p class="hint">输出达到 token 上限，绘图可能不完整。</p>':''}${r.status==='completed'?'<p class="hint">下载为独立 SVG 文件，包含预览中的动画样式，可用浏览器打开。</p>':''}<details class="raw-output"><summary>提示词、参数与原始输出</summary><p>${esc(r.prompt)}</p><p class="hint">Temperature ${esc(r.temperature)} · 最大输出 ${esc(r.max_tokens)} tokens · 思考 ${esc(thinking)} · ${r.request_stream?'请求流式生成':'旧版非流式请求'}</p><pre>${esc(r.content||(r.status==='running'?'正在接收模型响应，最终正文尚未返回…':'接口未返回最终文本。'))}</pre></details></div>`
}
function renderPelicanView(){
 const previousId=$('#pelicanResult')?.dataset.reportId,keepOpen=previousId===state.pelican?.id&&Boolean($('#pelicanResult .raw-output')?.open),main=document.querySelector('main'),scrollTop=main?.scrollTop;
 $('#crumb').textContent='鹈鹕 SVG 测试';$('#actions').innerHTML='';
 const r=state.pelican, busy=state.pelicanStarting||r?.status==='running';
 $('#app').innerHTML=evaluationModes()+`<section class="panel pelican-layout"><div class="pelican-controls"><h1>一只鹈鹕，一辆自行车。</h1><p>用同一句提示词，观察不同模型如何理解空间关系、编写 SVG 和安排画面。</p><div class="prompt-box"><span>固定提示词</span><blockquote>Generate an animated SVG of a pelican riding a bicycle. Animate the rotating wheels, pedaling legs, and gently moving wings using embedded CSS @keyframes. Return a complete, self-contained SVG with no JavaScript or external resources.</blockquote></div><p class="hint">当前模型</p><strong>${esc(state.settings?.model||'尚未设置模型')}</strong><div class="field" style="margin-top:18px"><label for="pelicanMaxTokens">生成额度（含思考 tokens）</label><select id="pelicanMaxTokens" ${busy?'disabled':''}>${[8192,16384,32768,65536,131072,262144].map(n=>`<option value="${n}" ${state.pelicanMaxTokens===n?'selected':''}>${n/1024}k · ${n.toLocaleString()} tokens</option>`).join('')}</select></div><div class="field" style="margin-top:14px"><label for="pelicanThinkingMode">思考模式</label><select id="pelicanThinkingMode" ${busy?'disabled':''}><option value="default" ${state.pelicanThinkingMode==='default'?'selected':''}>使用模型服务默认设置</option><option value="disabled" ${state.pelicanThinkingMode==='disabled'?'selected':''}>请求关闭思考（Qwen 兼容）</option></select></div><p class="hint">关闭思考需要接口支持 Qwen 参数；实际是否生效以返回内容为准。256k 上下文包含输入和生成内容，增加生成额度不会延长网关的等待上限。</p><div class="form-actions"><button class="primary" id="startPelican" ${busy?'disabled':''}>${busy?'正在生成…':'生成动画 SVG'}</button>${r?.status==='running'?'<button class="secondary" id="stopPelican">停止生成</button>':''}<button class="secondary" id="pelicanSettings">模型设置</button></div><p class="hint">流式接收会保存生成进度和已返回正文，客户端总时限为 10 分钟；模型服务或网关可能更早终止。结果自动存入“历史报告”，可选两次结果并排比较。</p><div class="review-guide"><h3>看什么？</h3><ul><li>鹈鹕的长喙和身体是否可辨认</li><li>车轮、车架与踏板是否合理连接</li><li>鸟是否真的骑在车上</li><li>构图、比例与细节是否协调</li></ul><p class="hint">这是非正式视觉观察测试，没有标准准确率，也不参与标准评测总分。</p></div></div><div id="pelicanResult" data-report-id="${esc(r?.id||'')}">${pelicanResult(r)}</div></section>`;
 bindModes();$('#pelicanMaxTokens').onchange=e=>state.pelicanMaxTokens=Number(e.target.value);$('#pelicanThinkingMode').onchange=e=>state.pelicanThinkingMode=e.target.value;$('#startPelican').onclick=startPelican;$('#pelicanSettings').onclick=()=>navigate('settings');if($('#stopPelican'))$('#stopPelican').onclick=stopPelican;
 if(previousId===r?.id){if($('#pelicanResult .raw-output'))$('#pelicanResult .raw-output').open=keepOpen;if(main&&scrollTop!=null)main.scrollTop=scrollTop}
}
async function startPelican(){
 if(state.pelicanStarting||state.pelican?.status==='running')return;
 state.pelicanStarting=true;render();
 try{state.pelican=await request('/api/pelican/start',{method:'POST',body:JSON.stringify({max_tokens:state.pelicanMaxTokens,thinking_mode:state.pelicanThinkingMode})});pollPelican();showToast('鹈鹕测试已启动')}
 catch(e){showToast(e.message)}finally{state.pelicanStarting=false;if(state.view==='evaluate'&&state.evalMode==='pelican')render()}
}
async function stopPelican(){try{state.pelican=await request('/api/pelican/stop',{method:'POST',body:'{}'});state.reports=await request('/api/reports');if(state.view==='evaluate'&&state.evalMode==='pelican')render();showToast('已停止，保留已接收的正文')}catch(e){showError(e)}}
let pelicanTimer=null;
function pollPelican(){if(pelicanTimer||!state.pelican||state.pelican.status!=='running')return;
 pelicanTimer=setTimeout(async()=>{try{
 const id=state.pelican.id;const report=await request(`/api/reports/${encodeURIComponent(id)}`);if(state.pelican?.id!==id)return;state.pelican=report;
 if(state.view==='evaluate'&&state.evalMode==='pelican')render();
 if(state.report?.id===id){state.report=report;if(state.view==='reports')render()}
 if(report.status!=='running'){state.reports=await request('/api/reports');if(state.view==='reports')render();showToast(report.status==='completed'?'鹈鹕 SVG 已生成':report.status==='cancelled'?'鹈鹕测试已停止':'鹈鹕测试失败，可查看原始输出')}
 }catch(e){showToast('生成状态更新失败：'+e.message)}finally{pelicanTimer=null;pollPelican()}},2500)
}
function renderReportsView(){
 $('#crumb').textContent='历史报告';$('#actions').innerHTML='<button class="refresh" id="refreshReports">刷新</button>';
 const gallery=state.reports.filter(r=>r.kind==='pelican'), reports=state.reports.filter(r=>!['pelican','performance'].includes(r.kind));
 $('#app').innerHTML=`<section class="panel"><div class="panel-head"><div><h2>鹈鹕结果画廊</h2><p class="hint">选择两次结果，比较不同模型或不同次生成的画面。</p></div><button class="secondary" id="newPelican">新建鹈鹕测试</button></div><div class="gallery">${gallery.length?gallery.map(r=>`<article class="gallery-card">${pelicanPreview(r)}<div class="gallery-caption"><h3>${esc(r.model)}</h3><p class="hint">${esc(r.created_at)}${r.elapsed_seconds!=null?' · '+esc(r.elapsed_seconds)+' s':''}</p><div class="form-actions"><button class="secondary" data-report="${esc(r.id)}">查看详情</button>${pelicanDownload(r)}<label><input type="checkbox" data-compare="${esc(r.id)}" ${state.compareIDs.includes(r.id)?'checked':''} ${r.status!=='completed'?'disabled':''}/> 加入对比</label></div></div></article>`).join(''):'<p class="empty">还没有绘图结果。在“开始评测”中选择“鹈鹕 SVG 测试”。</p>'}</div><div class="form-actions"><button class="primary" id="comparePelicans" ${state.compareIDs.length!==2?'disabled':''}>并排对比 (${state.compareIDs.length}/2)</button><span class="hint">视觉观察，不自动打分。</span></div></section><div id="comparison">${state.comparison?.length?`<section class="panel comparison-panel"><div class="panel-head"><h2>并排对比</h2><button class="secondary" id="closeComparison">收起对比</button></div><div class="comparison-grid">${state.comparison.map(r=>pelicanResult(r)).join('')}</div></section>`:''}</div>${performanceHistory()}<section class="panel report-history"><div class="panel-head"><h2>标准评测分析报告</h2><span class="muted">${reports.length} 份</span></div><div class="report-list">${reports.length?reports.map(r=>`<div class="report-card"><div><h3>${esc(r.title||r.model)}</h3><p>${esc(r.created_at)}</p></div><button class="secondary" data-report="${esc(r.id)}">查看报告</button></div>`).join(''):'<div class="empty">还没有分析报告。完成标准评测后，在结果中生成报告。</div>'}</div></section>${state.report?`<section class="panel report-history" id="reportDetail">${state.report.kind==='pelican'?pelicanResult(state.report):state.report.kind==='performance'?renderPerformanceReport(state.report):`<h2>${esc(state.report.title)}</h2><div class="report-content">${esc(state.report.content)}</div>`}</section>`:''}`;
 bindPerformanceReports();$('#refreshReports').onclick=()=>loadReports();$('#newPelican').onclick=()=>{state.evalMode='pelican';navigate('evaluate')};document.querySelectorAll('[data-report]').forEach(btn=>btn.onclick=()=>openReport(btn.dataset.report));
 document.querySelectorAll('[data-compare]').forEach(input=>input.onchange=()=>{const id=input.dataset.compare;if(input.checked){if(state.compareIDs.length>=2){input.checked=false;showToast('最多选择两次结果');return}state.compareIDs.push(id)}else state.compareIDs=state.compareIDs.filter(x=>x!==id);const button=$('#comparePelicans');button.disabled=state.compareIDs.length!==2;button.textContent=`并排对比 (${state.compareIDs.length}/2)`});
 $('#comparePelicans').onclick=async()=>{try{const rows=await Promise.all(state.compareIDs.map(id=>request(`/api/reports/${encodeURIComponent(id)}`)));state.comparison=rows;if(state.view==='reports'){render();$('#comparison').scrollIntoView({behavior:'smooth'})}}catch(e){showToast(e.message)}};
 if($('#closeComparison'))$('#closeComparison').onclick=()=>{state.comparison=null;render()};
}

function render(){document.querySelectorAll('.nav button').forEach(btn=>{const active=btn.dataset.view===state.view;btn.classList.toggle('selected',active);if(active)btn.setAttribute('aria-current','page');else btn.removeAttribute('aria-current')});if(state.view==='settings')renderSettingsView();else if(state.view==='reports')renderReportsView();else if(state.view==='evaluate')renderEvaluateView();else if(state.view==='performance')renderPerformanceView();else if(state.view==='environment')renderEnvironmentView();else renderRunsView();renderRuns()}
async function selectRun(id){state.selected=id;state.view='runs';try{const detail=await request(`/api/runs/${encodeURIComponent(id)}`);if(state.selected!==id||state.view!=='runs')return;state.detail=detail;render()}catch(e){showToast(e.message)}}
async function loadRuns(){try{state.runs=await request('/api/runs');renderRuns();if(state.view==='runs'){const id=state.runs.some(r=>r.id===state.selected)?state.selected:state.runs[0]?.id;if(id)await selectRun(id);else{state.detail=null;state.view='evaluate';render()}}}catch(e){showToast(e.message)}}
async function loadReports(){try{state.reports=await request('/api/reports');render()}catch(e){showError(e)}}
async function openReport(id){try{state.report=await request(`/api/reports/${encodeURIComponent(id)}`);if(state.view==='reports'){render();$('#reportDetail')?.scrollIntoView({behavior:'smooth'})}}catch(e){showError(e)}}
async function generateReport(){if(!state.selected){showToast('请先选择评测运行');return}const button=$('#generateReport');button.disabled=true;button.textContent='正在请求…';try{const report=await request('/api/reports/generate',{method:'POST',body:JSON.stringify({run_id:state.selected,instruction:$('#reportInstruction').value})});state.report=report;await loadReports();state.view='reports';render();showToast('报告已生成')}catch(e){showError(e)}finally{if(button){button.disabled=false;button.textContent='✦ 请求生成报告'}}}
async function navigate(view){
 state.view=view;render();
 if(view==='evaluate')await loadEvaluate();
 else if(view==='reports')await loadReports();
 else if(view==='performance')await loadPerformance();
 else if(view==='environment')await loadEnvironment();
}
document.querySelectorAll('.nav button').forEach(btn=>btn.onclick=()=>navigate(btn.dataset.view));
async function boot(){try{
 const [settings,runs,reports,datasets,collections,evalStatus,metadata]=await Promise.all([...['/api/settings','/api/runs','/api/reports','/api/datasets','/api/collections','/api/evaluation/status'].map(url=>request(url)),request('/api/benchmark-metadata').catch(()=>null)]);
 Object.assign(state,{settings,runs,reports,datasets,collections,evalStatus});acceptBenchmarkMetadata(metadata);const latest=reports.find(r=>r.kind==='pelican');if(latest){state.pelican=await request(`/api/reports/${encodeURIComponent(latest.id)}`);if(state.pelican.model===settings.model){if([8192,16384,32768,65536,131072,262144].includes(state.pelican.max_tokens))state.pelicanMaxTokens=state.pelican.max_tokens;state.pelicanThinkingMode=state.pelican.thinking_mode==='disabled'?'disabled':'default'}pollPelican()}render();if(evalStatus.status==='running')pollEvaluation();loadPerformance();
}catch(e){showError(e)}}boot();
</script></body></html>'''


class Handler(BaseHTTPRequestHandler):
    def _send(self, body, content_type="application/json", status=HTTPStatus.OK, download_name=None):
        payload = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header('X-Content-Type-Options', 'nosniff')
        if download_name:
            self.send_header('Content-Disposition', f'attachment; filename="{download_name}"')
        if content_type.startswith('image/svg+xml'):
            self.send_header('Content-Security-Policy', "sandbox; default-src 'none'; style-src 'unsafe-inline'")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):  # noqa: N802
        route = unquote(urlparse(self.path).path)
        if route == "/" or route == "/index.html":
            return self._send(PAGE, "text/html; charset=utf-8")
        if route in ('/assets/performance_ui.js', '/assets/performance_ui.css', '/assets/benchmark_ui.js', '/assets/benchmark_ui.css', '/assets/environment_ui.js', '/assets/environment_ui.css'):
            asset = ROOT / route.rsplit('/', 1)[1]
            content_type = 'text/javascript; charset=utf-8' if asset.suffix == '.js' else 'text/css; charset=utf-8'
            return self._send(asset.read_text(encoding='utf-8'), content_type)
        if route == '/api/environment/status':
            return self._send(json.dumps(ENVIRONMENT.status(), ensure_ascii=False))
        if route == '/api/performance/status':
            return self._send(json.dumps(PERFORMANCE.status(), ensure_ascii=False))
        if route.startswith('/api/performance/') and route.endswith('/export'):
            report = report_detail(route[len('/api/performance/'):-len('/export')])
            if not report or report.get('kind') != 'performance':
                return self._send('{"error":"performance report not found"}', status=HTTPStatus.NOT_FOUND)
            safe_id = re.sub(r'[^a-zA-Z0-9_-]', '-', str(report['id']))
            if parse_qs(urlparse(self.path).query).get('format') == ['csv']:
                return self._send(report_to_csv(report), 'text/csv; charset=utf-8', download_name=f'{safe_id}.csv')
            return self._send(json.dumps(report, ensure_ascii=False, indent=2), download_name=f'{safe_id}.json')
        if route == "/api/runs":
            return self._send(json.dumps(discover_runs(), ensure_ascii=False))
        if route.startswith("/api/runs/"):
            detail = run_detail(route.split("/", 3)[3])
            if detail is None:
                return self._send('{"error":"run not found"}', status=HTTPStatus.NOT_FOUND)
            return self._send(json.dumps(detail, ensure_ascii=False))
        if route.startswith('/api/pelican/') and route.endswith('/svg'):
            report = report_detail(route[len('/api/pelican/'):-len('/svg')])
            if not report or report.get('kind') != 'pelican' or report.get('status') != 'completed':
                return self._send('{"error":"SVG not found"}', status=HTTPStatus.NOT_FOUND)
            try:
                svg, _ = sanitize_svg(report.get('svg', ''))
            except ValueError:
                return self._send('{"error":"Invalid SVG"}', status=HTTPStatus.BAD_REQUEST)
            download_name = None
            if parse_qs(urlparse(self.path).query).get('download') == ['1']:
                safe_id = re.sub(r'[^a-zA-Z0-9_-]', '-', str(report['id']))
                download_name = f'pelican-{safe_id}.svg'
            return self._send(svg, 'image/svg+xml; charset=utf-8', download_name=download_name)
        if route == "/api/settings":
            return self._send(json.dumps(public_settings(), ensure_ascii=False))
        if route == "/api/datasets":
            return self._send(json.dumps(dataset_catalog(), ensure_ascii=False))
        if route == "/api/benchmark-metadata":
            return self._send(json.dumps({"categories": BENCHMARK_CATEGORIES, "presets": BENCHMARK_PRESETS}, ensure_ascii=False))
        if route == "/api/collections":
            return self._send(json.dumps(load_collections(), ensure_ascii=False))
        if route == "/api/evaluation/status":
            return self._send(json.dumps(_eval_public_state(), ensure_ascii=False))
        if route == "/api/reports":
            return self._send(json.dumps(discover_reports(), ensure_ascii=False))
        if route.startswith("/api/reports/"):
            detail = report_detail(route.split("/", 3)[3])
            if detail is None:
                return self._send('{"error":"report not found"}', status=HTTPStatus.NOT_FOUND)
            return self._send(json.dumps(detail, ensure_ascii=False))
        return self._send('{"error":"not found"}', status=HTTPStatus.NOT_FOUND)

    def _json_body(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, json.JSONDecodeError):
            raise ValueError("请求 JSON 格式无效")

    def do_POST(self):  # noqa: N802
        route = unquote(urlparse(self.path).path)
        if route == '/api/environment/install' and not _local_environment_request(
                self.client_address, self.headers, self.server.server_address[1]):
            return self._send('{"error":"环境安装仅接受本机页面的 JSON 请求"}', status=HTTPStatus.FORBIDDEN)
        try:
            payload = self._json_body()
            if not isinstance(payload, dict):
                raise ValueError('请求必须为 JSON 对象')
            if route == '/api/environment/install':
                if set(payload) != {'action'}:
                    raise ValueError('环境操作只接受 action 参数')
                return self._send(json.dumps(_start_environment(payload['action']), ensure_ascii=False), status=HTTPStatus.ACCEPTED)
            if route == '/api/performance/start':
                return self._send(json.dumps(_start_activity(lambda: PERFORMANCE.start(load_settings(), payload)), ensure_ascii=False), status=HTTPStatus.ACCEPTED)
            if route == '/api/performance/stop':
                return self._send(json.dumps(PERFORMANCE.stop(), ensure_ascii=False))
            if route == '/api/pelican/start':
                return self._send(json.dumps(_start_activity(lambda: PELICAN.start(load_settings(), payload.get("max_tokens", MAX_TOKENS), payload.get("thinking_mode", "default"))), ensure_ascii=False), status=HTTPStatus.ACCEPTED)
            if route == '/api/pelican/stop':
                return self._send(json.dumps(PELICAN.stop(), ensure_ascii=False))
            if route == "/api/settings":
                return self._send(json.dumps(save_settings(payload), ensure_ascii=False))
            if route == "/api/collections":
                return self._send(json.dumps(save_collection(payload), ensure_ascii=False))
            if route == "/api/evaluation/start":
                return self._send(json.dumps(_start_activity(lambda: start_evaluation(payload.get("datasets", []), payload.get("options"))), ensure_ascii=False), status=HTTPStatus.ACCEPTED)
            if route == "/api/reports/generate":
                report = generate_report(str(payload.get("run_id", "")), str(payload.get("instruction", "")))
                return self._send(json.dumps(report, ensure_ascii=False))
            return self._send('{"error":"not found"}', status=HTTPStatus.NOT_FOUND)
        except (ValueError, RuntimeError) as error:
            return self._send(json.dumps({"error": str(error)}, ensure_ascii=False), status=HTTPStatus.BAD_REQUEST)

    def log_message(self, fmt, *args):
        print(f"[web] {self.address_string()} - {fmt % args}")


def main():
    global OUTPUTS
    parser = argparse.ArgumentParser(description="Browse OpenCompass outputs in a local web page")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--outputs", type=Path, default=OUTPUTS)
    args = parser.parse_args()
    OUTPUTS = args.outputs.expanduser().resolve()
    print(f"OpenCompass viewer: http://{args.host}:{args.port}")
    print(f"Scanning: {OUTPUTS}")
    PELICAN.recover_interrupted()
    PERFORMANCE.recover_interrupted()
    ENVIRONMENT.recover_interrupted()
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
