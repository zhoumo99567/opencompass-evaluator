#!/usr/bin/env python3
"""Small local web UI for browsing OpenCompass output directories."""

from __future__ import annotations

import argparse
import csv
import json
import re
import os
import secrets
import subprocess
import threading
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
REPORTS = ROOT / "reports"
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

# Keep the first-run picker focused on small, text-only configs that are useful
# for checking an OpenAI-compatible LAN endpoint. More configs can be added to
# a collection by editing this catalog later without changing the UI contract.
DATASET_CATALOG = [
    {"id": "demo_gsm8k_chat_gen", "label": "Demo GSM8K", "category": "数学", "description": "小学数学推理，生成式评分"},
    {"id": "demo_math_chat_gen", "label": "Demo MATH", "category": "数学", "description": "竞赛数学，生成式评分"},
    {"id": "demo_cmmlu_chat_gen", "label": "Demo CMMLU", "category": "知识", "description": "中文知识问答，生成式评分"},
    {"id": "demo_gsm8k_base_gen", "label": "Demo GSM8K Base", "category": "数学", "description": "GSM8K 基础配置"},
    {"id": "demo_math_base_gen", "label": "Demo MATH Base", "category": "数学", "description": "MATH 基础配置"},
    {"id": "demo_cmmlu_base_ppl", "label": "Demo CMMLU PPL", "category": "知识", "description": "CMMLU 困惑度评测"},
    {"id": "humaneval_cn_gen", "label": "HumanEval CN", "category": "代码", "description": "中文代码生成能力"},
]


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
    return [dict(item, available=item["id"] in available) for item in DATASET_CATALOG]


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
            EVAL_STATE["process"] = None
        return {key: value for key, value in EVAL_STATE.items() if key != "process"}


def start_evaluation(dataset_ids):
    selected = list(dict.fromkeys(str(value) for value in dataset_ids))
    catalog_ids = {item["id"] for item in DATASET_CATALOG}
    selected = [value for value in selected if value in catalog_ids]
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
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = OUTPUTS / f"evaluation_{stamp}.log"
        command = [
            str(ROOT / ".venv" / "bin" / "opencompass"),
            str(ROOT / "lan_eval.py"),
            "--datasets",
            *selected,
            "-w",
            str(OUTPUTS / "web"),
        ]
        environment = os.environ.copy()
        environment.update({
            "OPENCOMPASS_BASE_URL": settings["base_url"],
            "OPENCOMPASS_MODEL": settings["model"],
            "OPENAI_API_KEY": settings["api_key"],
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
            "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "finished_at": None,
            "returncode": None,
            "log_path": str(log_path.relative_to(ROOT)),
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
    extra = instruction.strip() or "请重点说明模型表现、明显错误模式和下一步优化建议。"
    prompt = f"""你是一名 LLM 评测分析师。请根据下面的 OpenCompass 评测数据生成一份中文 Markdown 报告。

评测任务：{run['id']}
模型：{run['model']}
样本数：{run['samples']}

分数：
{score_lines}

样本摘录：
{sample_lines}

用户补充要求：{extra}

报告请包含：结论摘要、分数解读、错误模式、代表性样本、风险与局限、可执行的优化建议。不要编造数据；没有证据时明确写“数据不足”。"""
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
        report = read_json(path)
        if isinstance(report, dict) and report.get("id"):
            reports.append({key: report.get(key, "") for key in ("id", "created_at", "run_id", "model", "title")})
    return reports


def report_detail(report_id):
    safe = Path(report_id)
    if safe.name != report_id or ".." in safe.parts:
        return None
    return read_json(REPORTS / f"{report_id}.json")


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
            rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
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
                if score <= 1:
                    score *= 100
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
    return {
        "id": run_dir.name,
        "timestamp": run_dir.name,
        "model": model_name(run_dir),
        "status": "completed" if has_results else ("in progress" if predictions else "empty"),
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
    data["files"] = files
    data["samples"] = prediction_rows(run_dir)
    return data


PAGE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>OpenCompass Runs</title>
<style>
:root{--ink:#f1f0e9;--muted:#8e969a;--bg:#171b1d;--rail:#121618;--panel:#f5f4ee;--panel2:#ebeae3;--dark:#22282a;--lime:#a6ff44;--lime2:#6ed52c;--line:#d7d6ce;--danger:#ff7180;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);min-height:100vh}button,input,textarea{font:inherit}.shell{display:grid;grid-template-columns:280px 1fr;min-height:100vh}.rail{background:var(--rail);padding:26px 16px 20px;display:flex;flex-direction:column;border-right:1px solid #293032}.brand{font-size:22px;font-weight:760;letter-spacing:-.03em;padding:0 12px 24px}.brand span{color:var(--lime)}.rail-label{color:#687175;font-size:12px;text-transform:uppercase;letter-spacing:.12em;margin:16px 12px 8px}.run-list{display:flex;flex-direction:column;gap:4px;overflow:auto;flex:1;min-height:0}.run{border:1px solid transparent;border-radius:12px;background:transparent;color:var(--ink);text-align:left;padding:12px;cursor:pointer}.run:hover{background:#202729}.run.active{background:#2b3234;border-color:#3b4446}.run-title{font-size:13px;font-weight:650;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.run-meta{font-size:11px;color:var(--muted);margin-top:5px;display:flex;justify-content:space-between;gap:6px}.status{color:var(--lime)}.status.in-progress{color:#f0c76a}.status.failed{color:var(--danger)}.nav{margin-top:auto;border-top:1px solid #293032;padding-top:12px}.nav button{border:0;background:none;color:#a9b0b2;width:100%;text-align:left;padding:11px 12px;border-radius:9px;cursor:pointer}.nav button:hover,.nav button.selected{background:#242a2c;color:var(--ink)}main{padding:34px 42px 56px;min-width:0}.topbar{display:flex;align-items:center;justify-content:space-between;gap:20px;margin:0 auto 18px;max-width:1220px}.crumb{color:#aab2b4;font-size:14px}.crumb b{color:var(--ink)}.actions{display:flex;gap:8px}.refresh{background:transparent;border:1px solid #455052;color:var(--ink);border-radius:8px;padding:9px 13px;cursor:pointer}.secondary{background:transparent;border:1px solid #9aa09a;color:#1f2425;border-radius:8px;padding:9px 13px;cursor:pointer}.refresh:hover{border-color:var(--lime);color:var(--lime)}.secondary:hover{border-color:var(--lime2);color:#1f2425;background:#e4f5cf}.primary{background:var(--dark);border:1px solid var(--dark);color:var(--lime);border-radius:8px;padding:10px 14px;cursor:pointer;font-weight:650}.primary:hover{background:#101516}.primary:disabled{opacity:.55;cursor:wait}.hero,.panel,.stat{background:var(--panel);color:#1f2425;border-radius:16px}.hero{padding:28px 32px;margin:0 auto 16px;max-width:1220px}.hero h1{font-family:Georgia,"Times New Roman",serif;font-size:42px;letter-spacing:-.045em;line-height:1.05;margin:0 0 14px}.hero p{margin:0;line-height:1.65}.hero-meta{display:flex;gap:28px;flex-wrap:wrap}.hero-meta label{display:block;font-size:11px;color:#778083;text-transform:uppercase;letter-spacing:.08em}.hero-meta strong{display:block;margin-top:6px;font-size:14px}.stats{max-width:1220px;margin:0 auto 16px;display:grid;grid-template-columns:repeat(4,1fr);gap:16px}.stat{padding:20px 22px}.stat-label{font-size:12px;color:#6c7476}.stat-value{font-family:Georgia,serif;font-size:35px;letter-spacing:-.04em;margin-top:12px}.stat-sub{font-size:12px;color:#6c7476;margin-top:4px}.content{max-width:1220px;margin:0 auto;display:grid;grid-template-columns:1fr 1fr;gap:16px}.panel{padding:24px}.panel.full{grid-column:1/-1}.panel-head{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:20px}.panel h2{font-family:Georgia,serif;font-size:22px;letter-spacing:-.03em;margin:0}.muted{color:#7c8587;font-size:12px}.empty{color:#7c8587;padding:20px 0;line-height:1.6}.cta{border:1px dashed #b9bab2;background:#eeeee8;border-radius:12px;padding:24px;margin-top:18px}.cta h3{margin:0 0 8px;font-size:17px}.bar-row{display:grid;grid-template-columns:110px 1fr 54px;align-items:center;gap:12px;margin:17px 0}.bar-label{font-size:13px;font-weight:650}.bar-track{height:12px;background:#d8d9d5;border-radius:99px;overflow:hidden}.bar-fill{height:100%;background:linear-gradient(90deg,var(--lime),var(--lime2));border-radius:99px}.bar-score{text-align:right;font-family:Georgia,serif;font-size:18px}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:10px}.samples{border-collapse:collapse;width:100%;min-width:720px;font-size:12px}.samples th,.samples td{padding:12px 11px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}.samples th{color:#737b7d;font-weight:600;background:#eeeee8}.samples tr:last-child td{border-bottom:0}.samples td:nth-child(1){color:#8a9293;width:34px}.samples td:nth-child(4),.samples td:nth-child(5){max-width:280px;white-space:pre-wrap}.sample-prompt{color:#596163;white-space:pre-wrap;max-width:300px}.files{display:flex;flex-wrap:wrap;gap:8px}.file{background:#e6e6df;border-radius:7px;padding:7px 9px;color:#596163;font-size:11px}.search{width:220px;border:1px solid #c9c9c1;background:#faf9f3;border-radius:8px;padding:9px 11px;outline:0}.search:focus,.field input:focus,.field textarea:focus{border-color:var(--lime2)}.form-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.field{display:flex;flex-direction:column;gap:7px}.field.full{grid-column:1/-1}.field label{font-size:12px;color:#6c7476}.field input,.field textarea{border:1px solid #c9c9c1;background:#faf9f3;color:#1f2425;border-radius:8px;padding:11px 12px;outline:0}.field textarea{min-height:110px;resize:vertical}.form-actions{display:flex;align-items:center;gap:12px;margin-top:20px}.hint{font-size:12px;color:#7c8587;line-height:1.6}.report-list{display:grid;gap:10px}.report-card{display:flex;align-items:center;justify-content:space-between;gap:16px;border:1px solid var(--line);background:#eeeee8;border-radius:10px;padding:14px 16px}.report-card h3{margin:0 0 5px;font-size:14px}.report-card p{margin:0;color:#7c8587;font-size:12px}.report-content{white-space:pre-wrap;line-height:1.7;font-size:14px;color:#2c3334;background:#faf9f3;border-radius:10px;padding:22px;overflow:auto}.error{color:#9d3038;background:#ffe3e4;padding:12px;border-radius:8px}.toast{position:fixed;bottom:20px;right:20px;background:#303a3c;color:var(--ink);padding:10px 14px;border-radius:9px;box-shadow:0 10px 30px #0005;display:none;z-index:4}.dataset-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.dataset-card{border:1px solid var(--line);background:#eeeee8;border-radius:11px;padding:14px;cursor:pointer;transition:.15s}.dataset-card:hover{border-color:#96be6c}.dataset-card.selected{border-color:var(--lime2);background:#e4f5cf;box-shadow:inset 0 0 0 1px var(--lime2)}.dataset-card input{accent-color:#5c9d28;margin-right:8px}.dataset-title{font-weight:700;font-size:14px}.dataset-meta{font-size:11px;color:#788082;margin-top:7px;line-height:1.5}.collection-list{display:grid;gap:9px}.collection-card{display:flex;align-items:center;justify-content:space-between;gap:12px;border:1px solid var(--line);background:#eeeee8;border-radius:10px;padding:13px}.collection-card h3{font-size:14px;margin:0 0 5px}.collection-card p{margin:0;color:#7c8587;font-size:12px}.check-row{display:flex;align-items:center;gap:8px;font-size:12px;color:#596163}.status-box{display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:12px 14px;border-radius:9px;background:#e9e9e2;font-size:13px}.status-dot{width:9px;height:9px;border-radius:50%;background:#8b9495}.status-dot.running{background:#e4b240;box-shadow:0 0 0 4px #e4b24033}.status-dot.completed{background:#66ad31}.status-dot.failed{background:#d85261}@media(max-width:900px){.shell{grid-template-columns:1fr}.rail{position:static;padding:16px}.run-list{display:flex;flex-direction:row;overflow:auto}.run{min-width:180px}.rail-label{display:none}.nav{display:flex;border-top:0;padding-top:0;gap:4px;overflow:auto}.nav button{width:auto;white-space:nowrap}main{padding:22px 16px 40px}.stats{grid-template-columns:repeat(2,1fr)}.content,.form-grid,.dataset-grid{grid-template-columns:1fr}.panel.full,.field.full{grid-column:auto}.hero h1{font-size:32px}}
</style>
</head>
<body>
<div class="shell"><aside class="rail"><div class="brand">OpenCompass <span>Runs</span></div><div class="rail-label">Experiments</div><div id="runList" class="run-list"></div><div class="nav"><button data-view="runs" class="selected">▣ &nbsp; 评测运行</button><button data-view="evaluate">▶ &nbsp; 开始评测</button><button data-view="reports">▥ &nbsp; 历史报告</button><button data-view="settings">⚙ &nbsp; 模型设置</button></div></aside>
<main><div class="topbar"><div class="crumb">OpenCompass&nbsp; / &nbsp;<b id="crumb">Loading…</b></div><div class="actions" id="actions"></div></div><div id="app"></div></main></div><div class="toast" id="toast"></div>
<script>
const state={runs:[],selected:null,detail:null,settings:null,reports:[],report:null,datasets:[],collections:[],selectedDatasets:[],evalStatus:null,view:'runs'};
const $=s=>document.querySelector(s);
function esc(v){return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function fmtDate(v){if(!/^\d{8}_\d{6}$/.test(v))return v;return `${v.slice(0,4)}-${v.slice(4,6)}-${v.slice(6,8)} ${v.slice(9,11)}:${v.slice(11,13)}`}
async function request(url,options={}){const r=await fetch(url,{headers:{'Content-Type':'application/json',...(options.headers||{})},...options});const data=await r.json().catch(()=>({}));if(!r.ok)throw Error(data.error||'请求失败');return data}
function showToast(text){const el=$('#toast');el.textContent=text;el.style.display='block';setTimeout(()=>el.style.display='none',2200)}
function showError(error){$('#app').innerHTML=`<section class="panel"><div class="error">${esc(error.message||error)}</div></section>`}
function renderRuns(){const el=$('#runList');el.innerHTML=state.runs.length?state.runs.map(r=>`<button class="run ${r.id===state.selected?'active':''}" data-id="${esc(r.id)}"><div class="run-title">${esc(r.model)}</div><div class="run-meta"><span>${esc(fmtDate(r.timestamp))}</span><span class="status ${esc(r.status.replace(/\s+/g,'-'))}">${esc(r.status)}</span></div></button>`).join(''):`<div class="empty">outputs/ 下暂无评测</div>`;el.querySelectorAll('.run').forEach(x=>x.onclick=()=>{state.view='runs';selectRun(x.dataset.id);});}
function scoreCards(scores,samples){const vals=scores.map(s=>s.score).filter(Number.isFinite);const avg=vals.length?vals.reduce((a,b)=>a+b,0)/vals.length:null;const best=vals.length?Math.max(...vals):null;const low=vals.length?Math.min(...vals):null;return `<div class="stats"><div class="stat"><div class="stat-label">平均分</div><div class="stat-value">${avg==null?'—':avg.toFixed(1)}</div><div class="stat-sub">${scores.length||0} 个 benchmark</div></div><div class="stat"><div class="stat-label">最高分</div><div class="stat-value">${best==null?'—':best.toFixed(1)}</div><div class="stat-sub">${best==null?'尚无 summary':esc((scores.find(s=>s.score===best)||{}).dataset||'')}</div></div><div class="stat"><div class="stat-label">最低分</div><div class="stat-value">${low==null?'—':low.toFixed(1)}</div><div class="stat-sub">${low==null?'仅有预测记录':esc((scores.find(s=>s.score===low)||{}).dataset||'')}</div></div><div class="stat"><div class="stat-label">样本数</div><div class="stat-value">${samples}</div><div class="stat-sub">预测记录</div></div></div>`}
function renderRunsView(){const d=state.detail;$('#crumb').textContent=d?d.model:'评测运行';$('#actions').innerHTML='<button class="refresh" id="refresh">↻ 刷新</button>';if(!d){$('#app').innerHTML='<section class="hero"><h1>还没有评测运行</h1><p class="muted">当前没有报告和评测结果。选择评测集后，可以从网页直接启动一次 OpenCompass 评测。</p><div class="cta"><h3>从空状态开始</h3><p class="muted">勾选一个或多个评测集，也可以先保存成集合，点击开始后页面会显示进程状态。</p><button class="primary" id="goEvaluate">▶ 选择评测集并开始</button></div></section>';$('#goEvaluate').onclick=()=>{state.view='evaluate';loadEvaluate()};return}const scores=d.scores||[];const samples=d.samples||[];const bars=scores.length?scores.map(s=>`<div class="bar-row"><div class="bar-label">${esc(s.dataset)}</div><div class="bar-track"><div class="bar-fill" style="width:${Math.max(0,Math.min(100,s.score))}%"></div></div><div class="bar-score">${Number(s.score).toFixed(1)}</div></div>`).join(''):'<div class="empty">这次运行已有预测，但尚未生成 summary 分数。</div>';const rows=samples.slice(0,30).map((s,i)=>`<tr><td>${esc(s.idx??i)}</td><td>${esc(s.dataset)}</td><td class="sample-prompt">${esc(s.prompt)}</td><td>${esc(s.prediction)}</td><td>${esc(s.gold)}</td></tr>`).join('');$('#app').innerHTML=`<section class="hero"><h1>${esc(d.model)}</h1><div class="hero-meta"><div><label>状态</label><strong>${esc(d.status)}</strong></div><div><label>运行时间</label><strong>${esc(fmtDate(d.timestamp))}</strong></div><div><label>行数</label><strong>${samples.length}</strong></div><div><label>来源</label><strong>OpenCompass outputs</strong></div></div></section>${scoreCards(scores,samples.length)}<div class="content"><section class="panel"><div class="panel-head"><h2>Benchmark 结果</h2><span class="muted">分数越高越好</span></div>${bars}</section><section class="panel"><div class="panel-head"><h2>运行文件</h2><span class="muted">${d.files.length} 个文件</span></div><div class="files">${d.files.slice(0,40).map(f=>`<span class="file">${esc(f)}</span>`).join('')||'<span class="empty">暂无文件</span>'}</div></section><section class="panel"><div class="panel-head"><h2>生成分析报告</h2><span class="muted">调用已保存的本地模型接口</span></div><div class="field"><label for="reportInstruction">报告要求（可选）</label><textarea id="reportInstruction" placeholder="例如：重点分析数学题错误，并给出提示词优化建议"></textarea></div><div class="form-actions"><button class="primary" id="generateReport">✦ 请求生成报告</button><span class="hint">报告会保存到本地历史记录。</span></div></section><section class="panel"><div class="panel-head"><h2>最近样本</h2><input id="search" class="search" placeholder="筛选样本" /></div><div class="table-wrap"><table class="samples"><thead><tr><th>#</th><th>数据集</th><th>问题</th><th>模型输出</th><th>标准答案</th></tr></thead><tbody id="sampleRows">${rows||'<tr><td colspan="5" class="empty">暂无预测记录。</td></tr>'}</tbody></table></div></section></div>`;$('#refresh').onclick=()=>{state.selected=null;loadRuns();showToast('已刷新')};const search=$('#search');if(search)search.oninput=()=>{const q=search.value.toLowerCase();$('#sampleRows').querySelectorAll('tr').forEach(row=>row.style.display=row.textContent.toLowerCase().includes(q)?'':'none')};$('#generateReport').onclick=generateReport;}
function collectionNames(c){return c.datasets.map(id=>{const item=state.datasets.find(d=>d.id===id);return item?item.label:id}).join('、')}
function renderEvaluateView(){state.view='evaluate';$('#crumb').textContent='开始评测';$('#actions').innerHTML='<button class="refresh" id="refreshEvaluate">↻ 刷新</button>';const status=state.evalStatus||{};const running=status.status==='running';const selected=new Set(state.selectedDatasets);const cards=state.datasets.filter(d=>d.available).map(d=>`<label class="dataset-card ${selected.has(d.id)?'selected':''}"><div><input type="checkbox" data-dataset="${esc(d.id)}" ${selected.has(d.id)?'checked':''}/><span class="dataset-title">${esc(d.label)}</span></div><div class="dataset-meta">${esc(d.category)} · ${esc(d.id)}<br/>${esc(d.description)}</div></label>`).join('');const collections=state.collections.map(c=>`<div class="collection-card"><div><h3>${esc(c.name)}</h3><p>${esc(c.description||'')} · ${c.datasets.length} 个评测集<br/>${esc(collectionNames(c))}</p></div><button class="secondary" data-apply-collection="${esc(c.id)}">应用</button></div>`).join('');$('#app').innerHTML=`<div class="content"><section class="panel full"><div class="panel-head"><h2>选择评测集</h2><span class="muted">${state.datasets.filter(d=>d.available).length} 个可用配置</span></div><p class="hint">这些是当前 OpenCompass 安装中适合本地接口快速验证的文本评测集。可以多选后一次运行。</p><div class="dataset-grid">${cards||'<div class="empty">没有找到可用评测集。</div>'}</div><div class="form-actions"><button class="primary" id="startEvaluation" ${!selected.size||running?'disabled':''}>${running?'评测运行中…':'▶ 开始 OpenCompass 评测'}</button><button class="secondary" id="selectDemos" ${running?'disabled':''}>全选 Demo</button><span class="hint">已选 ${selected.size} 个</span></div></section><section class="panel"><div class="panel-head"><h2>我的评测集合</h2><span class="muted">保存常用组合</span></div><div class="collection-list">${collections||'<div class="empty">暂无集合。勾选评测集后可以在这里保存。</div>'}</div><div class="form-grid" style="margin-top:18px"><div class="field"><label for="collectionName">集合名称</label><input id="collectionName" placeholder="例如：快速回归" /></div><div class="field"><label for="collectionDescription">说明</label><input id="collectionDescription" placeholder="可选" /></div></div><div class="form-actions"><button class="secondary" id="saveCollection" ${!selected.size?'disabled':''}>保存当前选择</button></div></section><section class="panel"><div class="panel-head"><h2>运行状态</h2><span class="muted">网页启动的 OpenCompass 进程</span></div><div class="status-box"><span class="status-dot ${esc(status.status||'idle')}"></span><strong>${esc(status.status==='running'?'运行中':status.status==='completed'?'已完成':status.status==='failed'?'失败':'空闲')}</strong><span class="muted">${status.pid?'PID '+esc(status.pid):''}</span>${status.log_path?`<span class="muted">日志：${esc(status.log_path)}</span>`:''}</div><p class="hint" style="margin-top:12px">${status.status==='running'?'评测结束后刷新运行列表即可查看结果。':''}${status.status==='failed'?'请查看日志文件，确认本地接口地址、模型名和密钥。':''}</p></section></div>`;document.querySelectorAll('[data-dataset]').forEach(input=>input.onchange=()=>{state.selectedDatasets=[...document.querySelectorAll('[data-dataset]:checked')].map(x=>x.dataset.dataset);renderEvaluateView()});document.querySelectorAll('[data-apply-collection]').forEach(btn=>btn.onclick=()=>{const c=state.collections.find(x=>x.id===btn.dataset.applyCollection);if(c){state.selectedDatasets=[...c.datasets];renderEvaluateView();showToast('已应用集合')}});$('#refreshEvaluate').onclick=()=>loadEvaluate();$('#selectDemos').onclick=()=>{state.selectedDatasets=state.datasets.filter(d=>d.available&&d.id.startsWith('demo_')).map(d=>d.id);renderEvaluateView()};$('#saveCollection').onclick=saveCollection;$('#startEvaluation').onclick=startEvaluation;}
async function loadEvaluate(){try{const [datasets,collections,evalStatus]=await Promise.all([request('/api/datasets'),request('/api/collections'),request('/api/evaluation/status')]);state.datasets=datasets;state.collections=collections;state.evalStatus=evalStatus;renderEvaluateView()}catch(e){showError(e)}}
async function saveCollection(){const name=$('#collectionName').value.trim();if(!name){showToast('请填写集合名称');return}try{await request('/api/collections',{method:'POST',body:JSON.stringify({name,description:$('#collectionDescription').value,datasets:state.selectedDatasets})});state.collections=await request('/api/collections');showToast('集合已保存');renderEvaluateView()}catch(e){showError(e)}}
async function startEvaluation(){if(!state.selectedDatasets.length){showToast('请至少选择一个评测集');return}const button=$('#startEvaluation');button.disabled=true;button.textContent='正在启动…';try{state.evalStatus=await request('/api/evaluation/start',{method:'POST',body:JSON.stringify({datasets:state.selectedDatasets})});showToast('OpenCompass 已启动');renderEvaluateView();pollEvaluation()}catch(e){showError(e);button.disabled=false}}
let pollTimer=null;async function pollEvaluation(){clearInterval(pollTimer);pollTimer=setInterval(async()=>{try{state.evalStatus=await request('/api/evaluation/status');renderEvaluateView();if(state.evalStatus.status!=='running'){clearInterval(pollTimer);await loadRuns();showToast(state.evalStatus.status==='completed'?'评测已完成':'评测进程失败')}}catch(e){clearInterval(pollTimer);showError(e)}},3000)}
function renderSettingsView(){state.view='settings';$('#crumb').textContent='模型设置';$('#actions').innerHTML='';const s=state.settings||{};$('#app').innerHTML=`<section class="panel"><div class="panel-head"><h2>本地模型接口</h2><span class="muted">保存到当前项目目录</span></div><div class="form-grid"><div class="field full"><label for="baseUrl">接口地址</label><input id="baseUrl" value="${esc(s.base_url||'')}" placeholder="http://127.0.0.1:8080/v1 或完整 /chat/completions 地址" /></div><div class="field"><label for="model">模型名称</label><input id="model" value="${esc(s.model||'')}" placeholder="Qwen3.8-Flash-Next" /></div><div class="field"><label for="apiKey">API Key</label><input id="apiKey" type="password" placeholder="${s.api_key_set?`已保存 ${esc(s.api_key_hint)}，留空保持不变`:'请输入 API Key'}" /></div></div><div class="form-actions"><button class="primary" id="saveSettings">保存设置</button><span class="hint">密钥保存在 .viewer_settings.json（权限 600），不会显示在页面中。</span></div></section>`;$('#saveSettings').onclick=async()=>{const button=$('#saveSettings');button.disabled=true;try{state.settings=await request('/api/settings',{method:'POST',body:JSON.stringify({base_url:$('#baseUrl').value,model:$('#model').value,api_key:$('#apiKey').value})});showToast('设置已保存');renderSettingsView()}catch(e){showError(e)}finally{button.disabled=false}}}
function renderReportsView(){state.view='reports';$('#crumb').textContent='历史报告';$('#actions').innerHTML='<button class="refresh" id="refreshReports">↻ 刷新</button>';const list=state.reports;$('#app').innerHTML=`<section class="panel"><div class="panel-head"><h2>历史报告</h2><span class="muted">${list.length} 份</span></div><div class="report-list">${list.length?list.map(r=>`<div class="report-card"><div><h3>${esc(r.title||r.model)}</h3><p>${esc(r.created_at)} · 运行 ${esc(r.run_id)}</p></div><button class="secondary" data-report="${esc(r.id)}">查看报告</button></div>`).join(''):'<div class="empty">还没有报告。完成一次评测后选择运行并点击“请求生成报告”。</div>'}</div></section>${state.report?`<section class="panel" style="margin-top:16px"><div class="panel-head"><h2>${esc(state.report.title||'报告')}</h2><span class="muted">${esc(state.report.created_at||'')}</span></div><div class="report-content">${esc(state.report.content||'')}</div></section>`:''}`;$('#refreshReports').onclick=async()=>{await loadReports();showToast('已刷新')};document.querySelectorAll('[data-report]').forEach(btn=>btn.onclick=()=>openReport(btn.dataset.report));}
function render(){document.querySelectorAll('.nav button').forEach(btn=>btn.classList.toggle('selected',btn.dataset.view===state.view));if(state.view==='settings')renderSettingsView();else if(state.view==='reports')renderReportsView();else if(state.view==='evaluate')renderEvaluateView();else renderRunsView();renderRuns()}
async function selectRun(id){state.selected=id;renderRuns();try{state.detail=await request(`/api/runs/${encodeURIComponent(id)}`);render()}catch(e){showError(e)}}
async function loadRuns(){try{state.runs=await request('/api/runs');renderRuns();if(!state.selected&&state.runs[0])state.selected=state.runs[0].id;if(state.selected)await selectRun(state.selected);else{state.detail=null;render()}}catch(e){showError(e)}}
async function loadReports(){try{state.reports=await request('/api/reports');render()}catch(e){showError(e)}}
async function openReport(id){try{state.report=await request(`/api/reports/${encodeURIComponent(id)}`);renderReportsView()}catch(e){showError(e)}}
async function generateReport(){if(!state.selected){showToast('请先选择评测运行');return}const button=$('#generateReport');button.disabled=true;button.textContent='正在请求…';try{const report=await request('/api/reports/generate',{method:'POST',body:JSON.stringify({run_id:state.selected,instruction:$('#reportInstruction').value})});state.report=report;await loadReports();state.view='reports';render();showToast('报告已生成')}catch(e){showError(e)}finally{if(button){button.disabled=false;button.textContent='✦ 请求生成报告'}}}
document.querySelectorAll('.nav button').forEach(btn=>btn.onclick=async()=>{state.view=btn.dataset.view;if(state.view==='settings'&&!state.settings)state.settings=await request('/api/settings');if(state.view==='reports')await loadReports();if(state.view==='evaluate')await loadEvaluate();render()});
async function boot(){try{state.settings=await request('/api/settings');state.runs=await request('/api/runs');state.reports=await request('/api/reports');renderRuns();if(state.runs[0]){state.selected=state.runs[0].id;await selectRun(state.selected)}else render()}catch(e){showError(e)}}boot();
</script></body></html>'''


class Handler(BaseHTTPRequestHandler):
    def _send(self, body, content_type="application/json", status=HTTPStatus.OK):
        payload = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):  # noqa: N802
        route = unquote(urlparse(self.path).path)
        if route == "/" or route == "/index.html":
            return self._send(PAGE, "text/html; charset=utf-8")
        if route == "/api/runs":
            return self._send(json.dumps(discover_runs(), ensure_ascii=False))
        if route.startswith("/api/runs/"):
            detail = run_detail(route.split("/", 3)[3])
            if detail is None:
                return self._send('{"error":"run not found"}', status=HTTPStatus.NOT_FOUND)
            return self._send(json.dumps(detail, ensure_ascii=False))
        if route == "/api/settings":
            return self._send(json.dumps(public_settings(), ensure_ascii=False))
        if route == "/api/datasets":
            return self._send(json.dumps(dataset_catalog(), ensure_ascii=False))
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
        try:
            payload = self._json_body()
            if route == "/api/settings":
                return self._send(json.dumps(save_settings(payload), ensure_ascii=False))
            if route == "/api/collections":
                return self._send(json.dumps(save_collection(payload), ensure_ascii=False))
            if route == "/api/evaluation/start":
                return self._send(json.dumps(start_evaluation(payload.get("datasets", [])), ensure_ascii=False), status=HTTPStatus.ACCEPTED)
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
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
