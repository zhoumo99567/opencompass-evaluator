# OpenCompass LAN Evaluator

一个面向局域网本地大模型的 OpenCompass 评测与报告查看器。它使用 OpenAI-compatible API 连接本地模型，并提供一个本地网页来选择评测集、保存评测集合、启动评测、查看运行结果和生成历史报告。

## 快速开始

需要 Python 3.10+、已安装 `uv`，以及一个兼容 OpenAI Chat Completions 的模型接口。

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python opencompass
cp .env.example .env
```

编辑 `.env`：

```dotenv
OPENCOMPASS_BASE_URL=http://127.0.0.1:8080/v1/chat/completions
OPENCOMPASS_MODEL=your-model-name
OPENAI_API_KEY=your-api-key
OPENCOMPASS_QPS=1
```

启动网页：

```bash
./run_viewer.sh --host 127.0.0.1 --port 8765
```

打开 <http://127.0.0.1:8765/>，在“开始评测”中勾选评测集并启动 OpenCompass。评测结果会写入 `outputs/`，报告会写入 `reports/`；这些目录默认不会提交到 Git。

也可以直接运行 OpenCompass：

```bash
./run_opencompass.sh lan_eval.py --datasets demo_gsm8k_chat_gen
```

## 安全说明

`.env`、网页保存的设置、评测输出、报告和运行时缓存都已加入 `.gitignore`。提交代码前请确认不要把 API key 写入 Python 文件、日志或报告。

## 当前网页能力

- 选择一个或多个 Demo GSM8K、MATH、CMMLU、HumanEval CN 等评测集
- 保存、查看和应用自定义评测集合
- 从网页启动 OpenCompass 并查看运行状态与日志路径
- 浏览分数、预测样本和运行文件
- 调用已配置的本地模型生成中文 Markdown 评测报告
