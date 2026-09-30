"""OpenCompass config for an OpenAI-compatible LLM on the LAN.

Set OPENCOMPASS_MODEL, OPENCOMPASS_BASE_URL and OPENAI_API_KEY before running.
"""

import os as _os

from mmengine.config import read_base

from opencompass.models import OpenAI

with read_base():
    from opencompass.configs.datasets.demo.demo_gsm8k_chat_gen import (
        gsm8k_datasets,
    )


api_meta_template = dict(
    round=[
        dict(role="HUMAN", api_role="HUMAN"),
        dict(role="BOT", api_role="BOT", generate=True),
    ]
)

models = [
    dict(
        type=OpenAI,
        abbr=_os.getenv("OPENCOMPASS_MODEL", "lan-llm"),
        path=_os.getenv("OPENCOMPASS_MODEL", "lan-llm"),
        key=_os.getenv("OPENAI_API_KEY", "EMPTY"),
        openai_api_base=_os.getenv(
            "OPENCOMPASS_BASE_URL",
            "http://127.0.0.1:8000/v1/chat/completions",
        ),
        meta_template=api_meta_template,
        query_per_second=int(_os.getenv("OPENCOMPASS_QPS", "1")),
        max_out_len=1024,
        max_seq_len=8192,
        batch_size=1,
    )
]

datasets = gsm8k_datasets
