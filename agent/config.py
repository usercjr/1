# -*- coding: utf-8 -*-
"""
agent/config.py
===============
统一读取配置：环境变量优先，.env 兜底。
"""
from __future__ import annotations

import os
from pathlib import Path

# 1. 加载 .env（如果 python-dotenv 可用且存在 .env）
try:
    from dotenv import load_dotenv
    _ROOT = Path(__file__).resolve().parents[1]
    _ENV = _ROOT / ".env"
    if _ENV.exists():
        load_dotenv(_ENV, override=False)  # 环境变量优先
except ImportError:
    pass


def _get(name: str, default: str | None = None, required: bool = False) -> str | None:
    v = os.getenv(name)
    if v is None or v == "":
        if required:
            raise RuntimeError(
                f"环境变量 {name} 未设置。请 set {name}=... 或在 .env 写入"
            )
        return default
    return v


# ---- 公开配置 ----
DASHSCOPE_API_KEY: str = _get("DASHSCOPE_API_KEY", required=True)  # type: ignore[assignment]
# OpenAI 兼容模式端点（支持 qwen3.7-plus 等新模型）
DASHSCOPE_BASE_URL: str = _get(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
) or "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL: str = _get("QWEN_MODEL", "qwen-plus") or "qwen-plus"
QWEN_MODEL_TF: str = _get("QWEN_MODEL_TF", QWEN_MODEL) or QWEN_MODEL  # 判断题可分流到更便宜的模型

# 检索 / Prompt 超参（这里集中放，方便实验）
TOP_K_CHUNKS: int = int(_get("TOP_K_CHUNKS", "5") or 5)
MAX_CHUNK_CHARS_IN_PROMPT: int = int(_get("MAX_CHUNK_CHARS", "1000") or 1000)  # 学队友：证据片段给更足，少截断
MAX_COMPLETION_TOKENS: int = int(_get("MAX_COMPLETION_TOKENS", "800") or 800)  # 学队友：逐项分析留更多空间，缓解多选漏选
# 多文档题证据的总字符预算（按文档均分，保证每篇都有料），学自队友 per-doc budget
EVIDENCE_TOTAL_CHAR_BUDGET: int = int(_get("EVIDENCE_TOTAL_CHAR_BUDGET", "9000") or 9000)
TEMPERATURE: float = float(_get("TEMPERATURE", "0.0") or 0.0)

# Token 预算（用于警告）
TOKEN_BUDGET: int = int(_get("TOKEN_BUDGET", "5000000") or 5000000)
