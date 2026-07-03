# -*- coding: utf-8 -*-
"""
agent/llm.py
============
Qwen (DashScope) 调用封装 —— 通过 OpenAI 兼容模式 + OpenAI SDK
端点: https://dashscope.aliyuncs.com/compatible-mode/v1

支持模型：
  - qwen-plus, qwen-max, qwen-turbo（老命名，alias）
  - qwen3.7-plus, qwen3.7-max, qwen3-vl-* 等新命名
  - 任何 dashscope 在兼容端点上线的模型

要点：
  - 单例 token 统计（prompt / completion / total / calls / by_model）
  - 失败指数退避重试（429 / 5xx / 超时）
  - 强制低温度、低 max_tokens，输出尽量收敛
  - 调用日志可选写入 jsonl（按需开启）
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI
from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError

from agent import config

log = logging.getLogger(__name__)


# ===========================================================================
# Token 统计（进程内全局单例）
# ===========================================================================
@dataclass
class TokenStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0
    by_model: Dict[str, Dict[str, int]] = field(default_factory=dict)

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, model: str, usage: Any):
        """usage 可以是 dict 或 OpenAI SDK 的 CompletionUsage 对象。"""
        p = int(getattr(usage, "prompt_tokens", None) or
                (usage.get("prompt_tokens", 0) if isinstance(usage, dict) else 0))
        c = int(getattr(usage, "completion_tokens", None) or
                (usage.get("completion_tokens", 0) if isinstance(usage, dict) else 0))
        t = int(getattr(usage, "total_tokens", None) or
                (usage.get("total_tokens", p + c) if isinstance(usage, dict) else (p + c)))
        with self._lock:
            self.prompt_tokens += p
            self.completion_tokens += c
            self.total_tokens += t
            self.calls += 1
            slot = self.by_model.setdefault(model, {"calls": 0, "prompt": 0, "completion": 0, "total": 0})
            slot["calls"] += 1
            slot["prompt"] += p
            slot["completion"] += c
            slot["total"] += t

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "calls": self.calls,
                "by_model": {k: dict(v) for k, v in self.by_model.items()},
            }

    def reset(self):
        with self._lock:
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.total_tokens = 0
            self.calls = 0
            self.by_model = {}


_STATS = TokenStats()


def get_token_stats() -> TokenStats:
    return _STATS


# ===========================================================================
# Qwen 客户端（OpenAI 兼容模式）
# ===========================================================================
class QwenError(RuntimeError):
    pass


# 全局共享 client（连接池复用）
_CLIENT: Optional[OpenAI] = None
_CLIENT_LOCK = threading.Lock()


def _get_client(timeout: int = 60) -> OpenAI:
    global _CLIENT
    if _CLIENT is None:
        with _CLIENT_LOCK:
            if _CLIENT is None:
                _CLIENT = OpenAI(
                    api_key=config.DASHSCOPE_API_KEY,
                    base_url=config.DASHSCOPE_BASE_URL,
                    timeout=timeout,
                    max_retries=0,   # 自己控制重试
                )
    return _CLIENT


class QwenLLM:
    """轻量同步客户端，messages 协议调 DashScope 兼容端点。"""

    def __init__(
        self,
        model: Optional[str] = None,
        max_retries: int = 4,
        timeout: int = 60,
        log_path: Optional[str | Path] = None,
    ):
        self.model = model or config.QWEN_MODEL
        self.max_retries = max_retries
        self.timeout = timeout
        self.log_path = Path(log_path) if log_path else None
        self.client = _get_client(timeout=timeout)

    # ---- 主调用 ----
    def chat(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: int = config.MAX_COMPLETION_TOKENS,
        temperature: float = config.TEMPERATURE,
        meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        last_err: Optional[str] = None
        for attempt in range(self.max_retries):
            try:
                # 仅对 qwen3 思考系列关 thinking（否则 token 爆炸）；qwen-max 等不传此参数，
                # 保持与 v4 完全一致，避免误伤需要推理的单选题。
                _kwargs = {}
                if "qwen3" in (self.model or "").lower():
                    _kwargs["extra_body"] = {"enable_thinking": False}
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.1,
                    timeout=self.timeout,
                    **_kwargs,
                )
                # 文本提取
                try:
                    text = resp.choices[0].message.content or ""
                except Exception:
                    text = str(resp)

                # 统计 + 日志
                _STATS.add(self.model, resp.usage)
                if self.log_path:
                    self._log_call(prompt, text, resp.usage, meta)
                return text
            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                last_err = f"{type(e).__name__}: {e}"
                if attempt < self.max_retries - 1:
                    self._sleep_backoff(attempt)
                    continue
                raise QwenError(f"调用失败 ({self.max_retries} 次): {last_err}")
            except APIError as e:
                last_err = f"{type(e).__name__}: {e}"
                # 5xx 才重试，4xx 直接抛
                sc = getattr(e, "status_code", None) or 0
                if sc >= 500 and attempt < self.max_retries - 1:
                    self._sleep_backoff(attempt)
                    continue
                raise QwenError(f"DashScope error: {last_err}")
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                if attempt < self.max_retries - 1:
                    self._sleep_backoff(attempt)
                    continue
                raise QwenError(f"调用失败 ({self.max_retries} 次): {last_err}")

        raise QwenError(f"调用失败: {last_err}")

    @staticmethod
    def _sleep_backoff(attempt: int):
        time.sleep(min(2 ** attempt, 16))

    def _log_call(self, prompt: str, output: str, usage: Any, meta: Optional[Dict[str, Any]]):
        rec = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": self.model,
            "meta": meta or {},
            "prompt_chars": len(prompt),
            "output": output,
            "usage": {
                "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
            },
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ===========================================================================
# 简易 self-test
# ===========================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    llm = QwenLLM()
    out = llm.chat("回复一个字：你好。", max_tokens=4)
    print("model:", llm.model)
    print("output:", out)
    print("token stats:", get_token_stats().to_dict())
