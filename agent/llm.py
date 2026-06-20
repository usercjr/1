# -*- coding: utf-8 -*-
"""
agent/llm.py
============
Qwen (DashScope) 调用封装。要点：
  - 单例 token 统计（prompt / completion / total / calls）
  - 失败指数退避重试（针对 429 / 5xx / 超时）
  - 强制低温度、低 max_tokens，避免无意义生成
  - 调用日志可选写入文件（按需开启，方便事后审计）

用法：
    from agent.llm import QwenLLM, get_token_stats
    llm = QwenLLM()                # 用 config.QWEN_MODEL 默认 qwen-plus
    ans = llm.chat("...")          # 同步返回字符串
    stats = get_token_stats().to_dict()
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import dashscope
from dashscope import Generation

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
        """usage 可以是 dict 或 dashscope 返回的 usage 对象。"""
        p = int(getattr(usage, "input_tokens", None) or usage.get("input_tokens", 0))
        c = int(getattr(usage, "output_tokens", None) or usage.get("output_tokens", 0))
        t = int(getattr(usage, "total_tokens", None) or usage.get("total_tokens", p + c))
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
# Qwen 客户端
# ===========================================================================
class QwenError(RuntimeError):
    pass


class QwenLLM:
    """轻量同步客户端，按 messages 协议调 DashScope Generation。"""

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

        dashscope.api_key = config.DASHSCOPE_API_KEY
        if config.DASHSCOPE_BASE_URL:
            dashscope.base_http_api_url = config.DASHSCOPE_BASE_URL

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
                resp = Generation.call(
                    model=self.model,
                    messages=messages,
                    result_format="message",
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.1,                # 进一步收敛输出
                    timeout=self.timeout,
                )
                if getattr(resp, "status_code", 200) != 200:
                    last_err = f"{getattr(resp, 'code', '')} {getattr(resp, 'message', '')}"
                    if self._is_retryable(resp):
                        self._sleep_backoff(attempt)
                        continue
                    raise QwenError(f"DashScope error: {last_err}")

                # 提取文本
                try:
                    text = resp.output.choices[0].message.content
                except Exception:
                    text = str(resp.output)
                if isinstance(text, list):
                    # 多模态情况，拼一下
                    text = "".join(seg.get("text", "") for seg in text if isinstance(seg, dict))

                # 统计 + 日志
                _STATS.add(self.model, resp.usage)
                if self.log_path:
                    self._log_call(prompt, text, resp.usage, meta)
                return text or ""
            except QwenError:
                raise
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                if attempt < self.max_retries - 1:
                    self._sleep_backoff(attempt)
                    continue
                raise QwenError(f"调用失败 ({self.max_retries} 次): {last_err}")

        raise QwenError(f"调用失败: {last_err}")

    # ---- 工具 ----
    @staticmethod
    def _is_retryable(resp) -> bool:
        code = str(getattr(resp, "code", "") or "")
        sc = int(getattr(resp, "status_code", 0) or 0)
        if sc in (408, 429, 500, 502, 503, 504):
            return True
        if any(x in code for x in ("Throttling", "Timeout", "ServerInternal", "RateLimit")):
            return True
        return False

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
                "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
                "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
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
    print("model output:", out)
    print("token stats:", get_token_stats().to_dict())
