# -*- coding: utf-8 -*-
"""
agent/postproc.py
=================
答案标准化（v3）：
  1) 优先解析 JSON {"answer":"..."}（CoT 输出）
  2) 解析失败回退到正则字母抽取
  3) 按题型约束输出（mcq/tf 取首字母；multi 去重排序）

评测规则：
  - 单选 / 判断：首个有效字母
  - 多选：去重 + 排序后整体匹配，漏选/错选/多选均错
"""
from __future__ import annotations

import json
import re
from typing import List, Optional

_RE_LETTER = re.compile(r"[A-D]")
_RE_JSON_OBJ = re.compile(r"\{[^{}]*\"answer\"[^{}]*\}", re.DOTALL)
_RE_ANSWER_FIELD = re.compile(r"\"answer\"\s*:\s*\"([^\"]*)\"")
_TF_LETTERS = set("AB")


def _try_parse_json(raw: str) -> Optional[str]:
    """从模型 raw 输出里抽 answer 字段。多重保险。"""
    if not raw:
        return None
    raw = raw.strip()

    # 去 markdown code fence
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    # 1) 直接整段 json.loads
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "answer" in obj:
            return str(obj["answer"]).strip()
    except Exception:
        pass

    # 2) 找第一个 {...answer...} 子串
    m = _RE_JSON_OBJ.search(raw)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "answer" in obj:
                return str(obj["answer"]).strip()
        except Exception:
            pass

    # 3) 正则直接抓 "answer": "XX"
    m = _RE_ANSWER_FIELD.search(raw)
    if m:
        return m.group(1).strip()

    return None


def _extract_letters(s: str) -> List[str]:
    return _RE_LETTER.findall((s or "").upper())


def normalize_mcq(raw: str) -> str:
    js = _try_parse_json(raw)
    src = js if js is not None else raw
    letters = _extract_letters(src)
    return letters[0] if letters else "A"


def normalize_tf(raw: str) -> str:
    js = _try_parse_json(raw)
    src = js if js is not None else raw
    for ch in _extract_letters(src):
        if ch in _TF_LETTERS:
            return ch
    return "A"


def normalize_multi(raw: str) -> str:
    js = _try_parse_json(raw)
    src = js if js is not None else raw
    letters = _extract_letters(src)
    if not letters:
        return "A"
    return "".join(sorted(set(letters)))


def normalize(raw: str, fmt: str) -> str:
    fmt = (fmt or "mcq").lower()
    if fmt == "mcq":
        return normalize_mcq(raw)
    if fmt == "tf":
        return normalize_tf(raw)
    if fmt == "multi":
        return normalize_multi(raw)
    return normalize_mcq(raw)
