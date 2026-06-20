# -*- coding: utf-8 -*-
"""
agent/postproc.py
=================
答案标准化：从模型 raw 输出中抽取合法字母。

评测规则（来自赛题）：
- 单选/判断：首个有效字母即可
- 多选：字母去重 + 排序后整体匹配
- 漏选、错选、多选均计为错误（无部分分）
"""
from __future__ import annotations

import re
from typing import List

_RE_LETTER = re.compile(r"[A-D]")
_TF_LETTERS = set("AB")


def _extract_letters(raw: str) -> List[str]:
    return _RE_LETTER.findall((raw or "").upper())


def normalize_mcq(raw: str) -> str:
    letters = _extract_letters(raw)
    if not letters:
        return "A"  # 兜底，避免空答
    return letters[0]


def normalize_tf(raw: str) -> str:
    letters = _extract_letters(raw)
    for ch in letters:
        if ch in _TF_LETTERS:
            return ch
    return "A"


def normalize_multi(raw: str) -> str:
    letters = _extract_letters(raw)
    if not letters:
        return "A"
    # 去重 + 排序
    uniq = sorted(set(letters))
    return "".join(uniq)


def normalize(raw: str, fmt: str) -> str:
    fmt = (fmt or "mcq").lower()
    if fmt == "mcq":
        return normalize_mcq(raw)
    if fmt == "tf":
        return normalize_tf(raw)
    if fmt == "multi":
        return normalize_multi(raw)
    return normalize_mcq(raw)
