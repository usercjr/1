# -*- coding: utf-8 -*-
"""
agent/prompts.py
================
按题型生成 Prompt。原则：
  1) 强约束输出，只让模型吐字母，节省 completion token。
  2) 证据片段排版要紧凑，避免无意义空白吃 token。
  3) 多选题要单独强调"严格依据证据，没明说就不选"——这是丢分大头。
"""
from __future__ import annotations

from typing import Any, Dict, List

SYS_PROMPT = (
    "你是金融文档分析专家。请基于给定的证据片段判断问题。"
    "严格依据证据，证据未提及的内容视为未知，不要凭常识或外部知识补充。"
)


def _truncate(text: str, max_chars: int) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    # 在句号附近截断
    half = max_chars // 2
    head = text[:half]
    tail = text[-half:]
    return head + "……" + tail


def render_evidence(hits: List[Dict[str, Any]], max_chars: int = 700) -> str:
    """把检索结果排版为证据块。"""
    lines: List[str] = []
    for i, h in enumerate(hits, 1):
        sec = h.get("section") or ""
        sec_part = f" {sec}" if sec else ""
        page_part = f" p{h.get('page')}" if h.get("page") else ""
        head = f"[证据{i}] {h.get('doc_id', '?')}{sec_part}{page_part}"
        body = _truncate(h.get("text", ""), max_chars)
        lines.append(head + "\n" + body)
    return "\n\n".join(lines)


def _options_block(options: Dict[str, str]) -> str:
    parts = []
    for k in ("A", "B", "C", "D"):
        v = options.get(k)
        if v is None:
            continue
        parts.append(f"{k}. {v}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 三种题型模板
# ---------------------------------------------------------------------------
def build_mcq(question: str, options: Dict[str, str], evidence: str) -> str:
    return f"""【证据片段】
{evidence}

【问题】{question}

【选项】
{_options_block(options)}

任务：从 A/B/C/D 中选出唯一正确答案。
输出规则：只输出一个大写字母（A、B、C 或 D），不输出任何其它字符、标点或解释。
答案："""


def build_multi(question: str, options: Dict[str, str], evidence: str) -> str:
    return f"""【证据片段】
{evidence}

【问题】{question}

【选项】
{_options_block(options)}

任务：选出所有"基于证据可以直接判定为正确"的选项。证据未明确支持的不要选。
输出规则：
- 按字母顺序输出，例如 AC、ABD、ABCD。
- 不要空格、不要逗号、不要解释。
- 若四个都对，输出 ABCD。
- 至少输出一个字母（不要空答）。
答案："""


def build_tf(question: str, options: Dict[str, str], evidence: str) -> str:
    return f"""【证据片段】
{evidence}

【问题】{question}

【选项】
{_options_block(options)}

任务：根据证据判断陈述的对错，从 A/B 中二选一。
输出规则：只输出一个大写字母（A 或 B），不输出任何其它字符或解释。
答案："""


def build_prompt(question_obj: Dict[str, Any], hits: List[Dict[str, Any]],
                 max_chunk_chars: int = 700) -> str:
    """根据 answer_format 分发到对应模板。"""
    fmt = (question_obj.get("answer_format") or "mcq").lower()
    q = question_obj.get("question", "")
    opts = question_obj.get("options", {}) or {}
    evidence = render_evidence(hits, max_chars=max_chunk_chars)
    if fmt == "mcq":
        return build_mcq(q, opts, evidence)
    if fmt == "multi":
        return build_multi(q, opts, evidence)
    if fmt == "tf":
        return build_tf(q, opts, evidence)
    # 未知题型按 mcq 兜底
    return build_mcq(q, opts, evidence)
