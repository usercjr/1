# -*- coding: utf-8 -*-
"""
agent/prompts.py
================
Prompt 模板（v3）：
  - 5 个域专用 system prompt（外部文件 prompts/{domain}.txt）
  - JSON 结构化输出（逐选项分析 + 最终 answer 字段）
  - 三种题型 (mcq/multi/tf) 共用一个 JSON schema

输出 schema：
{
  "analysis": {"A": "...", "B": "...", "C": "...", "D": "..."},  # tf 只有 A/B
  "answer":   "AC"     # 单选/判断单字母；多选按字母序，无分隔符
}
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
_FALLBACK_SYSTEM = (
    "你是金融文档分析专家。严格依据证据回答，证据未提及视为未知，"
    "禁止用常识替代证据。"
)


@lru_cache(maxsize=8)
def _load_domain_system(domain: str) -> str:
    fp = _PROMPTS_DIR / f"{domain}.txt"
    if fp.exists():
        return fp.read_text(encoding="utf-8").strip()
    return _FALLBACK_SYSTEM


def _truncate(text: str, max_chars: int) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "……" + text[-half:]


def render_evidence(
    hits: List[Dict[str, Any]],
    max_chars: int = 700,
    doc_ord: Dict[str, int] | None = None,
) -> str:
    """证据片段排版。doc_ord: doc_id → 题目 doc_ids 中的序号（1 起），
    用于把每条证据标注为"第 N 份文档"，防止跨文档比对题张冠李戴。"""
    lines: List[str] = []
    for i, h in enumerate(hits, 1):
        sec = h.get("section") or ""
        sec_part = f" {sec}" if sec else ""
        page_part = f" p{h.get('page')}" if h.get("page") else ""
        ord_part = ""
        if doc_ord:
            n = doc_ord.get(h.get("doc_id", ""))
            if n:
                ord_part = f"【第{n}份文档】"
        head = f"[证据{i}] {ord_part}{h.get('doc_id', '?')}{sec_part}{page_part}"
        body = _truncate(h.get("text", ""), max_chars)
        lines.append(head + "\n" + body)
    return "\n\n".join(lines)


def _options_block(options: Dict[str, str], keys: List[str]) -> str:
    parts = []
    for k in keys:
        v = options.get(k)
        if v is not None:
            parts.append(f"{k}. {v}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# JSON CoT 模板
# ---------------------------------------------------------------------------
_JSON_INSTRUCTION_MULTI = """
请按下列 JSON 格式输出，不要任何额外文字、不要 markdown 代码块、不要 ```:
{{"analysis":{{"A":"<一句话推理依据，引用证据编号>","B":"...","C":"...","D":"..."}},"answer":"<按字母升序排列的所有正确选项，如 AC>"}}

判断规则：
- 独立判断每个选项是否被证据明确支持；证据未提及的选项视为错误，不要纳入答案。
- 漏选与错选同样计为全错：证据（含任何一条证据片段）明确支持的选项必须全部选入，一个都不能漏；证据不支持的一个都不能选。
- 数值/日期类选项：先在证据中找到对应数字再比对，数字相符即支持。
- answer 至少包含一个字母；多选题最多 4 个字母（ABCD）。
- 输出必须是合法 JSON，键值都用双引号。
"""

_JSON_INSTRUCTION_SINGLE = """
请按下列 JSON 格式输出，不要任何额外文字、不要 markdown 代码块、不要 ```:
{{"analysis":{{"A":"<一句话依据>","B":"...","C":"...","D":"..."}},"answer":"<唯一正确选项字母 A/B/C/D>"}}

判断规则：
- 对比四个选项与证据的一致性，选出唯一正确者。
- 涉及计算时，在 analysis 里列出算式与中间结果（先独立计算，再对照选项，不要用选项数字反推）。
- answer 必须是一个字母。
- 输出必须是合法 JSON，键值都用双引号。
"""

_JSON_INSTRUCTION_TF = """
请按下列 JSON 格式输出，不要任何额外文字、不要 markdown 代码块、不要 ```:
{{"analysis":{{"A":"<陈述与证据一致性分析>","B":"<反方依据>"}},"answer":"<A 或 B>"}}

判断规则：
- 选项 A 通常为"正确/陈述成立"，B 通常为"错误/陈述不成立"，以题面为准。
- answer 是一个字母（A 或 B）。
- 输出必须是合法 JSON，键值都用双引号。
"""


def _wrap_with_system(system: str, body: str) -> str:
    """system 与正文拼接为单条 user prompt（DashScope chat-template 用 user 角色更稳定）。"""
    return f"{system}\n\n{body}"


def build_mcq(question: str, options: Dict[str, str], evidence: str, system: str) -> str:
    body = f"""【证据】
{evidence}

【题目】{question}

【选项】
{_options_block(options, ['A', 'B', 'C', 'D'])}
{_JSON_INSTRUCTION_SINGLE}"""
    return _wrap_with_system(system, body)


def build_multi(question: str, options: Dict[str, str], evidence: str, system: str) -> str:
    body = f"""【证据】
{evidence}

【题目（多选）】{question}

【选项】
{_options_block(options, ['A', 'B', 'C', 'D'])}
{_JSON_INSTRUCTION_MULTI}"""
    return _wrap_with_system(system, body)


def build_tf(question: str, options: Dict[str, str], evidence: str, system: str) -> str:
    body = f"""【证据】
{evidence}

【陈述】{question}

【选项】
{_options_block(options, ['A', 'B'])}
{_JSON_INSTRUCTION_TF}"""
    return _wrap_with_system(system, body)


def build_prompt(
    question_obj: Dict[str, Any],
    hits: List[Dict[str, Any]],
    max_chunk_chars: int = 700,
    doc_roster: str = "",
    doc_ord: Dict[str, int] | None = None,
) -> str:
    """根据 domain + answer_format 分发模板。
    doc_roster: 多文档题的"文档对照表"文本块（第 N 份文档 = doc_id + 身份摘要），
    与 doc_ord 配合，让"第一份/第二份文档"类选项有明确指代。"""
    fmt = (question_obj.get("answer_format") or "mcq").lower()
    domain = (question_obj.get("domain") or "").lower()
    q = question_obj.get("question", "")
    opts = question_obj.get("options", {}) or {}
    evidence = render_evidence(hits, max_chars=max_chunk_chars, doc_ord=doc_ord)
    if doc_roster:
        evidence = doc_roster + "\n\n" + evidence
    system = _load_domain_system(domain)

    if fmt == "mcq":
        return build_mcq(q, opts, evidence, system)
    if fmt == "multi":
        return build_multi(q, opts, evidence, system)
    if fmt == "tf":
        return build_tf(q, opts, evidence, system)
    return build_mcq(q, opts, evidence, system)
