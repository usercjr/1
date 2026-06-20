# -*- coding: utf-8 -*-
"""
agent/solver.py
===============
单题求解主流程：检索 → 构造 prompt → 调 Qwen → 标准化答案。

公开接口：
    solver = Solver(retriever, llm)
    result = solver.solve(question, mode="oracle")  # 或 "domain" / "global"

返回:
    {
      "qid":      "...",
      "answer":   "ABC",
      "raw":      "<模型原始输出>",
      "evidence": [{doc_id, chunk_id, page, section, score}, ...],
      "mode":     "oracle",
      "domain":   "regulatory",
      "fmt":      "multi",
    }
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent import config
from agent.llm import QwenLLM
from agent.postproc import normalize
from agent.prompts import build_prompt
from agent.retriever import Retriever

log = logging.getLogger(__name__)


@dataclass
class SolverResult:
    qid: str
    answer: str
    raw: str
    evidence: List[Dict[str, Any]]
    mode: str
    domain: str
    fmt: str

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


class Solver:
    """单题求解器。

    mode:
      - "oracle" : 使用题目自带的 doc_ids（A 榜场景）
      - "domain" : 不用 doc_ids，但限定 question.domain
      - "global" : 跨域全召回（B 榜场景，需配合 domain router 才更准）
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: Optional[QwenLLM] = None,
        top_k: int = config.TOP_K_CHUNKS,
        max_chunk_chars: int = config.MAX_CHUNK_CHARS_IN_PROMPT,
    ):
        self.retriever = retriever
        self.llm = llm or QwenLLM()
        self.top_k = top_k
        self.max_chunk_chars = max_chunk_chars

    # ---- 工具：query 构造 ----
    @staticmethod
    def build_query(q: Dict[str, Any]) -> str:
        opts = q.get("options") or {}
        parts = [q.get("question", "")]
        for k in ("A", "B", "C", "D"):
            v = opts.get(k)
            if v:
                parts.append(v)
        return " ".join(parts)

    # ---- 主流程 ----
    def solve(self, question: Dict[str, Any], mode: str = "oracle") -> SolverResult:
        qid = question["qid"]
        domain = question.get("domain", "")
        fmt = (question.get("answer_format") or "mcq").lower()

        # 1) 检索
        query = self.build_query(question)
        if mode == "oracle":
            doc_ids = question.get("doc_ids") or None
            hits = self.retriever.search(query, domain=domain, doc_ids=doc_ids, top_k=self.top_k)
        elif mode == "domain":
            hits = self.retriever.search(query, domain=domain, top_k=self.top_k)
        elif mode == "global":
            hits = self.retriever.search(query, domain=None, top_k=self.top_k)
        else:
            raise ValueError(f"未知 mode: {mode}")

        if not hits:
            log.warning(f"[{qid}] 检索为空，将无证据作答")

        # 2) 构造 prompt
        prompt = build_prompt(question, hits, max_chunk_chars=self.max_chunk_chars)

        # 3) 调用 LLM（按题型选模型 + max_tokens）
        model_for_tf = config.QWEN_MODEL_TF
        if fmt == "tf" and model_for_tf != self.llm.model:
            llm_used = QwenLLM(model=model_for_tf)
        else:
            llm_used = self.llm

        # multi 最多吐 4 字母（ABCD），加点 buffer 给一些模型可能输出的空格
        max_tok = 6 if fmt == "multi" else 4
        raw = llm_used.chat(
            prompt,
            system=None,        # 系统提示已在 prompt 里
            max_tokens=max_tok,
            meta={"qid": qid, "fmt": fmt, "domain": domain, "mode": mode},
        )

        # 4) 标准化
        answer = normalize(raw, fmt)

        return SolverResult(
            qid=qid,
            answer=answer,
            raw=raw,
            evidence=[
                {
                    "doc_id": h["doc_id"], "chunk_id": h["chunk_id"],
                    "page": h.get("page", 0), "section": h.get("section", ""),
                    "score": h.get("score", 0.0),
                }
                for h in hits
            ],
            mode=mode,
            domain=domain,
            fmt=fmt,
        )


# 命令行单题调试
if __name__ == "__main__":
    import argparse, json, sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ap = argparse.ArgumentParser()
    ap.add_argument("--question_file", required=True, help="单文件 json 或 jsonl，第一题即调试")
    ap.add_argument("--qid", default=None, help="指定 qid（默认第一题）")
    ap.add_argument("--mode", default="oracle", choices=["oracle", "domain", "global"])
    ap.add_argument("--index_dir", default="data/index")
    args = ap.parse_args()

    text = open(args.question_file, encoding="utf-8").read()
    try:
        qs = json.loads(text)
        if isinstance(qs, dict):
            qs = [qs]
    except Exception:
        qs = [json.loads(line) for line in text.splitlines() if line.strip()]

    if args.qid:
        qs = [q for q in qs if q.get("qid") == args.qid]
        if not qs:
            print(f"找不到 qid={args.qid}", file=sys.stderr)
            sys.exit(2)

    q = qs[0]
    retriever = Retriever(args.index_dir)
    solver = Solver(retriever)
    res = solver.solve(q, mode=args.mode)
    print("=== Question ===")
    print(q.get("question"))
    print("Options:", q.get("options"))
    print("Expected fmt:", q.get("answer_format"))
    print("=== Evidence ===")
    for i, e in enumerate(res.evidence, 1):
        print(f"  [{i}] {e['doc_id']} {e['section'][:30]} p{e['page']} score={e['score']:.2f}")
    print("=== Raw ===", repr(res.raw))
    print("=== Answer ===", res.answer)
    from agent.llm import get_token_stats
    print("Token:", get_token_stats().to_dict())
