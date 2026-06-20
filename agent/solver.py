# -*- coding: utf-8 -*-
"""
agent/solver.py  (v3)
=====================
单题求解：检索 → 构造 prompt → 调 Qwen → JSON 抽取答案。

v3 升级：
  - 选项增强检索：每个选项单独跑一轮 BM25，融入主检索池，权重 0.7
  - 条款号 boost：见 retriever._search_one
  - 域专用 system prompt：见 prompts.py
  - JSON CoT 输出：max_tokens 调到 400+

接口：
    solver = Solver(retriever, llm)
    result = solver.solve(question, mode="oracle")  # 或 "domain"/"global"
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

# 选项增强检索的小权重（防止选项噪声压过主 query）
_OPTION_HIT_WEIGHT = 0.7
# 每个选项额外捞 chunk 数
_OPTION_TOP_K = 2
# 选项文本过短就不单独检索
_OPTION_MIN_CHARS = 8


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

    # ---- query 构造 ----
    @staticmethod
    def build_query(q: Dict[str, Any]) -> str:
        opts = q.get("options") or {}
        parts = [q.get("question", "")]
        for k in ("A", "B", "C", "D"):
            v = opts.get(k)
            if v:
                parts.append(v)
        return " ".join(parts)

    # ---- 选项增强检索 ----
    def _option_augmented_retrieve(
        self,
        question: Dict[str, Any],
        domain: Optional[str],
        doc_ids: Optional[List[str]],
    ) -> List[Dict[str, Any]]:
        """
        主检索 top_k chunks + 每个选项额外 top_2 chunks，按 score 合并去重。
        """
        q_text = question.get("question", "")
        opts = question.get("options") or {}

        # 1) 主检索：题干+所有选项拼接
        main_query = self.build_query(question)
        pool: Dict[str, Dict[str, Any]] = {}
        for h in self.retriever.search(main_query, domain=domain, doc_ids=doc_ids, top_k=self.top_k):
            pool[h["chunk_id"]] = h

        # 2) 选项级补充检索
        for k in ("A", "B", "C", "D"):
            opt_text = opts.get(k) or ""
            if len(opt_text) < _OPTION_MIN_CHARS:
                continue
            opt_query = f"{q_text} {opt_text}"
            for h in self.retriever.search(
                opt_query, domain=domain, doc_ids=doc_ids, top_k=_OPTION_TOP_K
            ):
                cid = h["chunk_id"]
                # 降权后入池：若已存在，取 max
                h = {**h, "score": h.get("score", 0.0) * _OPTION_HIT_WEIGHT}
                if cid not in pool or pool[cid].get("score", 0) < h["score"]:
                    pool[cid] = h

        # 3) 按 score 排序，截断
        hits = sorted(pool.values(), key=lambda x: -x.get("score", 0.0))
        # 上限：top_k + 4（每选项 1 个额外）
        max_total = self.top_k + 4
        return hits[:max_total]

    # ---- 主流程 ----
    def solve(self, question: Dict[str, Any], mode: str = "oracle") -> SolverResult:
        qid = question["qid"]
        domain = question.get("domain", "")
        fmt = (question.get("answer_format") or "mcq").lower()

        # 1) 检索（按 mode 选 doc_ids/domain）
        if mode == "oracle":
            doc_ids = question.get("doc_ids") or None
            hits = self._option_augmented_retrieve(question, domain=domain, doc_ids=doc_ids)
        elif mode == "domain":
            hits = self._option_augmented_retrieve(question, domain=domain, doc_ids=None)
        elif mode == "global":
            hits = self._option_augmented_retrieve(question, domain=None, doc_ids=None)
        else:
            raise ValueError(f"未知 mode: {mode}")

        if not hits:
            log.warning(f"[{qid}] 检索为空")

        # 2) 构造 prompt（含 5 个域 system prompt + JSON CoT 模板）
        prompt = build_prompt(question, hits, max_chunk_chars=self.max_chunk_chars)

        # 3) 调 LLM（JSON CoT 需要更多 completion token）
        # 估算：4 个选项的 analysis 各 ~30 字 + answer ≈ 200 token，给 400 buffer
        max_tok = 200 if fmt == "tf" else 400
        raw = self.llm.chat(
            prompt,
            max_tokens=max_tok,
            meta={"qid": qid, "fmt": fmt, "domain": domain, "mode": mode},
        )

        # 4) 答案抽取（先尝试 JSON，再正则兜底）
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
    ap.add_argument("--question_file", required=True)
    ap.add_argument("--qid", default=None)
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
    print("fmt:", q.get("answer_format"))
    print("=== Evidence (top) ===")
    for i, e in enumerate(res.evidence[:8], 1):
        print(f"  [{i}] {e['doc_id']} {e['section'][:30]} p{e['page']} score={e['score']:.2f}")
    print("=== Raw ===")
    print(res.raw)
    print("=== Answer ===", res.answer)
    from agent.llm import get_token_stats
    print("Token:", get_token_stats().to_dict())
