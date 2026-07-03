# -*- coding: utf-8 -*-
"""
agent/retriever.py
==================
BM25 检索器。统一的 query 接口：

    r = Retriever("data/index")
    hits = r.search(
        query="第四十七条 资产负债率 担保",
        domain="regulatory",            # 限制域；None=全域
        doc_ids=["strict_v3_008", ...], # A 榜 oracle：只在指定 doc 内检索
        top_k=5,
    )
    # hits: List[Dict] —— 含 chunk_id / doc_id / page / section / text / score

特性：
  - 每个域独立 BM25 索引（懒加载）
  - 支持 doc_ids 过滤（A 榜）和 doc_id 前缀加权（regulatory 优先 strict_v3）
  - 支持 chunk 级 dedup（同 doc 同 section 取最高分）
  - 0 LLM 调用，0 token 消耗
"""
from __future__ import annotations

import logging
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import jieba

log = logging.getLogger(__name__)

# 与 02_build_index.py 保持一致
_NON_TOKEN = re.compile(r"[^一-龥A-Za-z0-9%·年月日条款章节项]+")

# 条款号正则（中文/阿拉伯数字）
_RE_CLAUSE = re.compile(r"第[一二三四五六七八九十百千零0-9]+条")
_CLAUSE_BOOST = 2.5     # query 含条款号时，含此条款号的 chunk 分数 ×

# 从 build_index 同步过来的金融词（如果 jieba 已被 init 过，加重复也无害）
_FINANCE_TERMS = [
    "保险责任", "身故保险金", "现金价值", "退保金额", "账户价值", "保单贷款", "犹豫期",
    "受益人", "宽限期", "复效", "全残", "等待期", "豁免保险费", "重大疾病",
    "受益所有人", "尽职调查", "可疑交易报告", "特别决议", "普通决议", "上市公司治理准则",
    "章程指引", "独立董事", "反洗钱", "客户身份资料", "支付机构",
    "营业收入", "归属于上市公司股东的净利润", "经营活动产生的现金流量净额",
    "研发投入", "扣非净利润", "资产负债率", "毛利率", "净利率", "每股收益",
    "募集说明书", "发行人", "主承销商", "受托管理人", "募集资金", "信用评级",
    "复合增速", "市场规模", "市占率",
]


def _ensure_jieba_dict():
    for w in _FINANCE_TERMS:
        jieba.add_word(w, freq=2_000)
    jieba.setLogLevel(logging.WARNING)


def tokenize(text: str) -> List[str]:
    text = _NON_TOKEN.sub(" ", text)
    return [t for t in jieba.lcut(text) if t.strip()]


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    domain: str
    title: str
    page: int
    section: str
    text: str
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id, "doc_id": self.doc_id, "domain": self.domain,
            "title": self.title, "page": self.page, "section": self.section,
            "text": self.text, "score": float(self.score),
        }


class Retriever:
    """多域 BM25 检索器，懒加载。"""

    DOMAINS = [
        "insurance", "regulatory", "financial_contracts",
        "financial_reports", "research",
    ]

    def __init__(self, index_dir: str | Path = "data/index"):
        self.index_dir = Path(index_dir)
        self._idx: Dict[str, Dict[str, Any]] = {}
        _ensure_jieba_dict()

    # ---- 索引加载 ----
    def _load(self, domain: str) -> Dict[str, Any]:
        if domain not in self._idx:
            pkl = self.index_dir / f"{domain}.pkl"
            if not pkl.exists():
                raise FileNotFoundError(f"索引不存在: {pkl}")
            with pkl.open("rb") as f:
                self._idx[domain] = pickle.load(f)
            log.info(f"load index: {domain} ({len(self._idx[domain]['chunks'])} chunks)")
        return self._idx[domain]

    def load_all(self):
        for d in self.DOMAINS:
            try:
                self._load(d)
            except FileNotFoundError:
                log.warning(f"跳过域 {d}（无索引）")

    # ---- 文档身份头（封面/首页）----
    def doc_head(self, domain: Optional[str], doc_id: str, n: int = 1) -> List[Dict[str, Any]]:
        """返回某文档最前面的 n 个 chunk（通常是封面/首页，含发行人/证券简称/
        文件类型等身份信息）。用于跨文档比对题修复"张冠李戴"。"""
        if not domain:
            return []
        try:
            idx = self._load(domain)
        except FileNotFoundError:
            return []
        out: List[Dict[str, Any]] = []
        for c in idx["chunks"]:
            if c.get("doc_id") == doc_id:
                out.append(c)
                if len(out) >= n:
                    break
        return out

    # ---- 单域检索 ----
    def _search_one(
        self,
        domain: str,
        query_tokens: Sequence[str],
        doc_ids: Optional[Iterable[str]],
        top_k: int,
        oversample: int = 10,
        clause_hits: Optional[set] = None,
    ) -> List[Hit]:
        idx = self._load(domain)
        chunks: List[Dict[str, Any]] = idx["chunks"]
        bm25 = idx["bm25"]

        scores = bm25.get_scores(list(query_tokens))

        # 条款号 boost：query 里命中"第 X 条"时，含相同 section 的 chunk 分数 × _CLAUSE_BOOST
        if clause_hits:
            import numpy as _np
            scores = scores.copy() if hasattr(scores, "copy") else list(scores)
            for i, c in enumerate(chunks):
                sec = c.get("section") or ""
                if sec and any(ch in sec for ch in clause_hits):
                    scores[i] *= _CLAUSE_BOOST

        # doc_ids 过滤（A 榜 oracle）
        doc_filter = None
        if doc_ids:
            doc_filter = list(dict.fromkeys(doc_ids))  # 去重保序

        # 取 top (oversample * top_k) 候选，再做 dedup
        K = min(len(scores), max(top_k * oversample, 100))
        import numpy as np
        order = np.argpartition(-scores, kth=min(K - 1, len(scores) - 1))[:K]
        order = order[np.argsort(-scores[order])]

        # ---- 当 oracle 给了多个 doc_ids 时，强制 per-doc 公平采样 ----
        # 每个指定 doc 至少分到 max(1, top_k // N) 个 slot
        per_doc_cap = None
        if doc_filter and len(doc_filter) > 1:
            per_doc_cap = max(1, top_k // len(doc_filter))

        hits: List[Hit] = []
        seen_section = set()  # (doc_id, section, page)
        doc_count: Dict[str, int] = {}
        deferred: List[Hit] = []  # 超出 per_doc_cap 的备选，第二轮用

        for i in order:
            c = chunks[i]
            if doc_filter is not None and c["doc_id"] not in set(doc_filter):
                continue
            s = float(scores[i]) * float(c.get("priority", 1.0))
            if s <= 0:
                continue
            key = (c["doc_id"], c.get("section", ""), c.get("page", 0))
            if key in seen_section:
                continue
            seen_section.add(key)

            hit = Hit(
                chunk_id=c["chunk_id"], doc_id=c["doc_id"], domain=domain,
                title=c.get("title", ""), page=c.get("page", 0),
                section=c.get("section", ""), text=c["text"], score=s,
            )

            # per-doc 配额限制
            if per_doc_cap is not None and doc_count.get(c["doc_id"], 0) >= per_doc_cap:
                deferred.append(hit)
                continue

            hits.append(hit)
            doc_count[c["doc_id"]] = doc_count.get(c["doc_id"], 0) + 1
            if len(hits) >= top_k:
                break

        # 配额没用完 → 用 deferred 补齐（保留 score 排序）
        if len(hits) < top_k:
            for h in deferred:
                hits.append(h)
                if len(hits) >= top_k:
                    break

        # 输出按 score 降序
        hits.sort(key=lambda h: -h.score)
        return hits[:top_k]

    # ---- 公开接口 ----
    def search(
        self,
        query: str,
        domain: Optional[str] = None,
        doc_ids: Optional[Iterable[str]] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        toks = tokenize(query)
        if not toks:
            return []
        # 抽 query 里的条款号，传给单域检索做 boost
        clause_hits = set(_RE_CLAUSE.findall(query))
        if domain:
            hits = self._search_one(domain, toks, doc_ids, top_k, clause_hits=clause_hits)
        else:
            all_hits: List[Hit] = []
            for d in self.DOMAINS:
                try:
                    all_hits.extend(self._search_one(
                        d, toks, doc_ids, top_k, clause_hits=clause_hits))
                except FileNotFoundError:
                    continue
            all_hits.sort(key=lambda h: -h.score)
            hits = all_hits[:top_k]
        return [h.to_dict() for h in hits]

    # ---- 文档级召回（B 榜先定位 doc 再细查）----
    def search_docs(
        self,
        query: str,
        domain: Optional[str] = None,
        top_k_docs: int = 5,
        chunks_per_doc_for_scoring: int = 3,
    ) -> List[Dict[str, Any]]:
        """返回 [{doc_id, domain, score, title}, ...]
        每个 doc 的得分 = 该 doc 内 top-N chunk 得分之和（× priority）。"""
        toks = tokenize(query)
        if not toks:
            return []
        doms = [domain] if domain else self.DOMAINS
        agg: Dict[str, Dict[str, Any]] = {}
        for d in doms:
            try:
                idx = self._load(d)
            except FileNotFoundError:
                continue
            scores = idx["bm25"].get_scores(toks)
            chunks = idx["chunks"]
            doc_buckets: Dict[str, List[float]] = {}
            doc_meta: Dict[str, Dict[str, Any]] = {}
            for i, s in enumerate(scores):
                if s <= 0:
                    continue
                c = chunks[i]
                doc_buckets.setdefault(c["doc_id"], []).append(
                    float(s) * float(c.get("priority", 1.0))
                )
                if c["doc_id"] not in doc_meta:
                    doc_meta[c["doc_id"]] = {
                        "doc_id": c["doc_id"], "domain": d, "title": c.get("title", ""),
                    }
            for did, ss in doc_buckets.items():
                ss.sort(reverse=True)
                agg_score = sum(ss[:chunks_per_doc_for_scoring])
                key = (d, did)
                if key not in agg or agg[key]["score"] < agg_score:
                    agg[key] = {**doc_meta[did], "score": agg_score}
        out = sorted(agg.values(), key=lambda x: -x["score"])[:top_k_docs]
        return out


# ---- 简单 self-test ----
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", default="data/index")
    ap.add_argument("--query", required=True)
    ap.add_argument("--domain", default=None)
    ap.add_argument("--doc_ids", default="", help="逗号分隔")
    ap.add_argument("--top_k", type=int, default=5)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    r = Retriever(args.index_dir)
    dids = [x for x in args.doc_ids.split(",") if x] or None
    hits = r.search(args.query, domain=args.domain, doc_ids=dids, top_k=args.top_k)
    for i, h in enumerate(hits, 1):
        head = f"[{i}] {h['domain']}/{h['doc_id']} p{h['page']} {h['section'][:30]} score={h['score']:.2f}"
        print(head)
        print("    " + h["text"][:180].replace("\n", " ") + ("..." if len(h["text"]) > 180 else ""))
