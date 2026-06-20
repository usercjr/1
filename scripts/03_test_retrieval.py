# -*- coding: utf-8 -*-
"""
03_test_retrieval.py
====================
检索召回率 sanity test（不烧 token）。

用 A 榜公开的 regulatory_questions.json / research_questions.json，
逐题对比 BM25 召回结果与题目自带的 ground-truth doc_ids，统计：

  - Doc Recall@K   = (top-K 检索结果命中 GT 文档数) / (GT 文档数)
  - Doc Recall@K（题级）= 该题 GT 文档**全部**命中的题目比例
  - 两种检索 mode：
      * domain=fixed：限定题目 domain
      * 全域       ：不指定 domain，模拟 B 榜

会输出每题命中详情到 outputs/retrieval_report.json，并打印汇总。

使用：
    python scripts/03_test_retrieval.py \
        --questions_dir public_dataset_upload/questions/group_a \
        --index_dir     data/index \
        --top_k_docs    5
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

# 让脚本能 import agent/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.retriever import Retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("retr-test")


def build_query(q: Dict[str, Any]) -> str:
    """题干 + 所有选项一起作为 query。"""
    opts = q.get("options", {}) or {}
    parts = [q.get("question", "")]
    for k in ("A", "B", "C", "D"):
        v = opts.get(k)
        if v:
            parts.append(v)
    return " ".join(parts)


def eval_question(
    r: Retriever, q: Dict[str, Any], top_k_docs: int, mode: str
) -> Dict[str, Any]:
    gt = set(q.get("doc_ids") or [])
    query = build_query(q)
    domain = q["domain"] if mode == "domain" else None
    hits = r.search_docs(query, domain=domain, top_k_docs=top_k_docs)
    hit_ids = [h["doc_id"] for h in hits]
    matched = gt & set(hit_ids)
    return {
        "qid": q["qid"],
        "domain": q["domain"],
        "gt": sorted(gt),
        "predicted": hit_ids,
        "matched": sorted(matched),
        "all_matched": len(matched) == len(gt) and len(gt) > 0,
        "any_matched": len(matched) > 0,
        "recall": (len(matched) / len(gt)) if gt else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions_dir", default="public_dataset_upload/questions/group_a")
    ap.add_argument("--index_dir", default="data/index")
    ap.add_argument("--out", default="outputs/retrieval_report.json")
    ap.add_argument("--top_k_docs", type=int, default=8)
    args = ap.parse_args()

    qdir = Path(args.questions_dir).resolve()
    questions: List[Dict[str, Any]] = []
    for jf in sorted(qdir.glob("*.json")):
        questions.extend(json.loads(jf.read_text(encoding="utf-8")))
    log.info(f"加载题目 {len(questions)} 道 from {qdir}")

    r = Retriever(args.index_dir)
    r.load_all()

    results = {"domain_mode": [], "global_mode": []}
    for q in questions:
        results["domain_mode"].append(eval_question(r, q, args.top_k_docs, mode="domain"))
        results["global_mode"].append(eval_question(r, q, args.top_k_docs, mode="global"))

    # ---- 汇总 ----
    def summarize(rs: List[Dict[str, Any]]) -> Dict[str, Any]:
        by_dom: Dict[str, Dict[str, float]] = defaultdict(lambda: {
            "n": 0, "all_matched": 0, "any_matched": 0, "recall_sum": 0.0,
        })
        for x in rs:
            d = x["domain"]
            by_dom[d]["n"] += 1
            by_dom[d]["all_matched"] += int(x["all_matched"])
            by_dom[d]["any_matched"] += int(x["any_matched"])
            by_dom[d]["recall_sum"] += x["recall"]
        agg = {}
        total_n = total_all = total_any = 0; total_recall = 0.0
        for d, s in by_dom.items():
            n = s["n"]
            agg[d] = {
                "n": n,
                "all_doc_recall": s["all_matched"] / n if n else 0,
                "any_doc_recall": s["any_matched"] / n if n else 0,
                "avg_recall":     s["recall_sum"] / n if n else 0,
            }
            total_n += n; total_all += s["all_matched"]
            total_any += s["any_matched"]; total_recall += s["recall_sum"]
        agg["__total__"] = {
            "n": total_n,
            "all_doc_recall": total_all / total_n if total_n else 0,
            "any_doc_recall": total_any / total_n if total_n else 0,
            "avg_recall":     total_recall / total_n if total_n else 0,
        }
        return agg

    summary = {
        "top_k_docs": args.top_k_docs,
        "n_questions": len(questions),
        "domain_mode": summarize(results["domain_mode"]),
        "global_mode": summarize(results["global_mode"]),
    }

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"summary": summary, "details": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ---- 打印 ----
    def print_table(title: str, agg: Dict[str, Any]):
        print(f"\n=== {title} ===")
        print(f"{'domain':<22}{'n':>5}{'all_doc%':>12}{'any_doc%':>12}{'avg_rec':>12}")
        for d, s in agg.items():
            print(f"{d:<22}{s['n']:>5}{s['all_doc_recall']*100:>11.1f}%"
                  f"{s['any_doc_recall']*100:>11.1f}%{s['avg_recall']*100:>11.1f}%")

    print(f"\nTop-K docs = {args.top_k_docs}, 题目数 = {len(questions)}")
    print_table("DOMAIN MODE（限定域，A 榜模拟）", summary["domain_mode"])
    print_table("GLOBAL MODE（全域，B 榜模拟）", summary["global_mode"])
    print(f"\n详细报告 -> {out}")
    print("\n指标说明：")
    print("  all_doc% : 该题 ground-truth 文档全部被召回的题占比（最严格）")
    print("  any_doc% : 至少一个 GT 文档被召回的题占比")
    print("  avg_rec  : 平均文档召回率")


if __name__ == "__main__":
    main()
