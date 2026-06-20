# -*- coding: utf-8 -*-
"""
05_find_suspect.py
==================
从 outputs/run.jsonl 找出"模型可能答错"的题，0 token，纯本地分析。

输出可疑题清单（按可疑度排序），用于人工 review 或 v4 重点调优。

可疑信号：
  1. JSON 解析失败 / raw 不是合法 JSON
  2. multi 题只输出 1 个字母（通常 multi 答案 ≥ 2 个）
  3. multi 题输出 4 个字母 ABCD（极少有全选对）
  4. analysis 异常短（<30 字总长）—— 模型缺信息
  5. analysis 含"无法判定/证据不足/未提及"等不确定词
  6. evidence chunks 数远低于平均
  7. evidence score 极低（<10）—— 检索可能没命中

使用：
    python scripts/05_find_suspect.py
    python scripts/05_find_suspect.py --top 30 --questions_dir public_dataset_upload/questions/group_a
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_UNCERTAIN_PAT = re.compile(
    r"(无法判定|证据不足|未提及|没有提及|无法确定|不清楚|未明确|未说明|"
    r"无相关|不确定|难以判断|无证据|未涉及|未在证据)"
)


def _has_json(raw: str) -> bool:
    if not raw:
        return False
    try:
        json.loads(raw.strip().strip("`").replace("```json", "").replace("```", ""))
        return True
    except Exception:
        return False


def _analysis_text(raw: str) -> str:
    """从 JSON raw 里抽 analysis 字段拼成文本，失败则返回 raw"""
    try:
        obj = json.loads(raw.strip().strip("`").replace("```json", "").replace("```", ""))
        if isinstance(obj, dict) and "analysis" in obj:
            a = obj["analysis"]
            if isinstance(a, dict):
                return " ".join(str(v) for v in a.values())
            return str(a)
    except Exception:
        pass
    return raw or ""


def score_question(rec: Dict[str, Any], avg_chunks: float, avg_top_score: float) -> Dict[str, Any]:
    """计算可疑度，越高越可疑。"""
    raw = rec.get("raw", "") or ""
    ans = rec.get("answer", "") or ""
    fmt = (rec.get("fmt") or "").lower()
    evid = rec.get("evidence") or []

    flags: List[str] = []
    suspect = 0.0

    # 1) JSON 解析失败
    if not _has_json(raw):
        flags.append("非JSON")
        suspect += 3.0

    # 2) multi 题异常答案长度
    if fmt == "multi":
        if len(ans) == 1:
            flags.append("multi只1字母")
            suspect += 2.0
        elif len(ans) >= 4:
            flags.append("multi全选ABCD")
            suspect += 1.5

    # 3) analysis 过短
    analysis = _analysis_text(raw)
    if len(analysis) < 30:
        flags.append(f"analysis短({len(analysis)})")
        suspect += 2.0

    # 4) 含不确定词
    n_uncertain = len(_UNCERTAIN_PAT.findall(analysis))
    if n_uncertain > 0:
        flags.append(f"不确定词×{n_uncertain}")
        suspect += 1.0 * n_uncertain

    # 5) evidence 数偏少
    if len(evid) < max(3, avg_chunks * 0.5):
        flags.append(f"证据少({len(evid)})")
        suspect += 1.0

    # 6) evidence top score 偏低
    if evid:
        top_score = max(e.get("score", 0) for e in evid)
        if top_score < max(10.0, avg_top_score * 0.3):
            flags.append(f"top_score低({top_score:.1f})")
            suspect += 1.5

    return {
        "qid": rec.get("qid"),
        "domain": rec.get("domain"),
        "fmt": fmt,
        "answer": ans,
        "suspect": round(suspect, 2),
        "flags": flags,
        "evidence_docs": list({e.get("doc_id") for e in evid}),
        "tokens": rec.get("total_tokens", 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_file", default="outputs/run.jsonl")
    ap.add_argument("--questions_dir", default="public_dataset_upload/questions/group_a")
    ap.add_argument("--top", type=int, default=20, help="输出可疑度 Top N")
    ap.add_argument("--out", default="outputs/suspects.json")
    args = ap.parse_args()

    # 读 run.jsonl
    recs: List[Dict[str, Any]] = []
    for line in open(args.run_file, encoding="utf-8", errors="ignore"):
        line = line.strip().replace("\x00", "")
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except Exception:
            continue
    print(f"已读取 {len(recs)} 条记录")

    # 加载题目（用于打印题干）
    qmap = {}
    qdir = Path(args.questions_dir)
    if qdir.exists():
        for jf in qdir.glob("*.json"):
            for q in json.load(open(jf, encoding="utf-8")):
                qmap[q["qid"]] = q

    # 计算平均指标
    avg_chunks = sum(len(r.get("evidence", [])) for r in recs) / max(len(recs), 1)
    avg_top_score = sum(
        max((e.get("score", 0) for e in r.get("evidence", [])), default=0)
        for r in recs
    ) / max(len(recs), 1)
    print(f"avg evidence chunks={avg_chunks:.1f}, avg top score={avg_top_score:.1f}")

    # 评分
    scored = [score_question(r, avg_chunks, avg_top_score) for r in recs]
    scored.sort(key=lambda x: -x["suspect"])

    # 打印 top N
    print(f"\n=== 可疑题 Top {args.top} ===")
    for i, s in enumerate(scored[: args.top], 1):
        q = qmap.get(s["qid"], {})
        print(f"\n[{i}] {s['qid']}  域={s['domain']}  题型={s['fmt']}  "
              f"我答={s['answer']}  suspect={s['suspect']}  flags={s['flags']}")
        if q:
            print(f"     题干: {q.get('question', '')[:80]}...")
            print(f"     GT doc: {q.get('doc_ids', [])}")
            print(f"     召回 doc: {s['evidence_docs']}")
            miss = set(q.get("doc_ids") or []) - set(s["evidence_docs"])
            if miss:
                print(f"     ⚠️ 缺失 GT 文档: {miss}")

    # 落盘
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"top_suspects": scored[: args.top], "all": scored},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n详细 -> {args.out}")

    # 各域可疑题数
    from collections import Counter
    cnt = Counter((s["domain"], s["suspect"] > 2) for s in scored)
    print("\n=== 各域可疑分布（suspect>2 的题数） ===")
    domains = sorted({s["domain"] for s in scored})
    for d in domains:
        n = sum(1 for s in scored if s["domain"] == d)
        hi = sum(1 for s in scored if s["domain"] == d and s["suspect"] > 2)
        print(f"  {d:<22} 总{n}  高可疑{hi} ({hi/n*100:.0f}%)")


if __name__ == "__main__":
    main()
