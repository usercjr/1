# -*- coding: utf-8 -*-
"""
04_run_eval.py
==============
批量答题 + 生成 answer.csv（赛题提交格式）。

特性：
  - **检查点续跑**：每题答完立即落盘到 outputs/run.jsonl，意外中断后从下一题继续
  - **per-question token 记账**：精确到每题，便于事后分析
  - **错误容忍**：单题失败不阻塞批次，错误记入日志，最终答案给 "A"（合法兜底）
  - **进度可视化**：tqdm 进度条 + 实时 token 累计
  - **多 mode**：oracle (A 榜)、domain (B 榜模拟)、global (B 榜真盘)
  - **限速/限量**：--limit、--sleep_ms

使用：
    # A 榜 oracle 全量
    python scripts/04_run_eval.py \
        --questions_dir public_dataset_upload/questions/group_a \
        --out_dir outputs \
        --mode oracle

    # 试跑 5 题（约 5000 token，1 分钱）
    python scripts/04_run_eval.py --limit 5

    # 中断后续跑（自动跳过已完成的 qid）
    python scripts/04_run_eval.py --resume

输出：
    outputs/
      answer.csv          # 提交文件
      run.jsonl           # 每题详细记录（含 evidence、raw、token）
      token_stats.json    # 汇总
      errors.log          # 失败题目
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm

from agent import config
from agent.llm import QwenLLM, get_token_stats
from agent.retriever import Retriever
from agent.solver import Solver

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eval")


# ===========================================================================
# 工具
# ===========================================================================
def load_questions(qdir: Path) -> List[Dict[str, Any]]:
    qs: List[Dict[str, Any]] = []
    for jf in sorted(qdir.glob("*.json")):
        try:
            qs.extend(json.loads(jf.read_text(encoding="utf-8")))
        except Exception as e:
            log.warning(f"读取失败 {jf.name}: {e}")
    return qs


def load_checkpoint(jsonl: Path) -> Dict[str, Dict[str, Any]]:
    """读取已完成的题目记录 {qid: record}"""
    done: Dict[str, Dict[str, Any]] = {}
    if not jsonl.exists():
        return done
    with jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("qid"):
                    done[rec["qid"]] = rec
            except Exception:
                continue
    return done


def snapshot_stats(stats) -> Dict[str, int]:
    d = stats.to_dict()
    return {"p": d["prompt_tokens"], "c": d["completion_tokens"], "t": d["total_tokens"]}


def write_answer_csv(records: List[Dict[str, Any]], out_path: Path) -> None:
    """按赛题格式写 answer.csv：summary 行 + 每题一行"""
    total_p = sum(r["prompt_tokens"] for r in records)
    total_c = sum(r["completion_tokens"] for r in records)
    total_t = sum(r["total_tokens"] for r in records)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # newline="" 避免 Windows 下出现空行
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["qid", "answer", "prompt_tokens", "completion_tokens", "total_tokens"])
        w.writerow(["summary", "", total_p, total_c, total_t])
        for r in records:
            w.writerow([r["qid"], r["answer"],
                        r["prompt_tokens"], r["completion_tokens"], r["total_tokens"]])
    log.info(f"answer.csv -> {out_path} (题数 {len(records)}, total_tokens {total_t:,})")


def write_evidence_json(records: List[Dict[str, Any]], out_path: Path) -> None:
    """证据可追溯性输出（spec 7.4 节）。"""
    out: List[Dict[str, Any]] = []
    for r in records:
        out.append({
            "qid": r["qid"],
            "answer": r["answer"],
            "evidence_retrieval": [
                {
                    "doc_id": e.get("doc_id"),
                    "section": e.get("section", ""),
                    "page": e.get("page", 0),
                    "score": round(float(e.get("score", 0.0)), 3),
                }
                for e in r.get("evidence", [])
            ],
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"evidence.json -> {out_path}")


# ===========================================================================
# 主流程
# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions_dir", default="public_dataset_upload/questions/group_a")
    ap.add_argument("--index_dir", default="data/index")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--mode", default="oracle", choices=["oracle", "domain", "global"])
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（调试用）")
    ap.add_argument("--start_qid", default=None, help="从指定 qid 开始")
    ap.add_argument("--only_qids", default="", help="逗号分隔，只跑这些题")
    ap.add_argument("--resume", action="store_true", help="跳过已完成的 qid")
    ap.add_argument("--sleep_ms", type=int, default=0, help="每题之间睡眠毫秒，防限流")
    ap.add_argument("--top_k", type=int, default=config.TOP_K_CHUNKS)
    ap.add_argument("--model", default=None, help="覆盖默认模型")
    ap.add_argument("--dry_run", action="store_true", help="只跑检索不调 LLM")
    args = ap.parse_args()

    qdir = Path(args.questions_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_dir / "run.jsonl"
    csv_path = out_dir / "answer.csv"
    ev_path = out_dir / "evidence.json"
    err_log = out_dir / "errors.log"
    stats_path = out_dir / "token_stats.json"

    # 1) 加载题目
    questions = load_questions(qdir)
    if not questions:
        log.error(f"未读到题目，检查 {qdir}")
        sys.exit(2)
    log.info(f"题目总数: {len(questions)}")

    # 2) 过滤
    if args.only_qids:
        qset = {x.strip() for x in args.only_qids.split(",") if x.strip()}
        questions = [q for q in questions if q["qid"] in qset]
    if args.start_qid:
        idx = next((i for i, q in enumerate(questions) if q["qid"] == args.start_qid), 0)
        questions = questions[idx:]
    if args.limit > 0:
        questions = questions[: args.limit]
    log.info(f"待执行题目: {len(questions)} 道 (mode={args.mode})")

    # 3) 已完成（resume）
    done: Dict[str, Dict[str, Any]] = {}
    if args.resume:
        done = load_checkpoint(jsonl_path)
        if done:
            log.info(f"resume: 跳过已完成 {len(done)} 题")
    todo = [q for q in questions if q["qid"] not in done]

    # 4) 构造 retriever + solver
    retriever = Retriever(args.index_dir)
    # 预加载所有索引（global mode 需要；oracle mode 也提前 load 避免循环里 IO 抖动）
    retriever.load_all()
    llm = QwenLLM(model=args.model) if not args.dry_run else None
    # 题型路由：QWEN_MODEL_REASONER 非空时，mcq/tf 用思考模型作答
    reasoner = None
    if not args.dry_run and config.QWEN_MODEL_REASONER:
        reasoner = QwenLLM(model=config.QWEN_MODEL_REASONER, enable_thinking=True)
        log.info(f"题型路由: mcq/tf -> {config.QWEN_MODEL_REASONER} (thinking on)")
    solver = (Solver(retriever, llm=llm, top_k=args.top_k, reasoner=reasoner)
              if not args.dry_run else None)

    stats = get_token_stats()

    # 5) 主循环
    err_count = 0
    pbar = tqdm(todo, desc="solve")
    f_jsonl = jsonl_path.open("a", encoding="utf-8")
    try:
        for q in pbar:
            qid = q["qid"]
            before = snapshot_stats(stats)
            t0 = time.time()
            try:
                if args.dry_run:
                    # 仅检索，不答题
                    query = Solver.build_query(q)
                    domain = q.get("domain") if args.mode != "global" else None
                    doc_ids = q.get("doc_ids") if args.mode == "oracle" else None
                    hits = retriever.search(query, domain=domain, doc_ids=doc_ids, top_k=args.top_k)
                    rec = {
                        "qid": qid, "answer": "A",
                        "raw": "<dry_run>",
                        "evidence": hits,
                        "mode": args.mode, "domain": q.get("domain"),
                        "fmt": q.get("answer_format"),
                        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                        "ms": int((time.time() - t0) * 1000),
                    }
                else:
                    res = solver.solve(q, mode=args.mode)
                    after = snapshot_stats(stats)
                    rec = {
                        "qid": qid,
                        "answer": res.answer,
                        "raw": res.raw,
                        "evidence": res.evidence,
                        "mode": res.mode, "domain": res.domain, "fmt": res.fmt,
                        "prompt_tokens": after["p"] - before["p"],
                        "completion_tokens": after["c"] - before["c"],
                        "total_tokens": after["t"] - before["t"],
                        "ms": int((time.time() - t0) * 1000),
                    }
            except Exception as e:
                err_count += 1
                err_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                with err_log.open("a", encoding="utf-8") as ef:
                    ef.write(f"\n=== {qid} ===\n{err_msg}\n")
                # 兜底：避免空答（按题型给合法字母）
                fb = "A"
                if (q.get("answer_format") or "").lower() == "multi":
                    fb = "A"
                rec = {
                    "qid": qid, "answer": fb,
                    "raw": f"<error: {type(e).__name__}>",
                    "evidence": [],
                    "mode": args.mode, "domain": q.get("domain"),
                    "fmt": q.get("answer_format"),
                    "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                    "ms": int((time.time() - t0) * 1000),
                }
                pbar.write(f"[ERR] {qid}: {type(e).__name__}: {e}")

            # 写 jsonl（append + flush，确保断电不丢）
            f_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f_jsonl.flush()
            done[qid] = rec

            tot = stats.to_dict()
            pbar.set_postfix(
                tokens=f"{tot['total_tokens']:,}",
                errs=err_count,
                budget=f"{tot['total_tokens']/config.TOKEN_BUDGET*100:.1f}%",
            )
            if args.sleep_ms > 0:
                time.sleep(args.sleep_ms / 1000.0)
    finally:
        f_jsonl.close()

    # 6) 落盘最终产物
    # 按原题目顺序输出
    qid_order = [q["qid"] for q in questions]
    records = [done[qid] for qid in qid_order if qid in done]

    write_answer_csv(records, csv_path)
    write_evidence_json(records, ev_path)

    final_stats = stats.to_dict()
    final_stats["n_questions_solved"] = len(records)
    final_stats["n_errors"] = err_count
    stats_path.write_text(json.dumps(final_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"token_stats -> {stats_path}")

    # 7) 打印 FinalScore 估算（仅 token 部分，准确率得提交才知道）
    tt = final_stats["total_tokens"]
    token_score = max(0.0, min(1.0, (config.TOKEN_BUDGET - tt) / config.TOKEN_BUDGET))
    print(f"\n=== 估算 ===")
    print(f"  总题数        : {len(records)}")
    print(f"  错误题数      : {err_count}")
    print(f"  total_tokens : {tt:,}  (预算 {config.TOKEN_BUDGET:,}, 用 {tt/config.TOKEN_BUDGET*100:.2f}%)")
    print(f"  TokenScore   : {token_score:.4f}")
    print(f"  FinalScore   : 100 * Accuracy * (0.7 + 0.3 * {token_score:.4f}) = 100 * Acc * {0.7 + 0.3*token_score:.4f}")
    print(f"  → 例：Acc=0.50 → {100*0.50*(0.7+0.3*token_score):.2f}")
    print(f"  → 例：Acc=0.65 → {100*0.65*(0.7+0.3*token_score):.2f}")


if __name__ == "__main__":
    main()
