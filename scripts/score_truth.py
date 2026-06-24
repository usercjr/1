# -*- coding: utf-8 -*-
"""用人工真值给 answer.csv 打分（0 token，本地核验 prompt 变体）。
用法: python scripts/score_truth.py outputs_cal/answer.csv [更多.csv ...]
真值: outputs/analysis/ground_truth_manual.json
"""
import csv, json, sys, os

GT_PATH = "outputs/analysis/ground_truth_manual.json"


def load_csv(fn):
    return {r["qid"]: r["answer"] for r in csv.DictReader(open(fn, encoding="utf-8"))
            if r["qid"] != "summary"}


def main():
    gt = json.load(open(GT_PATH, encoding="utf-8"))
    files = sys.argv[1:] or ["outputs/answer_v4_7165.csv"]
    hi = [q for q in gt if gt[q].get("conf") != "中"]   # 高置信子集

    print(f"真值共 {len(gt)} 道（高置信 {len(hi)}）\n")
    for fn in files:
        if not os.path.exists(fn):
            print(f"[缺失] {fn}"); continue
        a = load_csv(fn)
        rows = []
        for q in gt:
            got, truth = a.get(q, "?"), gt[q]["answer"]
            rows.append((q, got, truth, got == truth, gt[q].get("conf") != "中"))
        c_all = sum(1 for r in rows if r[3])
        c_hi = sum(1 for r in rows if r[3] and r[4])
        print(f"=== {fn} ===")
        print(f"  全部: {c_all}/{len(gt)}   高置信: {c_hi}/{len(hi)}")
        wrong = [r for r in rows if not r[3]]
        if wrong:
            print("  错题: " + ", ".join(f"{q}({got}≠{truth})" for q, got, truth, _, _ in wrong))
        print()


if __name__ == "__main__":
    main()
