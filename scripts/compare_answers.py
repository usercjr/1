# -*- coding: utf-8 -*-
"""对比两个 answer.csv 的答案差异"""
import csv
import sys

def load(path):
    d = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["qid"] != "summary":
                d[r["qid"]] = r["answer"]
    return d

f1 = sys.argv[1] if len(sys.argv) > 1 else "outputs/answer_v4_7165.csv"
f2 = sys.argv[2] if len(sys.argv) > 2 else "outputs_v5/answer.csv"

v1 = load(f1)
v2 = load(f2)

same = 0
diff = 0
print(f"{'qid':15s} {'v4':8s} {'v5':8s} {'变化'}")
print("-" * 50)
for q in sorted(v1):
    a1 = v1[q]
    a2 = v2.get(q, "")
    if a1 == a2:
        same += 1
    else:
        diff += 1
        if len(a2) > len(a1):
            ch = "v5多选了"
        elif len(a2) < len(a1):
            ch = "v5少选了"
        else:
            ch = "答案不同"
        print(f"{q:15s} {a1:8s} {a2:8s} {ch}")

print(f"\n相同={same}  不同={diff}")
