# -*- coding: utf-8 -*-
"""
08_run_b.py — B 榜端到端跑批
=============================
与 A 榜差异：
  - 题目：upload_b/question_b/*.json|jsonl（BOM/两种格式混合），无 answer_format，
    用 type 映射（单选题→mcq 多选题→multi 判断题→tf 计算题/抽取题→calc）
  - 无 doc_ids：走 blind 管线（别名+编号+数字锚定位 → Qwen 文档选择 → 证据组装）
  - 计算/抽取题（options 为空）：专用 prompt，多空答案（answer_1..4），
    格式硬约束（两位小数/百分号/中文日期/排序>）；槽数与格式提示取自官方 submit.csv 占位符
  - 每题输出 reasoning（模型同调用 JSON 字段，token 完全可审计）
  - 输出 CSV：qid,answer_1..4,prompt_tokens,completion_tokens,total_tokens,reasoning
  - 模型白名单：Qwen3.6/3.5 系列（默认 qwen3.6-plus）

用法：
    QWEN_MODEL=qwen3.6-plus python scripts/08_run_b.py --out_dir outputs_b1
    可选 --limit 5 冒烟 / --resume 续跑 / --only_qids
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm

from agent import config
from agent.llm import QwenLLM, get_token_stats
from agent.retriever import Retriever, _num_norm
from agent.solver import Solver, _budget_by_doc
from agent.postproc import extract_or_none
from agent.prompts import render_evidence, _load_domain_system

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("run_b")

TYPE_MAP = {"单选题": "mcq", "多选题": "multi", "判断题": "tf",
            "计算题": "calc", "抽取题": "calc"}


# ---------------------------------------------------------------------------
# 加载
# ---------------------------------------------------------------------------
def load_b_questions(qdir: Path) -> List[Dict[str, Any]]:
    qs: List[Dict[str, Any]] = []
    for fn in sorted(qdir.glob("*")):
        if fn.suffix not in (".json", ".jsonl"):
            continue
        text = fn.read_text(encoding="utf-8-sig").strip()
        try:
            d = json.loads(text)
            items = d if isinstance(d, list) else [d]
        except json.JSONDecodeError:
            items = [json.loads(l) for l in text.splitlines() if l.strip()]
        qs.extend(items)
    for q in qs:
        q["answer_format"] = TYPE_MAP.get(q.get("type", ""), "multi")
    return qs


def load_template(fp: Path) -> Dict[str, List[str]]:
    """官方 submit.csv 占位符 → {qid: [slot1_placeholder, ...]}（槽数+格式提示）。"""
    out: Dict[str, List[str]] = {}
    for r in csv.DictReader(open(fp, encoding="utf-8-sig")):
        if r["qid"] == "summary":
            continue
        slots = [(r.get(f"answer_{i}") or "").strip() for i in range(1, 5)]
        out[r["qid"]] = [s for s in slots if s]
    return out


# ---------------------------------------------------------------------------
# 计算/抽取题
# ---------------------------------------------------------------------------
_CALC_RULES = """格式硬性要求（逐条遵守，格式错误按零分计）：
- 数值答案只填数字不带单位，保留两位小数（如 1234.56；整数也写成 100.00）
- 百分数必须带 %，保留两位小数（如 12.34%）
- 日期必须用中文格式 YYYY年M月D日（如 2026年1月1日，月/日不补零）
- 排序类用英文半角 > 连接，前后不加空格（如 甲>乙>丙）
- 文本类按题目要求填写完整文本，不添加说明
- 多个答案严格按题目要求的顺序排列"""


def build_calc_prompt(q: Dict[str, Any], hits, roster: str, doc_ord, slots: List[str],
                      max_chunk_chars: int) -> str:
    system = _load_domain_system((q.get("domain") or "").lower())
    evidence = render_evidence(hits, max_chars=max_chunk_chars, doc_ord=doc_ord)
    if roster:
        evidence = roster + "\n\n" + evidence
    n = max(1, len(slots))
    hint = ""
    if slots:
        hint = "\n提交模板占位符提示（答案格式须与之同型）：" + " | ".join(
            f"answer_{i+1} 形如 {s}" for i, s in enumerate(slots))
    return (f"{system}\n\n【证据】\n{evidence}\n\n【题目（计算/抽取题）】{q.get('question','')}\n\n"
            f"本题需给出 {n} 个答案。{hint}\n{_CALC_RULES}\n\n"
            "解题要求：先在证据中定位原始数据（引用证据编号与原文数字），列出计算式，"
            "再按格式要求写出最终答案。证据不足时基于最接近的证据给出最合理答案，不允许留空。\n"
            '只输出一行 JSON：{"answers":["答案1","答案2"],'
            '"reasoning":"<3-5句、150字内的干净推理摘要：文档定位→关键原始数据（含证据编号）→计算式→结论；只写最终定稿陈述，严禁出现犹豫、自我纠正、草稿式语言>"}')


def _norm_number(s: str, hint: str) -> str:
    """按占位提示规范化数值/百分号格式（两位小数）。日期/文本/排序不动。"""
    t = (s or "").strip().replace("，", ",").replace(" ", "")
    is_pct_hint = hint.endswith("%")
    m = re.fullmatch(r"(-?[\d,]+(?:\.\d+)?)(%?)", t)
    if not m:
        return (s or "").strip()
    num, pct = m.group(1).replace(",", ""), m.group(2)
    try:
        val = float(num)
    except ValueError:
        return (s or "").strip()
    if pct or is_pct_hint:
        return f"{val:.2f}%"
    return f"{val:.2f}"


def parse_calc(raw: str, slots: List[str]) -> (List[str], str):
    """从模型输出抽 answers 数组 + reasoning。"""
    s = (raw or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    answers, reasoning = [], ""
    try:
        obj = json.loads(s)
        answers = [str(a) for a in obj.get("answers", [])]
        reasoning = str(obj.get("reasoning", ""))
    except Exception:
        m = re.search(r'"answers"\s*:\s*\[([^\]]*)\]', s, re.S)
        if m:
            answers = re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))
        m2 = re.search(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', s, re.S)
        if m2:
            reasoning = m2.group(1)
    n = max(1, len(slots))
    answers = answers[:4]
    while len(answers) < n:
        answers.append(answers[-1] if answers else "")
    answers = [_norm_number(a, slots[i] if i < len(slots) else "")
               for i, a in enumerate(answers)]
    return answers, reasoning


def extract_reasoning(raw: str) -> str:
    """选择题：抽模型 JSON 里的 reasoning；缺失则拼 analysis。"""
    m = re.search(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', raw or "", re.S)
    if m and len(m.group(1)) >= 20:
        return m.group(1).replace('\\"', '"').replace("\\n", " ")
    an = Solver._parse_analysis(raw or "")
    if an:
        return "逐项判断：" + "；".join(f"{k}:{v[:80]}" for k, v in sorted(an.items()))
    return ""


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def solve_calc(solver: Solver, q: Dict[str, Any], slots: List[str], meta) -> Dict[str, Any]:
    """计算/抽取题：blind 文档定位 + 证据组装 + calc prompt。"""
    domain = q.get("domain")
    query = q.get("question", "")
    r = solver.retriever
    alias_hits = [d for d, _ in r.alias_candidates(query, domain)]
    resolved = r.resolve_doc_ids(query, domain)
    selected = []
    if resolved and domain not in ("regulatory",):
        selected = solver._select_docs_llm(q, resolved, domain, meta)
    if selected:
        merged = alias_hits + [d for d in selected if d not in alias_hits]
    else:
        merged = alias_hits or resolved
    doc_ids = merged or None
    hits, _g = solver._option_augmented_retrieve(q, domain=domain, doc_ids=doc_ids)
    hits = solver._inject_doc_headers(hits, (alias_hits or None), domain)
    hits = solver._pin_fin_key_chunks(hits, q, (alias_hits or merged or None), domain)
    hits = solver._pin_ins_clause_chunks(hits, q, (alias_hits or merged or None), domain)
    budget = config.EVIDENCE_TOTAL_CHAR_BUDGET + (3000 if domain == "financial_reports" else 0)
    hits = _budget_by_doc(hits, budget)
    roster, doc_ord = solver._doc_roster((alias_hits or None), domain)
    prompt = build_calc_prompt(q, hits, roster, doc_ord, slots, solver.max_chunk_chars)
    raw = solver.llm.chat(prompt, max_tokens=1200, meta=meta)
    answers, reasoning = parse_calc(raw, slots)
    return {"answers": answers, "reasoning": reasoning, "raw": raw,
            "evidence": [{"doc_id": h["doc_id"], "chunk_id": h.get("chunk_id"),
                          "section": h.get("section", ""), "page": h.get("page", 0)}
                         for h in hits]}


def write_csv(records: List[Dict[str, Any]], out_path: Path):
    tp = sum(r["prompt_tokens"] for r in records)
    tc = sum(r["completion_tokens"] for r in records)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["qid", "answer_1", "answer_2", "answer_3", "answer_4",
                    "prompt_tokens", "completion_tokens", "total_tokens", "reasoning"])
        w.writerow(["summary", "", "", "", "", tp, tc, tp + tc, ""])
        for r in records:
            ans = (r["answers"] + ["", "", "", ""])[:4]
            reasoning = (r.get("reasoning") or "").replace("\n", " ").replace("\r", " ")
            w.writerow([r["qid"], *ans, r["prompt_tokens"], r["completion_tokens"],
                        r["total_tokens"], reasoning])
    log.info(f"submit -> {out_path} (题数 {len(records)}, total {tp+tc:,})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions_dir", default="upload_b/question_b")
    ap.add_argument("--template", default="upload_b/submit.csv")
    ap.add_argument("--index_dir", default="data/index")
    ap.add_argument("--out_dir", default="outputs_b1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only_qids", default="")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "run.jsonl"
    err_log = out_dir / "errors.log"

    questions = load_b_questions(Path(args.questions_dir))
    template = load_template(Path(args.template))
    log.info(f"题目 {len(questions)}，模板槽位 {len(template)}")
    if args.only_qids:
        qset = {x.strip() for x in args.only_qids.split(",") if x.strip()}
        questions = [q for q in questions if q["qid"] in qset]
    if args.limit > 0:
        questions = questions[:args.limit]

    done: Dict[str, Dict[str, Any]] = {}
    if args.resume and jsonl_path.exists():
        for line in jsonl_path.open(encoding="utf-8"):
            try:
                rec = json.loads(line)
                if rec.get("qid"):
                    done[rec["qid"]] = rec
            except Exception:
                continue
        log.info(f"resume: 跳过 {len(done)}")
    todo = [q for q in questions if q["qid"] not in done]

    retriever = Retriever(args.index_dir); retriever.load_all()
    llm = QwenLLM()  # QWEN_MODEL 环境变量控制（白名单：qwen3.6-plus）
    log.info(f"主模型: {llm.model}")
    solver = Solver(retriever, llm=llm)
    stats = get_token_stats()

    f_jsonl = jsonl_path.open("a", encoding="utf-8")
    errs = 0
    for q in tqdm(todo, desc="B"):
        qid = q["qid"]
        before = stats.to_dict()
        try:
            fmt = q["answer_format"]
            slots = template.get(qid, [""])
            meta = {"qid": qid, "fmt": fmt, "domain": q.get("domain"), "mode": "B"}
            if fmt == "calc":
                res = solve_calc(solver, q, slots, meta)
                answers, reasoning = res["answers"], res["reasoning"]
                raw, evidence = res["raw"], res["evidence"]
            else:
                r = solver.solve(q, mode="domain")
                answers = [r.answer]
                reasoning = extract_reasoning(r.raw)
                raw, evidence = r.raw, r.evidence
            after = stats.to_dict()
            rec = {"qid": qid, "type": q.get("type"), "answers": answers,
                   "reasoning": reasoning, "raw": raw, "evidence": evidence,
                   "prompt_tokens": after["prompt_tokens"] - before["prompt_tokens"],
                   "completion_tokens": after["completion_tokens"] - before["completion_tokens"],
                   "total_tokens": after["total_tokens"] - before["total_tokens"]}
        except Exception as e:
            errs += 1
            import traceback
            with err_log.open("a", encoding="utf-8") as ef:
                ef.write(f"\n=== {qid} ===\n{traceback.format_exc()}\n")
            after = stats.to_dict()
            rec = {"qid": qid, "type": q.get("type"),
                   "answers": ["A" if q.get("options") else "0.00"],
                   "reasoning": "", "raw": f"<error {type(e).__name__}>", "evidence": [],
                   "prompt_tokens": after["prompt_tokens"] - before["prompt_tokens"],
                   "completion_tokens": after["completion_tokens"] - before["completion_tokens"],
                   "total_tokens": after["total_tokens"] - before["total_tokens"]}
        f_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n"); f_jsonl.flush()
        done[qid] = rec
    f_jsonl.close()

    order = [q["qid"] for q in questions]
    records = [done[qid] for qid in order if qid in done]
    write_csv(records, out_dir / "submit_b.csv")
    (out_dir / "token_stats.json").write_text(
        json.dumps(stats.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成 {len(records)} 题，错误 {errs}，总 token {stats.to_dict()['total_tokens']:,}")


if __name__ == "__main__":
    main()
