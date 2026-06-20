# -*- coding: utf-8 -*-
"""
02_build_index.py
=================
离线索引构建（不计 token）：
  1) 读取 data/parsed/{domain}/*.json
  2) 按域规则切片（300-600 字/块，含 overlap）
  3) jieba 分词 + 自定义金融词典
  4) rank_bm25.BM25Okapi 建索引（每个域一份）
  5) pickle 落盘到 data/index/{domain}.pkl + chunks 元数据到 data/chunks/{domain}.jsonl

切片策略：
  - regulatory  : 按"第 X 条"切；strict_v3_* 给高优先级
  - insurance   : 按"第 X 条 / X.Y 条"切，无标题则按页+滑窗
  - financial_contracts / financial_reports : 按页切，超长页用滑窗
  - research    : 按页切（研报每页一个话题）

每条 chunk schema:
{
  "chunk_id":   "doc_id::000",
  "doc_id":     "...",
  "domain":     "...",
  "title":      "<doc 标题>",
  "page":       3,
  "section":    "第四十七条" / "" ,
  "text":       "...",
  "n_chars":    580,
  "priority":   1.0          # 检索时作为 score 的乘子
}

使用：
    python scripts/02_build_index.py \
        --parsed_dir data/parsed \
        --out_dir    data/index \
        --chunks_dir data/chunks
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    import jieba
    import jieba.analyse  # noqa: F401
except ImportError:
    jieba = None
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw):  # type: ignore
        return x

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("index")

# ---------------------------------------------------------------------------
# 切片超参
# ---------------------------------------------------------------------------
MAX_CHARS = 600        # 单 chunk 上限
MIN_CHARS = 80         # 短于此与上一块合并
TARGET_CHARS = 500     # 软目标
OVERLAP = 80           # 滑窗 overlap
ARTICLE_HARD_CAP = 1200  # 单条条文超此则强切

# 中文数字
_CHN_NUM = "一二三四五六七八九十百千零0-9"
RE_ARTICLE = re.compile(rf"(第[{_CHN_NUM}]+条[\s　]?)")
RE_SECTION = re.compile(rf"(第[{_CHN_NUM}]+(?:章|节|编|部分)[\s　]?[^\n]{{0,40}})")
RE_BREAK_PUNCT = ("。", "\n", "！", "？", "；", ";", "!", "?")

# ---------------------------------------------------------------------------
# 自定义金融词典（按需扩充）
# ---------------------------------------------------------------------------
FINANCE_TERMS = [
    # 保险
    "保险责任", "身故保险金", "现金价值", "退保金额", "账户价值", "保单贷款", "犹豫期",
    "受益人", "宽限期", "复效", "全残", "等待期", "豁免保险费", "重大疾病",
    # 监管
    "受益所有人", "尽职调查", "可疑交易报告", "特别决议", "普通决议", "上市公司治理准则",
    "章程指引", "独立董事", "反洗钱", "客户身份资料", "支付机构",
    # 财报
    "营业收入", "归属于上市公司股东的净利润", "经营活动产生的现金流量净额",
    "研发投入", "扣非净利润", "资产负债率", "毛利率", "净利率", "每股收益",
    # 合同
    "募集说明书", "发行人", "主承销商", "受托管理人", "募集资金", "信用评级",
    # 通用
    "复合增速", "市场规模", "市占率", "毛利率", "净利率",
]

# 域优先级：用正则更精细地判定（注意：判定顺序自上而下，首个命中生效）
# 重要：csrc_NNNN_attN 是正式法规附件 PDF（干货），与 csrc_NNNN.html 通告（噪声）必须分开
DOC_PRIORITY_RULES: List[Tuple[str, str, float]] = [
    # (domain, pattern, priority)
    ("regulatory", r"^strict_v3_",            1.0),  # 央行/金监会等正式令
    ("regulatory", r"^strict_csrc_",          1.0),  # 证监会正式准则
    ("regulatory", r"^csrc_\d+_att\d+$",      1.0),  # 证监会公告附件 PDF（正式法规）
    ("regulatory", r"^csrc_\d+$",             0.7),  # 证监会公告 HTML 本身（含部分正式通告，轻度降权而非压制）
]
DEFAULT_PRIORITY = 1.0

_COMPILED_PRIORITY = [(d, re.compile(p), pri) for d, p, pri in DOC_PRIORITY_RULES]


def doc_priority(domain: str, doc_id: str) -> float:
    for d, pat, p in _COMPILED_PRIORITY:
        if d == domain and pat.search(doc_id):
            return p
    return DEFAULT_PRIORITY


# ===========================================================================
# 1. 分词
# ===========================================================================
def init_jieba():
    if jieba is None:
        raise RuntimeError("需要 jieba: pip install jieba")
    for w in FINANCE_TERMS:
        jieba.add_word(w, freq=2_000)
    # 关闭烦人的初始化日志
    jieba.setLogLevel(logging.WARNING)


_NON_TOKEN = re.compile(r"[^一-龥A-Za-z0-9%·年月日条款章节项]+")


def tokenize(text: str) -> List[str]:
    text = _NON_TOKEN.sub(" ", text)
    return [t for t in jieba.lcut(text) if t.strip()]


# ===========================================================================
# 2. 切片
# ===========================================================================
def _split_long(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP) -> List[str]:
    """长文本滑窗切片，尝试在标点处断开。"""
    if len(text) <= max_chars:
        return [text]
    out: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + max_chars, n)
        chunk = text[i:end]
        if end < n:
            # 在后 40% 区间找一个标点收尾
            cut = -1
            search_from = int(max_chars * 0.6)
            for p in RE_BREAK_PUNCT:
                idx = chunk.rfind(p, search_from)
                if idx > cut:
                    cut = idx
            if cut > 0:
                chunk = chunk[: cut + 1]
                end = i + cut + 1
        out.append(chunk)
        if end >= n:
            break
        i = max(end - overlap, end - max_chars // 4)
    return out


def chunk_by_articles(text: str) -> List[Tuple[str, str]]:
    """按"第 X 条"切，返回 [(section_marker, text), ...]"""
    parts = RE_ARTICLE.split(text)
    out: List[Tuple[str, str]] = []
    # parts: [pre, marker1, content1, marker2, content2, ...]
    if parts and parts[0].strip():
        out.append(("preamble", parts[0].strip()))
    i = 1
    while i < len(parts):
        marker = parts[i].strip().rstrip()
        content = parts[i + 1] if i + 1 < len(parts) else ""
        body = (marker + " " + content).strip()
        if not body:
            i += 2
            continue
        if len(body) > ARTICLE_HARD_CAP:
            for sub in _split_long(body):
                out.append((marker, sub))
        else:
            out.append((marker, body))
        i += 2
    return out


def chunk_by_pages(pages: List[Dict[str, Any]]) -> List[Tuple[int, str, str]]:
    """按页切，过长页滑窗。返回 [(page_num, section_hint, text), ...]"""
    out: List[Tuple[int, str, str]] = []
    for pg in pages:
        text = (pg.get("text") or "").strip()
        if not text:
            continue
        page_num = pg.get("page_num", 0)
        # 抓页首的章节标题
        section = ""
        m = RE_SECTION.search(text[:200])
        if m:
            section = m.group(1).strip()
        if len(text) <= MAX_CHARS:
            out.append((page_num, section, text))
        else:
            for sub in _split_long(text):
                out.append((page_num, section, sub))
    return out


def _merge_short(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把过短的相邻 chunk 合并（同页、同 doc）。"""
    merged: List[Dict[str, Any]] = []
    for c in chunks:
        if (
            merged
            and merged[-1]["doc_id"] == c["doc_id"]
            and merged[-1]["page"] == c["page"]
            and len(merged[-1]["text"]) < MIN_CHARS
            and len(merged[-1]["text"]) + len(c["text"]) <= MAX_CHARS + 100
        ):
            merged[-1]["text"] = (merged[-1]["text"] + "\n" + c["text"]).strip()
            merged[-1]["n_chars"] = len(merged[-1]["text"])
        else:
            merged.append(c)
    return merged


def chunk_doc(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    domain = doc["domain"]
    doc_id = doc["doc_id"]
    title = doc.get("title") or doc_id
    prio = doc_priority(domain, doc_id)
    chunks: List[Dict[str, Any]] = []

    if domain == "regulatory":
        # 整篇 raw_text 按条切
        for section, text in chunk_by_articles(doc.get("raw_text") or ""):
            chunks.append({
                "doc_id": doc_id, "domain": domain, "title": title,
                "page": 0, "section": section, "text": text,
                "n_chars": len(text), "priority": prio,
            })
    else:
        # 其余按页切
        for page, section, text in chunk_by_pages(doc.get("pages") or []):
            chunks.append({
                "doc_id": doc_id, "domain": domain, "title": title,
                "page": page, "section": section, "text": text,
                "n_chars": len(text), "priority": prio,
            })

    chunks = _merge_short(chunks)
    # 给 chunk_id
    for i, c in enumerate(chunks):
        c["chunk_id"] = f"{doc_id}::{i:04d}"
    return chunks


# ===========================================================================
# 3. 主流程
# ===========================================================================
def iter_parsed(parsed_dir: Path) -> Iterable[Dict[str, Any]]:
    for jf in sorted(parsed_dir.rglob("*.json")):
        if jf.name in ("manifest.json",) or jf.name.startswith("_"):
            continue
        try:
            obj = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning(f"读取失败 {jf}: {e}")
            continue
        if isinstance(obj, dict) and obj.get("doc_id"):
            yield obj


def build_index_for_domain(
    domain: str,
    chunks: List[Dict[str, Any]],
    out_dir: Path,
    chunks_dir: Path,
):
    if not chunks:
        log.warning(f"{domain}: 无 chunk，跳过")
        return

    # 1) 落 chunks jsonl
    chunks_dir.mkdir(parents=True, exist_ok=True)
    cf = chunks_dir / f"{domain}.jsonl"
    with cf.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # 2) 分词 + BM25
    tokens_list: List[List[str]] = []
    for c in tqdm(chunks, desc=f"  tokenize[{domain}]", leave=False):
        # 切片文本 + section/title 一起入 BM25（提高条款号、文档名命中）
        text_for_index = " ".join([c.get("title", ""), c.get("section", ""), c["text"]])
        tokens_list.append(tokenize(text_for_index))

    if BM25Okapi is None:
        raise RuntimeError("需要 rank_bm25: pip install rank_bm25")
    bm25 = BM25Okapi(tokens_list)

    # 3) pickle
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl = out_dir / f"{domain}.pkl"
    with pkl.open("wb") as f:
        pickle.dump({
            "domain": domain,
            "bm25": bm25,
            "chunks": chunks,
            "tokens": tokens_list,
        }, f)

    avg_len = sum(c["n_chars"] for c in chunks) / max(len(chunks), 1)
    log.info(
        f"[{domain}] chunks={len(chunks)}  avg_chars={avg_len:.0f}  "
        f"-> {pkl.name} ({pkl.stat().st_size/1024:.0f} KB)"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed_dir", default="data/parsed")
    ap.add_argument("--out_dir",    default="data/index")
    ap.add_argument("--chunks_dir", default="data/chunks")
    args = ap.parse_args()

    parsed_dir = Path(args.parsed_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    chunks_dir = Path(args.chunks_dir).resolve()

    if not parsed_dir.exists():
        log.error(f"parsed_dir 不存在: {parsed_dir}")
        sys.exit(2)

    init_jieba()

    # 按域分桶
    by_domain: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    n_docs = 0
    log.info(f"扫描 {parsed_dir} ...")
    for doc in tqdm(list(iter_parsed(parsed_dir)), desc="chunk"):
        chunks = chunk_doc(doc)
        by_domain[doc["domain"]].extend(chunks)
        n_docs += 1
    log.info(f"共 {n_docs} 个 doc，切出 chunk 总数: {sum(len(v) for v in by_domain.values())}")

    # 每域建索引
    for domain, chunks in by_domain.items():
        build_index_for_domain(domain, chunks, out_dir, chunks_dir)

    # 写一份索引 manifest
    summary = {
        "domains": {
            d: {"docs": len({c["doc_id"] for c in cs}), "chunks": len(cs)}
            for d, cs in by_domain.items()
        },
        "total_chunks": sum(len(v) for v in by_domain.values()),
        "params": {
            "MAX_CHARS": MAX_CHARS,
            "MIN_CHARS": MIN_CHARS,
            "OVERLAP": OVERLAP,
            "ARTICLE_HARD_CAP": ARTICLE_HARD_CAP,
        },
    }
    (out_dir / "index_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info(f"summary -> {out_dir / 'index_summary.json'}")


if __name__ == "__main__":
    main()
