# -*- coding: utf-8 -*-
"""
01_parse_docs.py
================
离线文档解析（不计 token）：把 raw/ 下的 PDF / HTML / TXT 统一解析成结构化 JSON，
写入 parsed/{domain}/{doc_id}.json。

doc_id 约定：
- 规则：取文件名去扩展名 (Path.stem)
- 这与 questions 里的 doc_ids 字段对齐（如 "pack2_text01", "strict_v3_008_..."）

输出 schema（每个 doc 一个 json 文件）：
{
  "doc_id":     "pack2_text01",
  "domain":     "research",
  "source":     "research/pack2_text01.pdf",      # 相对 raw_dir
  "format":     "pdf" | "html" | "txt",
  "title":      "<尽力抽取>",
  "pages":      [{"page_num": 1, "text": "..."}, ...],   # 仅 PDF 有；其它放 [{"page_num":1,"text":full}]
  "tables":     [{"page_num": 3, "rows": [[...], ...]}], # 仅 PDF 提取
  "raw_text":   "<整篇拼接，页面间用 \\f 分页符>",
  "stats":      {"chars": N, "pages": M, "tables": K}
}

使用：
    python scripts/01_parse_docs.py \
        --raw_dir   public_dataset_upload/raw \
        --out_dir   data/parsed \
        --workers   4
    # 增量解析（已有 json 跳过）默认开启；加 --force 覆盖

依赖：pdfplumber, pymupdf, beautifulsoup4, lxml, tqdm
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- 第三方库（用时再 import，方便单机缺包时给清晰报错） ---
try:
    import pdfplumber
except ImportError:
    pdfplumber = None
try:
    import fitz  # pymupdf
except ImportError:
    fitz = None
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw):  # type: ignore
        return x

# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("parse")

# ---------------------------------------------------------------------------
# 域目录约定
# ---------------------------------------------------------------------------
# regulatory 域有三个子目录：txt / html / attachments(pdf)
# 其它域文件直接平铺在域目录下
DOMAIN_DIRS = [
    "insurance",
    "regulatory",
    "financial_contracts",
    "financial_reports",
    "research",
]

PDF_EXTS = {".pdf"}
HTML_EXTS = {".html", ".htm"}
TXT_EXTS = {".txt"}

# 文件名中常见的乱七八糟字符 → 替换/移除
_SAFE_NAME = re.compile(r"[\s　]+")


# ===========================================================================
# 1. PDF 解析
# ===========================================================================
def parse_pdf(path: Path) -> Dict[str, Any]:
    """优先用 pdfplumber（含表格），失败回退 pymupdf（仅文本）。"""
    if pdfplumber is None and fitz is None:
        raise RuntimeError("需要 pdfplumber 或 pymupdf")

    pages: List[Dict[str, Any]] = []
    tables: List[Dict[str, Any]] = []

    if pdfplumber is not None:
        try:
            with pdfplumber.open(str(path)) as pdf:
                for i, page in enumerate(pdf.pages, start=1):
                    try:
                        text = page.extract_text() or ""
                    except Exception:
                        text = ""
                    pages.append({"page_num": i, "text": text})

                    # 表格抽取（容错）：失败就跳过该页
                    try:
                        for tb in page.extract_tables() or []:
                            if tb and any(any(cell for cell in row) for row in tb):
                                tables.append({"page_num": i, "rows": tb})
                    except Exception:
                        pass
            if pages and any(p["text"].strip() for p in pages):
                return _pack_pdf_result(pages, tables)
            log.warning(f"pdfplumber 解析 {path.name} 为空，回退 pymupdf")
        except Exception as e:
            log.warning(f"pdfplumber 失败 {path.name}: {e}, 回退 pymupdf")

    # ---- 回退 pymupdf ----
    if fitz is None:
        raise RuntimeError(f"pdfplumber 失败且 pymupdf 未安装: {path}")
    pages = []
    with fitz.open(str(path)) as doc:
        for i, page in enumerate(doc, start=1):
            try:
                text = page.get_text("text") or ""
            except Exception:
                text = ""
            pages.append({"page_num": i, "text": text})
    return _pack_pdf_result(pages, tables)


def _pack_pdf_result(pages, tables) -> Dict[str, Any]:
    raw_text = "\f".join(p["text"] for p in pages)
    return {
        "format": "pdf",
        "pages": pages,
        "tables": tables,
        "raw_text": raw_text,
        "stats": {
            "chars": len(raw_text),
            "pages": len(pages),
            "tables": len(tables),
        },
    }


# ===========================================================================
# 2. HTML 解析
# ===========================================================================
_HTML_NOISE_TAGS = ("script", "style", "noscript", "header", "footer", "nav", "form")


def parse_html(path: Path) -> Dict[str, Any]:
    if BeautifulSoup is None:
        raise RuntimeError("需要 beautifulsoup4")

    raw = _read_text_any(path)
    soup = BeautifulSoup(raw, "lxml")

    for tag in soup(_HTML_NOISE_TAGS):
        tag.decompose()

    title_tag = soup.find("title")
    h1 = soup.find(["h1", "h2"])
    title = ""
    if title_tag and title_tag.text.strip():
        title = title_tag.text.strip()
    elif h1 and h1.text.strip():
        title = h1.text.strip()

    # 主体优先取常见正文容器，否则取 body 全文
    main = soup.find(id=re.compile(r"(content|main|article|body)", re.I)) \
        or soup.find(class_=re.compile(r"(content|main|article|body)", re.I)) \
        or soup.body \
        or soup

    text = main.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return {
        "format": "html",
        "title": title,
        "pages": [{"page_num": 1, "text": text}],
        "tables": [],
        "raw_text": text,
        "stats": {"chars": len(text), "pages": 1, "tables": 0},
    }


# ===========================================================================
# 3. TXT 解析
# ===========================================================================
def parse_txt(path: Path) -> Dict[str, Any]:
    text = _read_text_any(path)
    # 简单的标题猜测：第一行非空
    title = ""
    for line in text.splitlines():
        if line.strip():
            title = line.strip()[:120]
            break
    return {
        "format": "txt",
        "title": title,
        "pages": [{"page_num": 1, "text": text}],
        "tables": [],
        "raw_text": text,
        "stats": {"chars": len(text), "pages": 1, "tables": 0},
    }


def _read_text_any(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "gb18030", "gbk", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return path.read_bytes().decode("utf-8", errors="ignore")


# ===========================================================================
# 4. 文件枚举 & 调度
# ===========================================================================
def iter_doc_files(raw_dir: Path) -> List[Tuple[str, Path, str]]:
    """
    返回 [(domain, file_path, doc_id), ...]
    doc_id = file.stem
    """
    out: List[Tuple[str, Path, str]] = []
    for domain in DOMAIN_DIRS:
        domain_dir = raw_dir / domain
        if not domain_dir.exists():
            log.warning(f"域目录不存在，跳过: {domain_dir}")
            continue
        # 递归收集（兼容 regulatory/txt|html|attachments 子结构）
        for fp in domain_dir.rglob("*"):
            if not fp.is_file():
                continue
            ext = fp.suffix.lower()
            if ext in PDF_EXTS or ext in HTML_EXTS or ext in TXT_EXTS:
                doc_id = fp.stem
                out.append((domain, fp, doc_id))
    return out


def parse_one(domain: str, fp: Path, doc_id: str, raw_dir: Path) -> Dict[str, Any]:
    ext = fp.suffix.lower()
    if ext in PDF_EXTS:
        body = parse_pdf(fp)
    elif ext in HTML_EXTS:
        body = parse_html(fp)
    elif ext in TXT_EXTS:
        body = parse_txt(fp)
    else:
        raise ValueError(f"不支持的扩展名: {fp}")

    rel = fp.relative_to(raw_dir).as_posix()
    title = body.pop("title", "") if "title" in body else ""
    if not title:
        title = doc_id  # 兜底用文件名

    return {
        "doc_id": doc_id,
        "domain": domain,
        "source": rel,
        "format": body["format"],
        "title": title,
        "pages": body["pages"],
        "tables": body["tables"],
        "raw_text": body["raw_text"],
        "stats": body["stats"],
    }


def _worker(args):
    domain, fp_str, doc_id, raw_dir_str, out_dir_str, force = args
    fp = Path(fp_str)
    raw_dir = Path(raw_dir_str)
    out_dir = Path(out_dir_str)
    out_file = out_dir / domain / f"{doc_id}.json"

    if out_file.exists() and not force:
        return ("skip", domain, doc_id, out_file.stat().st_size, None)

    try:
        record = parse_one(domain, fp, doc_id, raw_dir)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        return ("ok", domain, doc_id, record["stats"]["chars"], None)
    except Exception as e:
        return ("err", domain, doc_id, 0, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=2)}")


# ===========================================================================
# 5. 主入口
# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True, help="raw 根目录，包含 5 个域子目录")
    ap.add_argument("--out_dir", required=True, help="parsed 输出目录")
    ap.add_argument("--workers", type=int, default=4, help="并行进程数")
    ap.add_argument("--force", action="store_true", help="覆盖已有 json")
    ap.add_argument("--limit", type=int, default=0, help="仅处理前 N 个文件（调试用）")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        log.error(f"raw_dir 不存在: {raw_dir}")
        sys.exit(2)

    files = iter_doc_files(raw_dir)
    if args.limit > 0:
        files = files[: args.limit]

    log.info(f"待处理文件: {len(files)} (raw_dir={raw_dir})")
    by_domain: Dict[str, int] = {}
    for d, _, _ in files:
        by_domain[d] = by_domain.get(d, 0) + 1
    for d, n in by_domain.items():
        log.info(f"  - {d}: {n} 个文件")

    work_args = [
        (d, str(fp), doc_id, str(raw_dir), str(out_dir), args.force)
        for d, fp, doc_id in files
    ]

    ok = skip = err = 0
    total_chars = 0
    errors: List[str] = []

    if args.workers <= 1:
        iterator = (_worker(a) for a in work_args)
    else:
        ex = ProcessPoolExecutor(max_workers=args.workers)
        iterator = (f.result() for f in as_completed([ex.submit(_worker, a) for a in work_args]))

    pbar = tqdm(iterator, total=len(work_args), desc="parse")
    for status, domain, doc_id, chars, err_msg in pbar:
        if status == "ok":
            ok += 1
            total_chars += chars
        elif status == "skip":
            skip += 1
        else:
            err += 1
            errors.append(f"[{domain}/{doc_id}] {err_msg}")
        pbar.set_postfix(ok=ok, skip=skip, err=err)

    log.info(f"完成: ok={ok}, skip={skip}, err={err}, 累计字符={total_chars:,}")
    if errors:
        err_log = out_dir / "_parse_errors.log"
        err_log.write_text("\n\n".join(errors), encoding="utf-8")
        log.warning(f"错误明细写入 {err_log}")

    # 写一份 manifest，方便后续脚本枚举
    _write_manifest(out_dir)


def _write_manifest(out_dir: Path):
    manifest = []
    skipped_non_dict = 0
    for jf in out_dir.rglob("*.json"):
        # 跳过元数据文件 / 自身
        if jf.name.startswith("_") or jf.name == "manifest.json":
            continue
        try:
            obj = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        # 只接受 dict 型（doc 记录），跳过 list / 其它
        if not isinstance(obj, dict):
            skipped_non_dict += 1
            continue
        manifest.append({
            "doc_id": obj.get("doc_id"),
            "domain": obj.get("domain"),
            "source": obj.get("source"),
            "format": obj.get("format"),
            "title": (obj.get("title") or "")[:80],
            "chars": obj.get("stats", {}).get("chars", 0),
            "pages": obj.get("stats", {}).get("pages", 0),
            "tables": obj.get("stats", {}).get("tables", 0),
            "path": str(jf.relative_to(out_dir).as_posix()),
        })
    if skipped_non_dict:
        log.warning(f"manifest: 跳过 {skipped_non_dict} 个非 dict 型 json 文件")
    manifest.sort(key=lambda x: (x["domain"] or "", x["doc_id"] or ""))
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info(f"manifest -> {out_dir / 'manifest.json'} ({len(manifest)} 条)")


if __name__ == "__main__":
    main()
