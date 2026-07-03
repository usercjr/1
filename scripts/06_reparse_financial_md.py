# -*- coding: utf-8 -*-
"""
06_reparse_financial_md.py
==========================
用 pymupdf4llm 把表格密集的两个域（financial_reports / financial_contracts）的 PDF
重新解析为**保留表格结构的 Markdown**，覆盖 data/parsed/{domain}/{doc_id}.json。
学自队友 reparse_tables_md.py，但适配我们的 JSON schema（与 01_parse_docs 一致），
所以重建索引（02_build_index.py）无需改动。

为什么：财报/合同的数值题，pdfplumber 抽出的纯文本常把表格数字打乱
（如"4.5 2.19 461.9"糊在一起），导致数值题无据可判。pymupdf4llm 的 markdown
表格能保留"行列对齐"，让营收/净利/现金流/分红等指标可被检索和判断。

合规：纯解析工具，不使用任何模型，离线、零 token。

用法：
    pip install pymupdf4llm
    python scripts/06_reparse_financial_md.py
    python scripts/02_build_index.py        # 重建索引
旧解析自动备份到 data/parsed_md_backup/。加 --domains 可指定域。
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

try:
    import pymupdf4llm
except ImportError:
    raise SystemExit("需要 pymupdf4llm：pip install pymupdf4llm")

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "public_dataset_upload" / "raw"
PARSED_DIR = ROOT / "data" / "parsed"
BACKUP_DIR = ROOT / "data" / "parsed_md_backup"

DEFAULT_DOMAINS = ["financial_reports", "financial_contracts"]
PDF_EXTS = {".pdf"}


def reparse_one(domain: str, pdf_path: Path) -> dict:
    """pymupdf4llm 逐页转 markdown，产出与 01_parse_docs 一致的 schema。"""
    pages_md = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True, show_progress=False)
    pages = []
    for i, pg in enumerate(pages_md, start=1):
        md = pg.get("text", "") if isinstance(pg, dict) else str(pg)
        pages.append({"page_num": i, "text": (md or "").strip()})
    raw_text = "\f".join(p["text"] for p in pages)
    doc_id = pdf_path.stem
    return {
        "doc_id": doc_id,
        "domain": domain,
        "source": str(pdf_path.relative_to(RAW_DIR).as_posix()),
        "format": "pdf",
        "title": doc_id,
        "pages": pages,
        "tables": [],   # 表格已内联在 markdown 文本里
        "raw_text": raw_text,
        "stats": {"chars": len(raw_text), "pages": len(pages), "tables": 0},
        "parser": "pymupdf4llm",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="*", default=DEFAULT_DOMAINS)
    ap.add_argument("--raw_dir", default=str(RAW_DIR))
    args = ap.parse_args()
    raw_dir = Path(args.raw_dir)

    ok = err = 0
    for domain in args.domains:
        dom_raw = raw_dir / domain
        if not dom_raw.exists():
            print(f"[跳过] 无 raw 目录: {dom_raw}")
            continue
        out_dom = PARSED_DIR / domain
        out_dom.mkdir(parents=True, exist_ok=True)
        bak_dom = BACKUP_DIR / domain
        bak_dom.mkdir(parents=True, exist_ok=True)

        pdfs = [p for p in dom_raw.rglob("*") if p.suffix.lower() in PDF_EXTS]
        print(f"[{domain}] 待重解析 {len(pdfs)} 个 PDF")
        for pdf in sorted(pdfs):
            out_file = out_dom / f"{pdf.stem}.json"
            # 备份旧解析
            if out_file.exists():
                shutil.copy(out_file, bak_dom / out_file.name)
            try:
                rec = reparse_one(domain, pdf)
                out_file.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
                ok += 1
                print(f"  OK {pdf.stem}  chars={rec['stats']['chars']:,} pages={rec['stats']['pages']}")
            except Exception as e:
                err += 1
                print(f"  ERR {pdf.stem}: {type(e).__name__}: {e}")

    print(f"\n完成: ok={ok}, err={err}。旧解析已备份到 {BACKUP_DIR}")
    print("下一步: python scripts/02_build_index.py  重建索引")


if __name__ == "__main__":
    main()
