# -*- coding: utf-8 -*-
"""
07_build_doc_aliases.py
=======================
离线构建"实体→doc_id"别名映射（B 榜文档定位用，纯规则 0 token，合规）。

输出 data/index/doc_aliases.json：
{
  "<domain>": {
    "<doc_id>": [ ["别名词1"], ["公司名","2024"], ... ]   # 内层列表=须全部命中题面
  }
}

匹配语义：某别名（术语组）内所有词都出现在题面文本中 → 该 doc_id 成为强候选。

各域来源：
- financial_reports: doc_id 自带公司代码+年份（annual_byd_2024_report），
  公司代码→中文名为手工字典（10 份文档，稳定）；别名 = [公司名, 年份]。
- financial_contracts: 封面首块抽 股票/证券简称、股票代码、公司全名。
- insurance: 封面首块抽产品名（…保险/…条款），并生成去公司前缀的短别名（智盈金生等）。
- regulatory: 主文档（无 _att 后缀）title 中的《法规名》，同时挂到其 _att 子文档。
- research: 标题噪声大、BM25 召回已够（18/20@top5），不建别名。

用法：python scripts/07_build_doc_aliases.py   （构建后自动打印目检报告）
"""
from __future__ import annotations

import json
import pickle
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "data" / "index"
OUT_PATH = INDEX_DIR / "doc_aliases.json"

# ---------------------------------------------------------------------------
# financial_reports：公司代码→中文名（含常用别称）
# ---------------------------------------------------------------------------
_FIN_COMPANY = {
    "byd": ["比亚迪"],
    "catl": ["宁德时代"],
    "chinamobile": ["中国移动"],
    "cmb": ["招商银行", "招行"],
    "cscec": ["中国建筑"],
    "midea": ["美的"],
}

# insurance 公司前缀（产品名必须锚定在公司前缀上，避免抽到自由文本碎片）
_INS_PREFIX_STR = r"平安产险|平安人寿|平安养老|平安|国寿|中国人寿|太平洋|太保|泰康|众安在线|众安|人保|太平"
_INS_PREFIX = re.compile(r"^(%s)" % _INS_PREFIX_STR)
# 产品名模式：公司前缀锚定 + 中间名（允许括号，如富鸿金生（悦享版））+ 险种后缀
_INS_PRODUCT = re.compile(
    r"((?:%s)[一-龥A-Za-zｅＥeE（）()]{0,22}?"
    r"(?:专属商业养老保险|养老年金保险（分红型）|养老年金保险|终身寿险|重大疾病保险|"
    r"住院医疗保险|医疗保险|意外伤害保险|家庭财产综合保险|家庭财产保险|"
    r"食品安全责任保险|特种车商业保险|年金保险|寿险|保险))" % _INS_PREFIX_STR
)
_INS_SUFFIX = re.compile(
    r"(专属商业养老保险|养老年金保险（分红型）|养老年金保险|终身寿险|重大疾病保险|"
    r"住院医疗保险|医疗保险|意外伤害保险|家庭财产综合保险|家庭财产保险|"
    r"食品安全责任保险|特种车商业保险|年金保险|寿险|保险)$"
)
# 短别名黑名单：泛词做别名会把半个域都召回
_INS_GENERIC = {"财产", "健康", "医疗", "养老", "年金", "意外", "人寿", "保险", "产险"}
# 公司短称 + 险种类型 的组合别名（题面常用"众安特种车/平安e生保"式说法）
_INS_COMPANY_SHORT = [("平安", "平安"), ("众安", "众安"), ("太平洋", "太保"), ("太保", "太保"),
                      ("中国人寿", "国寿"), ("国寿", "国寿"), ("泰康", "泰康"), ("人保", "人保")]
_INS_TYPE_SHORT = ["特种车", "食品安全", "家庭财产", "预防接种", "营运交通", "重大疾病",
                   "住院医疗", "白血病", "e生保", "ｅ生保"]
# 险种俗称：题面常说"家财险/食责险"，与条款正文用词不同
_INS_TYPE_COLLOQUIAL = {"家庭财产": ["家财"], "食品安全": ["食责"], "预防接种": ["接种"]}

# fc 封面字段
_FC_ABBR = re.compile(r"(?:股票|证券)简称[:：]\s*([一-龥A-Za-z0-9]{2,12})")
_FC_CODE = re.compile(r"(?:股票|证券)代码[:：]\s*(\d{6})")
_FC_COMPANY = re.compile(r"([一-龥（）()]{4,26}?(?:股份有限公司|集团有限公司|（集团）有限公司|有限公司))")

# regulatory 主文档标题里的《法规名》
_REG_NAME = re.compile(r"《([^《》]{4,40})》")


def _load(domain):
    with (INDEX_DIR / f"{domain}.pkl").open("rb") as f:
        return pickle.load(f)


def _doc_heads(domain, n_chunks=2):
    """每文档前 n 个 chunk 的拼接文本 + title。"""
    idx = _load(domain)
    heads, titles = {}, {}
    for c in idx["chunks"]:
        d = c["doc_id"]
        if d not in heads:
            heads[d] = []
            titles[d] = c.get("title", "")
        if len(heads[d]) < n_chunks:
            heads[d].append(c.get("text") or "")
    return {d: "\n".join(v) for d, v in heads.items()}, titles


def build_fin():
    heads, _ = _doc_heads("financial_reports")
    out = {}
    for d in heads:
        m = re.match(r"annual_([a-z]+)_(\d{4})_report", d)
        if not m:
            continue
        code, year = m.group(1), m.group(2)
        names = _FIN_COMPANY.get(code, [])
        out[d] = [[nm, year] for nm in names] + [[code, year]]
    return out


def build_fc():
    heads, _ = _doc_heads("financial_contracts", n_chunks=2)
    # 兜底用的深层窗口（text13 的发行人名在 p6）；只给"发行人：XX"兜底用，
    # 避免主抽取在深层扫到保荐机构/可比公司的简称造成误报
    deep_heads, _ = _doc_heads("financial_contracts", n_chunks=12)
    out = {}
    for d, head in heads.items():
        aliases = []
        for m in _FC_ABBR.finditer(head):
            aliases.append([m.group(1)])
        for m in _FC_CODE.finditer(head):
            aliases.append([m.group(1)])
        # 公司全名取封面前 300 字里的第一个（发行人）
        m = _FC_COMPANY.search(head[:300])
        if m:
            full = m.group(1)
            aliases.append([full])
            # 去后缀短名：广东省广晟控股集团有限公司 → 广晟控股
            short = re.sub(r"(股份有限公司|集团有限公司|（集团）有限公司|有限公司)$", "", full)
            short = re.sub(r"^(广东省|江苏|山东|海南|福建省|陕西省|深圳市|厦门|上海|北京)", "", short)
            if 2 <= len(short) <= 8:
                aliases.append([short])
        # 兜底：封面无简称/公司名的（如 text13 声明页开头），深层窗口抽"发行人：XX"
        if not aliases:
            m = re.search(r"发行人[:：]?\s*([一-龥（）()]{4,26}?(?:公司|集团))", deep_heads.get(d, ""))
            if m:
                full = m.group(1)
                aliases.append([full])
                short = re.sub(r"^(中国)", "", re.sub(r"(集团|股份有限公司|有限公司)$", "", full))
                if 2 <= len(short) <= 10:
                    aliases.append([short])
        # 去重
        seen, uniq = set(), []
        for a in aliases:
            k = tuple(a)
            if k not in seen:
                seen.add(k)
                uniq.append(a)
        out[d] = uniq
    return out


def build_ins():
    # 每文档取前 8 个 chunk，逐 chunk 在头部窗口内抽
    # （产品名标题可能在目录页之后，如 doc15 鑫享添盈在第 4 块）
    idx = _load("insurance")
    chunks_by_doc = {}
    for c in idx["chunks"]:
        chunks_by_doc.setdefault(c["doc_id"], [])
        if len(chunks_by_doc[c["doc_id"]]) < 8:
            chunks_by_doc[c["doc_id"]].append(c.get("text") or "")
    out = {}
    for d, chunks in chunks_by_doc.items():
        aliases = []
        cand = set()
        for t in chunks:
            for m in _INS_PRODUCT.finditer(t[:250]):
                name = m.group(1)
                if len(name) < 6:  # 滤掉"平安保险"这类泛称
                    continue
                cand.add(name)
        for name in sorted(cand, key=len, reverse=True)[:3]:
            aliases.append([name])
            # 短别名：去公司前缀、去险种后缀（智盈金生 / 增益宝 / 富鸿金生）
            short = _INS_PREFIX.sub("", name)
            short = _INS_SUFFIX.sub("", short)
            short = re.sub(r"[（(][^（）()]*[)）]", "", short).strip("（）() ")
            if 2 <= len(short) <= 10 and short != name and short not in _INS_GENERIC:
                aliases.append([short])
        # 组合别名：公司短称 + 险种类型（"众安 特种车"须同现才算命中）
        head = "\n".join(chunks)[:400]
        comp = next((s for pat, s in _INS_COMPANY_SHORT if pat in head), None)
        if comp:
            for ty in _INS_TYPE_SHORT:
                if ty in head:
                    aliases.append([comp, ty])
                    for coll in _INS_TYPE_COLLOQUIAL.get(ty, []):
                        aliases.append([comp, coll])
        seen, uniq = set(), []
        for a in aliases:
            k = tuple(a)
            if k not in seen:
                seen.add(k)
                uniq.append(a)
        out[d] = uniq
    return out


def build_reg():
    heads, titles = _doc_heads("regulatory", n_chunks=1)
    out = {}
    # 主文档：doc_id 无 _att 后缀；其《法规名》同时挂到 _att 子文档
    children = {}
    for d in heads:
        m = re.match(r"(.+?)_att\d+$", d)
        if m:
            children.setdefault(m.group(1), []).append(d)
    for d, head in heads.items():
        if re.search(r"_att\d+$", d):
            continue
        # 标题/头部先归一化空白（HTML 解析常在名称中间断行）
        t_norm = re.sub(r"\s+", "", titles.get(d, ""))
        h_norm = re.sub(r"\s+", "", head[:300])
        names = set(_REG_NAME.findall(t_norm)) | set(_REG_NAME.findall(h_norm))
        # "关于修改〈X〉的决定"同时展开内层名 X（题面引用的是被修改的法规名）
        for n in list(names):
            for inner in re.findall(r"[〈《]([^〈〉《》]{5,40})[〉》]", n):
                names.add(inner)
        # 长名"X准则第N号——Y"拆段：题面常只引用 X 或 Y
        for n in list(names):
            if "——" in n:
                left, right = n.split("——", 1)
                left = re.sub(r"第[0-9一二三四五六七八九十]+号$", "", left)
                if len(left) >= 6:
                    names.add(left)
                if len(right) >= 6:
                    names.add(right)
        names = {n for n in names if len(n) >= 5}
        aliases = [[n] for n in sorted(names, key=len, reverse=True)[:4]]
        # 行政处罚决定书：无《》名，题面引用当事人名（世纪华通/苏亚金诚）
        if "行政处罚决定书" in t_norm or "行政处罚决定书" in h_norm:
            m = re.search(r"当事人[:：]?\s*([一-龥]{2,6}?(?:集团|公司|会计师事务所|事务所))", h_norm)
            if m:
                party = re.sub(r"^(浙江|江苏|山东|广东|北京|上海|深圳)", "", m.group(1))
                party_core = re.sub(r"(集团|公司|会计师事务所|事务所)$", "", party)
                if len(party_core) >= 3:
                    aliases.append([party_core])
            for p in re.findall(r"（([一-龥]{3,10})）", t_norm):
                if [p] not in aliases:
                    aliases.append([p])
        if not aliases:
            continue
        for target in [d] + children.get(d, []):
            out.setdefault(target, [])
            for a in aliases:
                if a not in out[target]:
                    out[target].append(a)
    return out


def main():
    aliases = {
        "financial_reports": build_fin(),
        "financial_contracts": build_fc(),
        "insurance": build_ins(),
        "regulatory": build_reg(),
        "research": {},
    }
    OUT_PATH.write_text(json.dumps(aliases, ensure_ascii=False, indent=1), encoding="utf-8")
    # 目检报告
    for dom in ("financial_reports", "financial_contracts", "insurance"):
        print(f"===== {dom}")
        for d, al in aliases[dom].items():
            print(f"  {d}: {al}")
    reg = aliases["regulatory"]
    print(f"===== regulatory: {len(reg)} docs 有别名（含 att 子文档）")
    for d in list(reg)[:8]:
        print(f"  {d}: {reg[d]}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
