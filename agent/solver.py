# -*- coding: utf-8 -*-
"""
agent/solver.py  (v3)
=====================
单题求解：检索 → 构造 prompt → 调 Qwen → JSON 抽取答案。

v3 升级：
  - 选项增强检索：每个选项单独跑一轮 BM25，融入主检索池，权重 0.7
  - 条款号 boost：见 retriever._search_one
  - 域专用 system prompt：见 prompts.py
  - JSON CoT 输出：max_tokens 调到 400+

接口：
    solver = Solver(retriever, llm)
    result = solver.solve(question, mode="oracle")  # 或 "domain"/"global"
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from agent import config
from agent.llm import QwenLLM
from agent.postproc import normalize, extract_or_none
from agent.prompts import build_prompt
from agent.retriever import Retriever

log = logging.getLogger(__name__)

# 选项增强检索的小权重（防止选项噪声压过主 query）
_OPTION_HIT_WEIGHT = 0.7
# 每个选项额外捞 chunk 数
_OPTION_TOP_K = 2
# 选项文本过短就不单独检索（4=放过 tf 的"正确/错误"，但保留纯产品名选项，
# 如 ins_a_019-D"平安富鸿金生"6 字被 8 挡掉 → 犹豫期条款证据进不来）
_OPTION_MIN_CHARS = 4
# 文档身份头注入开关。队友 A/B 实测此项可能净负(无关封面文本挤占注意力→多选漏选),
# 故设为开关,便于隔离验证(True=注入 / False=关闭)。
_INJECT_DOC_HEADERS = True
# 领域选择性选项增强(学队友):财报/研报关键信息集中在摘要表,选项增强会拉噪声、助长过选,
# 故对这两类关闭、退化为纯主检索;法规/合同/保险(条款分散、多选漏选重灾区)才开。
_PLAIN_DOMAINS = {"financial_reports", "research"}
# 选项保底证据：每个选项的 top-1 命中若没进最终证据，则追加（截断到此字符数）。
# 只追加、不挤占主检索证据 → 修"选项证据根本没进上下文导致漏选"（res_a_002-D、fin_a_012-B 型错误）。
_GUARD_CHARS = 600
# 多选"证据缺口复核"：首答分析里明说"未找到/未提及证据"的未选选项 → 定向补检 + 小额验证调用。
_RECHECK_ENABLED = True
_RECHECK_MAX_PER_Q = 2          # 单题最多复核选项数（控 token）
_RECHECK_TOP_K = 3              # 复核时定向补检 chunk 数
_RECHECK_CHUNK_CHARS = 600
# 选项指纹数字：百分比 / 亿万金额 / 带小数的数 / 6位代码（带单位提取降噪，
# "63%"和"2500亿"是指纹，裸"63"不是）。匹配前去空格/逗号归一化（"2,500 亿"→"2500亿"）。
_RE_OPT_NUM = re.compile(r"\d+(?:\.\d+)?%|\d+(?:\.\d+)?[亿万]|\d+\.\d+|(?<!\d)\d{6}(?!\d)")


def _num_norm_full(s: str) -> str:
    """全归一（去空白+逗号）：只用于从选项短文本提取指纹。"""
    return re.sub(r"[,，\s]", "", s or "")


def _num_norm(s: str) -> str:
    """温和归一（只去数字内千分位逗号）：用于证据/引文的存在性匹配。
    不能去空白——表格"63.51 65.54 66.38"去空白会连成串破坏数字边界。"""
    return re.sub(r"(?<=\d)[,，](?=\d)", "", s or "")


def _opt_fingerprints(text: str) -> List[str]:
    return list(dict.fromkeys(_RE_OPT_NUM.findall(_num_norm_full(text))))


def _num_pattern(n: str, loose: bool = False) -> "re.Pattern":
    """指纹数字 → 匹配正则。
    严格版（缺口检测）：整数带单位的（63% / 2500亿）匹配带小数变体（63.51%），单位必须在。
    宽松版（引文门槛）：单位可省略——表格引文常是"63.51"而 % 在表头
    （fc_a_004 曾因此把正确引文拦下）。"""
    m = re.match(r"^(\d+(?:\.\d+)?)([%亿万])$", n)
    if m:
        base, unit = m.group(1), m.group(2)
        dec = r"(?:\.\d+)?" if "." not in base else ""
        u = r"\s*" + re.escape(unit) + ("?" if loose else "")
        return re.compile(r"(?<!\d)" + re.escape(base) + dec + u)
    if re.match(r"^\d", n):
        return re.compile(r"(?<!\d)" + re.escape(n))
    return re.compile(re.escape(n))


def _nums_in(nums: List[str], text_norm: str, loose: bool = False) -> int:
    """归一化文本中命中的指纹数字个数（按变体正则）。"""
    return sum(1 for n in nums if _num_pattern(n, loose).search(text_norm))


# 首答分析中的"证据缺口"标志词
# 注意"未直接提供/没有直接给出/缺乏直接支持"这类插入"直接"的变体（fin 域高频，
# v8 里 fin_a_002-A/fin_a_005-A/fin_a_012-B 均因此漏触发复核）。
_GAP_RE = re.compile(
    r"未提及|未提到|未找到|没有找到|未给出|未提供|未出现|未涉及|没有提及|"
    r"缺乏.{0,8}(依据|证据|信息|支持)|无相关|没有相关|无法(?:确定|判断|证实)|"
    r"证据中没有|证据未|无证据|没有证据|不足以|没有.{0,4}信息|未能找到|未在证据|"
    r"未直接(?:提供|给出|披露|说明)|没有直接(?:给出|提供|披露)|"
    r"无法直接(?:计算|比较|得出|确认)|未.{0,4}直接支持|"
    r"未明确(?:说明|提及|规定|表述|给出)|"
    r"无依据|没有依据|依据不足|缺少依据"
)
# 文档对照表里每份文档的身份摘要字符数
_ROSTER_HEAD_CHARS = 80

# ---- B 榜（blind）文档选择：Qwen 从候选中挑出题目真正涉及的文档 ----
# 动机：候选 12-15 份直接喂 chunk 检索时，证据中 65-74% 来自错误文档（盲测 33/63 vs
# oracle 50/63 的主因）。合规：Qwen 系列模型参与检索/rerank 是允许的（仅禁非 Qwen）。
_DOC_SELECT_ENABLED = True
_DOC_SELECT_HEAD_CHARS = 120   # 每份候选文档的身份摘要长度
_DOC_SELECT_MAX_TOKENS = 120
_DOC_SELECT_MAX_PICK = 4
# reg 跳过文档选择：案例组合题不点名、处罚决定书封面雷同，选择易挑错同类文档；
# 而 reg 宽候选盲测本就接近 oracle（9/11），15 份候选靠 chunk 级 BM25 排序即可。
_DOC_SELECT_SKIP_DOMAINS = {"regulatory"}

# ---- 财报关键指标钉入（规则式定位，0 token，合规：纯正则非模型）----
# 年报对比题反复需要 营收/归母净利/现金流(+同比)、研发占比、分红，且都在固定结构表块里；
# BM25 常召不回这些表块（v8 里 fin_a_002/005/012 的营收/净利证据缺失即此因）→ 直接钉入。
_FIN_DOMAIN = "financial_reports"
_FIN_PIN_SCORE = 8e9          # 低于身份头(9e9)、高于一切检索结果
_FIN_PIN_CHARS = 700          # 研发/分红钉入块截断；主表不截（数字密集）
_FIN_EXTRA_BUDGET = 3000      # fin 域证据总预算增量（给钉入块腾空间，不挤占检索证据）
_RE_FIN_RD_TRIG = re.compile(r"研发")
_RE_FIN_DIV_TRIG = re.compile(r"分红|股利|股息|派息|派现|利润分配|回购")
_RE_FIN_RD = re.compile(r"研发投入总额占营业收入|研发投入占营业收入|研发投入总额.{0,6}占|研发投入合计.{0,20}占")
_RE_FIN_DIV = re.compile(r"利润分配预案|每 ?10 ?股派|现金分红总额|派发现金红利|末期股息|全年股息|派息")


def _fin_key_table_pred(t: str) -> bool:
    """主要会计数据表块：核心科目 + 同比字样同现。"""
    return (("营业收入" in t or "营业总收入" in t)
            and ("净利润" in t or "现金流量净额" in t)
            and ("增减" in t or "同比" in t))


# ---- 保险条款钉入（同 fin 思路：标准条款结构，规则式定位）----
# ins 证据缺失型错误（如 ins_a_019：富鸿犹豫期"无息退还全部保险费"条款在索引里但
# BM25 召不回）。保险条款有标准小节结构（犹豫期/身故保险金/贷款/免赔额…），
# 题目提到哪类条款，就把每份相关文档的该条款块钉入。
_INS_DOMAIN = "insurance"
_INS_PIN_CHARS = 500
_INS_PIN_MAX = 8  # 单题钉入上限（ins 题常给 4 份文档）
_RE_TOC = re.compile(r"\.{6,}|…{3,}|目录")  # 目录页特征，不作为条款证据


def _not_toc(t: str) -> bool:
    return not _RE_TOC.search(t)


# 只保留有明确失败案例支撑、选块精度经目检过关的规则；宁少勿滥
# （贷款/宽限期/退保/自杀等规则试验中选块精度差，删除——主检索+选项保底已覆盖）。
_INS_PIN_RULES = [
    # (题目触发正则, 条款主关键词, 特征词元组——特征词需出现在主关键词后 120 字窗口内)
    # 犹豫期退费条款正文的标志：犹豫期 + 近处"退还…全部/全额/无息/已收保险费"
    (r"犹豫期", "犹豫期", ("退还您所支付的全部", "退还已收", "全额退还", "无息退还", "退还全部")),
    # 身故保险金规则试验中选块精度差（封面/目录级噪声），已删除。
    (r"免赔额", "免赔额", ("扣除", "抵扣")),
]
_INS_PIN_MIN_SCORE = 11  # 必须命中邻近特征词(+10)才可能过阈，滤掉交叉引用块


def _clause_score(c: Dict[str, Any], kw: str, prox_terms) -> int:
    """条款块打分：主关键词在小节标题/块首 +6（条款定义处）；出现次数累加；
    主关键词后 120 字内出现特征词 +10（条款正文 vs 交叉引用的核心区分）。"""
    t = c.get("text") or ""
    sec = c.get("section") or ""
    s = (6 if (kw in sec or kw in t[:60]) else 0) + t.count(kw)
    for m in re.finditer(re.escape(kw), t):
        window = t[m.end(): m.end() + 120]
        if any(p in window for p in prox_terms):
            s += 10
            break
    return s


def _budget_by_doc(hits, total_char_budget: int):
    """按文档分配证据字符预算（学自队友）：每个文档至少 max(1500, 总预算/文档数) 字符，
    保证多文档对比题每篇都有足够证据，不被某一篇长文档挤占。每篇至少保留 1 个 chunk。"""
    by_doc: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []
    for h in sorted(hits, key=lambda x: -x.get("score", 0.0)):
        d = h["doc_id"]
        if d not in by_doc:
            by_doc[d] = []
            order.append(d)
        by_doc[d].append(h)
    per_doc = max(1500, total_char_budget // max(1, len(by_doc)))
    kept: List[Dict[str, Any]] = []
    for d in order:
        used = 0
        for h in by_doc[d]:
            tlen = len(h.get("text", "") or "")
            if used > 0 and used + tlen > per_doc:
                continue
            kept.append(h)
            used += tlen
    kept.sort(key=lambda h: -h.get("score", 0.0))
    return kept


@dataclass
class SolverResult:
    qid: str
    answer: str
    raw: str
    evidence: List[Dict[str, Any]]
    mode: str
    domain: str
    fmt: str

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


class Solver:
    def __init__(
        self,
        retriever: Retriever,
        llm: Optional[QwenLLM] = None,
        top_k: int = config.TOP_K_CHUNKS,
        max_chunk_chars: int = config.MAX_CHUNK_CHARS_IN_PROMPT,
        reasoner: Optional[QwenLLM] = None,
    ):
        self.retriever = retriever
        self.llm = llm or QwenLLM()
        self.top_k = top_k
        self.max_chunk_chars = max_chunk_chars
        # 题型路由：mcq/tf 的主答题调用走思考模型（计算/推理题实测 17/17 零回归）；
        # multi 与 复核/文档选择 仍走主模型。None = 不路由。
        self.reasoner = reasoner

    # ---- query 构造 ----
    @staticmethod
    def build_query(q: Dict[str, Any]) -> str:
        opts = q.get("options") or {}
        parts = [q.get("question", "")]
        for k in ("A", "B", "C", "D"):
            v = opts.get(k)
            if v:
                parts.append(v)
        return " ".join(parts)

    # ---- 选项增强检索 ----
    def _option_augmented_retrieve(
        self,
        question: Dict[str, Any],
        domain: Optional[str],
        doc_ids: Optional[List[str]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """
        主检索 top_k chunks + 每个选项额外 top_2 chunks，按 score 合并去重。
        返回 (hits, guards)：guards[选项字母] = 该选项 top-1 命中，供保底追加
        （财报/研报选项命中不进主池抢排名——学队友避免拉噪声——但仍留作保底）。
        """
        q_text = question.get("question", "")
        opts = question.get("options") or {}

        # 1) 主检索：题干+所有选项拼接
        main_query = self.build_query(question)
        pool: Dict[str, Dict[str, Any]] = {}
        for h in self.retriever.search(main_query, domain=domain, doc_ids=doc_ids, top_k=self.top_k):
            pool[h["chunk_id"]] = h

        # 2) 选项级检索：所有域都取 top-1 作保底；仅非 PLAIN 域并入主池参与排名
        guards: Dict[str, Dict[str, Any]] = {}
        for k in ("A", "B", "C", "D"):
            opt_text = opts.get(k) or ""
            if len(opt_text) < _OPTION_MIN_CHARS:
                continue
            opt_query = f"{q_text} {opt_text}"
            opt_hits = self.retriever.search(
                opt_query, domain=domain, doc_ids=doc_ids, top_k=_OPTION_TOP_K
            )
            if not opt_hits:
                continue
            guards[k] = opt_hits[0]
            if domain in _PLAIN_DOMAINS:
                continue  # 不进主池（避免噪声助长过选），仅保底
            for h in opt_hits:
                cid = h["chunk_id"]
                # 降权后入池：若已存在，取 max
                h = {**h, "score": h.get("score", 0.0) * _OPTION_HIT_WEIGHT}
                if cid not in pool or pool[cid].get("score", 0) < h["score"]:
                    pool[cid] = h

        # 3) 按 score 排序，截断
        hits = sorted(pool.values(), key=lambda x: -x.get("score", 0.0))
        # 上限：top_k + 4（每选项 1 个额外）
        max_total = self.top_k + 4
        return hits[:max_total], guards

    # ---- 文档身份头注入 ----
    def _inject_doc_headers(self, hits, trusted_ids, domain):
        """为每份相关文档置顶注入封面/首页 chunk（发行人/证券简称/文件类型等身份信息），
        让模型清楚"每份文档是谁"，修复跨文档比对题的张冠李戴。
        trusted_ids：oracle=题目 doc_ids；blind=别名命中，均额外补齐头。"""
        want: Dict[str, str] = {}
        for h in hits:
            want.setdefault(h["doc_id"], h.get("domain") or domain)
        for d in (trusted_ids or []):
            want.setdefault(d, domain)
        existing = {h["chunk_id"] for h in hits}
        heads: List[Dict[str, Any]] = []
        for d, dom in want.items():
            for c in self.retriever.doc_head(dom, d, n=1):
                if c["chunk_id"] in existing:
                    continue
                heads.append({
                    "chunk_id": c["chunk_id"], "doc_id": c["doc_id"],
                    "domain": c.get("domain", dom), "title": c.get("title", ""),
                    "page": c.get("page", 0),
                    "section": "【文档身份】" + (c.get("section", "") or ""),
                    # 只取封面开头的身份信息（发行人/证券简称/类型），不占用细节证据预算
                    "text": (c["text"] or "")[:220], "score": 9e9,  # 置顶
                })
                existing.add(c["chunk_id"])
        return heads + hits

    # ---- 财报关键指标钉入 ----
    def _pin_fin_key_chunks(self, hits, question, doc_ids, domain):
        """financial_reports oracle 题：每份年报的"主要会计数据"表块固定钉入证据；
        题目涉及研发/分红时再钉对应表块。规则式定位（find_doc_chunks），0 token。"""
        if domain != _FIN_DOMAIN or not doc_ids:
            return hits
        opts = question.get("options") or {}
        qtext = question.get("question", "") + " " + " ".join(v or "" for v in opts.values())
        wants = [(_fin_key_table_pred, 2, "主要指标", 0)]
        if _RE_FIN_RD_TRIG.search(qtext):
            wants.append((lambda t: bool(_RE_FIN_RD.search(t)), 1, "研发投入", _FIN_PIN_CHARS))
        if _RE_FIN_DIV_TRIG.search(qtext):
            wants.append((lambda t: bool(_RE_FIN_DIV.search(t)), 2, "利润分配", _FIN_PIN_CHARS))
        return self._apply_pins(hits, doc_ids, domain, wants, tag="关键指标")

    # ---- 保险条款钉入 ----
    def _pin_ins_clause_chunks(self, hits, question, doc_ids, domain):
        """insurance oracle 题：题目/选项提到标准条款类别（犹豫期/身故保险金/…）时，
        把每份相关文档的对应条款块钉入证据（截 500 字，跳过目录页）。"""
        if domain != _INS_DOMAIN or not doc_ids:
            return hits
        opts = question.get("options") or {}
        qtext = question.get("question", "") + " " + " ".join(v or "" for v in opts.values())
        rules = [(kw, prox) for trig, kw, prox in _INS_PIN_RULES if re.search(trig, qtext)]
        if not rules:
            return hits
        existing = {h["chunk_id"] for h in hits}
        pins: List[Dict[str, Any]] = []
        for kw, prox in rules:
            for d in doc_ids:
                if len(pins) >= _INS_PIN_MAX:
                    break
                cands = self.retriever.find_doc_chunks(
                    domain, d,
                    lambda t, kw=kw: kw in t and _not_toc(t),
                    n=99)
                if not cands:
                    continue
                best = max(cands, key=lambda c: _clause_score(c, kw, prox))
                if _clause_score(best, kw, prox) < _INS_PIN_MIN_SCORE:
                    continue  # 只有交叉引用/弱相关块，宁缺毋滥
                if best["chunk_id"] in existing:
                    continue
                pins.append({
                    "chunk_id": best["chunk_id"], "doc_id": best["doc_id"], "domain": domain,
                    "title": best.get("title", ""), "page": best.get("page", 0),
                    "section": f"【条款·{kw}】" + (best.get("section") or ""),
                    "text": (best.get("text") or "")[:_INS_PIN_CHARS],
                    "score": _FIN_PIN_SCORE,
                })
                existing.add(best["chunk_id"])
        return pins + hits

    # ---- 钉入公共实现 ----
    def _apply_pins(self, hits, doc_ids, domain, wants, tag, max_pins=99):
        existing = {h["chunk_id"] for h in hits}
        pins: List[Dict[str, Any]] = []
        for d in doc_ids:
            for pred, n, label, cap in wants:
                if len(pins) >= max_pins:
                    break
                for c in self.retriever.find_doc_chunks(domain, d, pred, n=n):
                    if c["chunk_id"] in existing:
                        continue
                    text = c.get("text") or ""
                    pins.append({
                        "chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "domain": domain,
                        "title": c.get("title", ""), "page": c.get("page", 0),
                        "section": f"【{tag}·{label}】" + (c.get("section") or ""),
                        "text": text[:cap] if cap else text,
                        "score": _FIN_PIN_SCORE,
                    })
                    existing.add(c["chunk_id"])
        return pins + hits

    # ---- B 榜文档选择（Qwen 从候选中挑题目涉及的文档）----
    def _select_docs_llm(
        self,
        question: Dict[str, Any],
        candidates: List[str],
        domain: Optional[str],
        meta: Dict[str, Any],
    ) -> List[str]:
        """给出候选文档身份摘要，让 Qwen 选出回答本题所需的文档（按题目提及顺序）。
        解析失败/选空 → 返回 []（调用方回退到别名命中/BM25 头部）。"""
        query = self.build_query(question)
        lines = []
        for d in candidates:
            heads = self.retriever.doc_head(domain, d, n=1)
            ident = ((heads[0].get("text") or "") if heads else "")
            ident = ident[:_DOC_SELECT_HEAD_CHARS].replace("\n", " ").strip()
            # 附本题相关的 top-1 片段：封面雷同的文档（处罚决定书/募集说明书）靠内容区分
            snip = ""
            top = self.retriever.search(query, domain=domain, doc_ids=[d], top_k=1)
            if top:
                snip = (top[0].get("text") or "")[:150].replace("\n", " ").strip()
            lines.append(f"- {d}: {ident}\n  相关片段: {snip}")
        opts = question.get("options") or {}
        opt_block = "\n".join(f"{k}. {v}" for k, v in opts.items() if v)
        prompt = (
            "你是金融文档检索助手。下面是候选文档清单（doc_id: 文档身份信息 + 与题目相关的"
            "内容片段）和一道题目。\n"
            "请选出回答该题必须阅读的全部文档（通常 1-4 份，宁全勿漏：题目和选项涉及的每个"
            "主体/产品/法规/报告都要选到；对比类题目须选入所有被对比的文档）。"
            "若题目用\"第一份文档/第二份文档\"等序数指代，请按题意顺序排列输出。\n\n"
            "【候选文档】\n" + "\n".join(lines) + "\n\n"
            f"【题目】{question.get('question','')}\n【选项】\n{opt_block}\n\n"
            '只输出一行 JSON：{"docs":["doc_id1","doc_id2"]}'
        )
        try:
            out = self.llm.chat(prompt, max_tokens=_DOC_SELECT_MAX_TOKENS,
                                meta={**meta, "doc_select": True})
        except Exception as e:
            log.warning(f"[{meta.get('qid')}] doc_select 调用失败: {e}")
            return []
        m = re.search(r'"docs"\s*:\s*\[([^\]]*)\]', out or "")
        if not m:
            return []
        picked = re.findall(r'"([^"]+)"', m.group(1))
        cand_set = set(candidates)
        return [d for d in picked if d in cand_set][:_DOC_SELECT_MAX_PICK]

    # ---- 文档对照表（多文档题序数指代）----
    def _doc_roster(self, ids: Optional[List[str]], domain) -> Tuple[str, Dict[str, int]]:
        """给定 ≥2 个 doc_ids（oracle=题目提供；blind=别名按题面首现排序）时，
        生成"第 N 份文档 = doc_id：身份摘要"对照表。
        修复 fc 域"第一份/第二份文档"类选项因序数指代不明导致的张冠李戴
        （fc_a_001/002/004/019/020 五题在 v6 全错，机制即此）。"""
        ids = ids or []
        if len(ids) < 2:
            return "", {}
        doc_ord: Dict[str, int] = {}
        lines: List[str] = []
        for i, d in enumerate(ids, 1):
            doc_ord[d] = i
            heads = self.retriever.doc_head(domain, d, n=1)
            ident = ((heads[0].get("text") or "") if heads else "")
            ident = ident[:_ROSTER_HEAD_CHARS].replace("\n", " ").strip()
            lines.append(f"第{i}份文档 = {d}：{ident}")
        roster = (
            "【文档对照表】本题共 %d 份文档；题目与选项中\"第一份文档/第二份文档\"等序数按下表对应，"
            "判断前先确认证据属于哪份文档：\n%s" % (len(ids), "\n".join(lines))
        )
        return roster, doc_ord

    # ---- 首答 JSON 里的逐项分析解析 ----
    @staticmethod
    def _parse_analysis(raw: str) -> Dict[str, str]:
        """从首答 raw 中抽 analysis 字典（A/B/C/D → 分析文本）。解析失败返回 {}。"""
        s = (raw or "").strip()
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
        try:
            obj = json.loads(s)
            if isinstance(obj, dict) and isinstance(obj.get("analysis"), dict):
                return {k: str(v) for k, v in obj["analysis"].items() if k in "ABCD"}
        except Exception:
            pass
        # 兜底：正则逐项抓 "A":"..."（排除 answer 字段）
        out: Dict[str, str] = {}
        for m in re.finditer(r'"([A-D])"\s*:\s*"((?:[^"\\]|\\.)*)"', s):
            out[m.group(1)] = m.group(2)
        return out

    # ---- 按指纹数字全域定位 chunk（复核补检回退用）----
    def _find_number_chunks(
        self,
        nums: List[str],
        domain: Optional[str],
        exclude_ids: set,
        max_chunks: int = 2,
        prefer_doc_ids: Optional[List[str]] = None,
        opt_text: str = "",
    ) -> List[Dict[str, Any]]:
        """在整个域内找含指纹数字的 chunk（0 token 精确匹配）。
        含数字最多的 chunk 优先；排除已展示的。"""
        if not domain:
            return []
        scored: List[Tuple[int, Dict[str, Any]]] = []
        try:
            idx = self.retriever._load(domain)
        except FileNotFoundError:
            return []
        prefer = set(prefer_doc_ids or [])
        # 选项关键词（≥3字的词，如"资产负债率"）：数字必须与语义上下文共现才是真证据
        # （text01 供应商占比表恰好同含 63.x%/66.x%，靠关键词把 text13 资产负债率块顶上来）
        import jieba as _jieba
        _KW_STOP = {"文档", "第一份", "第二份", "第三份", "第四份", "提供了", "具体的"}
        kws = ([w for w in _jieba.lcut(opt_text)
                if len(w) >= 3 and not re.search(r"\d", w) and w not in _KW_STOP]
               if opt_text else [])
        for c in idx["chunks"]:
            if c["chunk_id"] in exclude_ids:
                continue
            raw_t = c.get("text") or ""
            t = _num_norm(raw_t)
            cnt = _nums_in(nums, t)
            if cnt > 0:
                kw_hits = sum(1 for w in kws if w in raw_t)
                scored.append((cnt + 2 * kw_hits, c))
        # 题目 doc_ids 内的块优先，其内按 指纹命中+关键词共现 排序
        scored.sort(key=lambda x: (-(x[1]["doc_id"] in prefer), -x[0]))
        out = []
        for cnt, c in scored[:max_chunks]:
            out.append({
                "chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "domain": domain,
                "title": c.get("title", ""), "page": c.get("page", 0),
                "section": c.get("section", ""), "text": c.get("text") or "",
                "score": 0.0,
            })
        return out

    # ---- 多选证据缺口复核 ----
    def _recheck_gap_options(
        self,
        question: Dict[str, Any],
        search_domain: Optional[str],
        doc_ids: Optional[List[str]],
        raw: str,
        answer: str,
        shown_hits: List[Dict[str, Any]],
        meta: Dict[str, Any],
    ) -> Tuple[str, str]:
        """对首答中"因证据缺失而未选"的选项做定向补检 + 独立小验证。
        仅当补充证据被验证调用明确支持（support=true 且给出引用）才补入该字母。
        返回 (可能更新的 answer, 复核日志)。"""
        opts = question.get("options") or {}
        analysis = self._parse_analysis(raw)
        if not analysis:
            return answer, ""
        shown_ids = {h["chunk_id"] for h in shown_hits}
        shown_norm = _num_norm(" ".join(h.get("text") or "" for h in shown_hits))
        chosen = set(answer)
        logs: List[str] = []
        fired = 0
        # 先收集全部触发候选，按信号强度排序（指纹数字多者优先——fc_a_004 曾因
        # B/C 先占满名额导致真正缺证据的 D 没轮上），再按 cap 复核
        cands: List[Tuple[int, str, List[str]]] = []
        for k in ("A", "B", "C", "D"):
            if k in chosen or k not in opts:
                continue
            note = analysis.get(k, "")
            # 触发条件①：模型措辞承认证据缺口（依赖措辞，见 _GAP_RE）
            wording_gap = bool(note and _GAP_RE.search(note))
            # 触发条件②（机械，不依赖措辞）：选项含指纹数字（百分比/亿万金额/小数/代码）
            # 但这些数字一个都没出现在已展示证据里 → 证据必然缺失
            opt_nums = _opt_fingerprints(opts[k])
            number_gap = bool(opt_nums) and _nums_in(opt_nums, shown_norm) == 0
            if not (wording_gap or number_gap):
                continue
            cands.append((len(opt_nums) if number_gap else 0, k, opt_nums))
        cands.sort(key=lambda x: (-x[0], x[1]))
        for _sig, k, opt_nums in cands:
            if fired >= _RECHECK_MAX_PER_Q:
                break
            # 选项指认"第N份文档"时，补检证据硬性限定在该文档内
            # （fc_a_004-B 曾被 doc_ids 外的 text14"注册金额180亿"跨文档误支持）
            restrict_ids = doc_ids
            m_ord = re.search(r"第([一二三四1-4])份文档", opts[k])
            if m_ord and doc_ids:
                oi = "一二三四".find(m_ord.group(1))
                if oi < 0:
                    oi = int(m_ord.group(1)) - 1
                if 0 <= oi < len(doc_ids):
                    restrict_ids = [doc_ids[oi]]
            # 定向补检：只要"没给模型看过"的新 chunk；oracle doc_ids 内找不到时，
            # 数字缺口题扩到全域按指纹数字精确定位（doc_ids 是提示不是限制——
            # res_a_002-D 的信创数据实际在 doc_ids 外的文档里）
            q_text = question.get("question", "")
            fresh = self.retriever.search(
                f"{q_text} {opts[k]}", domain=search_domain, doc_ids=restrict_ids,
                top_k=_RECHECK_TOP_K)
            new_hits = [h for h in fresh if h["chunk_id"] not in shown_ids]
            if opt_nums and not any(
                    _nums_in(opt_nums, _num_norm(h.get("text") or "")) > 0
                    for h in new_hits):
                extra = self._find_number_chunks(
                    opt_nums, search_domain, shown_ids, prefer_doc_ids=restrict_ids,
                    opt_text=opts[k])
                if restrict_ids is not doc_ids:  # 指认特定文档 → 硬过滤
                    extra = [c for c in extra if c["doc_id"] in set(restrict_ids)]
                seen_new = {h["chunk_id"] for h in extra}
                new_hits = (extra + [h for h in new_hits if h["chunk_id"] not in seen_new]
                            )[:_RECHECK_TOP_K]
            if not new_hits:
                continue
            fired += 1
            roster, doc_ord = self._doc_roster(question, search_domain)
            ev_lines = []
            for i, h in enumerate(new_hits, 1):
                ord_part = f"【第{doc_ord[h['doc_id']]}份文档】" if h.get("doc_id") in doc_ord else ""
                ev_lines.append(
                    f"[{i}] {ord_part}{h['doc_id']} {h.get('section','')}\n"
                    + (h.get("text") or "")[:_RECHECK_CHUNK_CHARS])
            verify_prompt = (
                "你是金融文档核验员。此前因证据不足未选下列选项，现补充了新证据片段。\n"
                + (roster + "\n" if roster else "")
                + f"【题目】{q_text}\n"
                f"【待复核选项{k}】{opts[k]}\n"
                "【补充证据】\n" + "\n\n".join(ev_lines) + "\n\n"
                "判定标准（从严）：\n"
                "1. 证据必须直接、明确支持选项的具体断言（数字、日期、主体、口径、年份相符），"
                "仅话题相关或间接沾边一律不算支持；分部/板块数据不能支持全公司口径的断言。\n"
                "2. 每条证据开头已标注所属文档；若选项断言的是某一份文档的内容，"
                "只有来自该文档的证据才能支持它，来自其他文档的相同内容不算。\n"
                "3. 有任何不确定 → support 取 false。\n"
                '只输出一行 JSON：{"support":true或false,"quote":"<成立时引用证据原文关键句，否则留空>"}'
            )
            try:
                out = self.llm.chat(verify_prompt, max_tokens=150,
                                    meta={**meta, "recheck": k})
            except Exception as e:  # 复核失败不影响主答案
                log.warning(f"[{meta.get('qid')}] recheck {k} 调用失败: {e}")
                continue
            m_sup = re.search(r'"support"\s*:\s*(true|false)', out or "", re.I)
            m_quote = re.search(r'"quote"\s*:\s*"([^"]{10,})"', out or "")
            # 机械门槛：选项含指纹数字时，支持引文必须含其中至少一个
            # （fin_a_017 曾用"新签4.5万亿"的引文去支持"新签2.19万亿"的选项）
            if (m_quote and opt_nums
                    and _nums_in(opt_nums, _num_norm(m_quote.group(1)), loose=True) == 0):
                logs.append(f"[recheck {k}→引文无选项数字，拒绝] {(out or '').strip()[:100]}")
                continue
            if m_sup and m_sup.group(1).lower() == "true" and m_quote:
                chosen.add(k)
                logs.append(f"[recheck {k}→选入] {out.strip()[:200]}")
                # 把补检证据并入 evidence 记录
                for h in new_hits:
                    shown_hits.append({**h, "section": f"【复核{k}】" + (h.get("section") or "")})
                    shown_ids.add(h["chunk_id"])
            else:
                logs.append(f"[recheck {k}→不改] {(out or '').strip()[:120]}")
        return "".join(sorted(chosen)), "\n".join(logs)

    # ---- 主流程 ----
    def solve(self, question: Dict[str, Any], mode: str = "oracle") -> SolverResult:
        qid = question["qid"]
        domain = question.get("domain", "")
        fmt = (question.get("answer_format") or "mcq").lower()

        # 1) 检索（按 mode 选 doc_ids/domain）
        # blind（domain/global）模式：先做文档定位（别名强命中 + BM25 补足），
        # 再把候选 doc_ids 喂给与 oracle 相同的整条链路（对照表/钉入/复核）。
        # roster/pins 只用别名命中（高精度、含题面首现顺序），BM25 补充仅参与过滤。
        if mode == "oracle":
            doc_ids = question.get("doc_ids") or None
            search_domain: Optional[str] = domain
            trusted_ids: Optional[List[str]] = list(doc_ids) if doc_ids else None
        elif mode in ("domain", "global"):
            search_domain = domain if mode == "domain" else None
            query = self.build_query(question)
            alias_hits = [d for d, _ in self.retriever.alias_candidates(query, search_domain)]
            resolved = self.retriever.resolve_doc_ids(query, search_domain)
            # Qwen 文档选择：候选(≈12)身份摘要 → 选出题目真正涉及的 1-4 份，
            # 与别名命中取并集（选择漏了点名文档时别名兜底）
            selected: List[str] = []
            if (_DOC_SELECT_ENABLED and self.llm is not None and resolved
                    and domain not in _DOC_SELECT_SKIP_DOMAINS):
                selected = self._select_docs_llm(
                    question, resolved, search_domain,
                    {"qid": qid, "fmt": fmt, "domain": domain, "mode": mode})
            if selected:
                # 合并顺序：别名/题面编号命中在前（自带题面首现顺序，序数指代以此为准；
                # fc_a_019 曾因选择结果排前导致 roster 序数错位），选择结果去重后补充在后
                if alias_hits:
                    merged = alias_hits + [d for d in selected if d not in alias_hits]
                else:
                    merged = selected  # 题面不点名时用选择顺序（prompt 已要求按题意排序）
                doc_ids = merged
                trusted_ids = merged
            else:
                doc_ids = resolved or None
                trusted_ids = alias_hits or None
        else:
            raise ValueError(f"未知 mode: {mode}")
        hits, guards = self._option_augmented_retrieve(
            question, domain=search_domain, doc_ids=doc_ids)

        if not hits:
            log.warning(f"[{qid}] 检索为空")

        # 1.4) 文档身份头注入（可开关，见 _INJECT_DOC_HEADERS）：把每份相关文档的封面/首页
        #      置顶进证据，修跨文档"张冠李戴"。注意：可能挤占注意力致多选漏选，需 A/B 验证。
        if _INJECT_DOC_HEADERS:
            hits = self._inject_doc_headers(hits, trusted_ids, domain)

        # 1.45) 财报关键指标钉入（主要会计数据表必进；研发/分红按题目触发）
        #        blind 模式只钉别名命中的文档（trusted_ids），不给 BM25 补充文档钉表
        hits = self._pin_fin_key_chunks(hits, question, trusted_ids, domain)
        # 1.46) 保险条款钉入（犹豫期/免赔额按题目触发）
        hits = self._pin_ins_clause_chunks(hits, question, trusted_ids, domain)

        # 1.5) 按文档分配证据字符预算（多文档题每篇都有料，学自队友；
        #      fin 域加预算给钉入表块腾空间，避免挤掉检索证据）
        budget = config.EVIDENCE_TOTAL_CHAR_BUDGET
        if domain == _FIN_DOMAIN:
            budget += _FIN_EXTRA_BUDGET
        hits = _budget_by_doc(hits, budget)

        # 1.6) 选项保底证据：每个选项的 top-1 命中若被预算挤掉/未入池，截断后追加。
        #      只追加不挤占 → 保证逐项判断时"每个选项都有料"，治多选漏选的证据缺口。
        kept_ids = {h["chunk_id"] for h in hits}
        for k in sorted(guards):
            g = guards[k]
            if g["chunk_id"] in kept_ids:
                continue
            hits.append({
                **g,
                "text": (g.get("text") or "")[:_GUARD_CHARS],
                "section": f"【选项{k}相关】" + (g.get("section") or ""),
            })
            kept_ids.add(g["chunk_id"])

        # 1.7) 多文档题：文档对照表 + 证据按"第 N 份文档"标注
        #      oracle 用题目 doc_ids 顺序；blind 用别名题面首现顺序（≥2 个才建）
        roster, doc_ord = self._doc_roster(trusted_ids, domain)

        # 2) 构造 prompt（含 5 个域 system prompt + JSON CoT 模板）
        prompt = build_prompt(question, hits, max_chunk_chars=self.max_chunk_chars,
                              doc_roster=roster, doc_ord=doc_ord)

        # 3) 调 LLM（逐项分析需要更多 completion token；tf 题简单给少些）
        #    题型路由：mcq/tf 走思考模型（需给足思考 token），multi 走主模型
        meta = {"qid": qid, "fmt": fmt, "domain": domain, "mode": mode}
        use_llm = self.llm
        max_tok = 300 if fmt == "tf" else config.MAX_COMPLETION_TOKENS
        if self.reasoner is not None and fmt in ("mcq", "tf"):
            use_llm = self.reasoner
            max_tok = config.REASONER_MAX_TOKENS
        raw = use_llm.chat(prompt, max_tokens=max_tok, meta=meta)

        # 4) 答案抽取；抽不到则强制重答一次（学自队友 retry）
        answer = extract_or_none(raw, fmt)
        if answer is None:
            retry_prompt = (
                prompt
                + "\n\n上面未给出可解析的答案。请只输出一行最终答案，"
                  '格式严格为 JSON：{"answer":"<字母，多选按字母升序如 AC>"}'
            )
            raw_retry = self.llm.chat(retry_prompt, max_tokens=40, meta={**meta, "retry": True})
            answer = extract_or_none(raw_retry, fmt)
            if answer is not None:
                raw = raw + "\n[retry] " + raw_retry
        if answer is None:
            answer = normalize(raw, fmt)  # 最终合法兜底

        # 5) 多选"证据缺口复核"：首答里明说"证据未提及/未找到"的未选选项，
        #    很可能是证据没检索到而非选项为假 → 定向补检 + 小额验证，支持则补入。
        if _RECHECK_ENABLED and fmt == "multi":
            new_answer, recheck_log = self._recheck_gap_options(
                question, search_domain, doc_ids, raw, answer, hits, meta)
            if recheck_log:  # 无论是否改答案都记录，便于审计复核行为
                raw = raw + "\n" + recheck_log
            answer = new_answer

        return SolverResult(
            qid=qid,
            answer=answer,
            raw=raw,
            evidence=[
                {
                    "doc_id": h["doc_id"], "chunk_id": h["chunk_id"],
                    "page": h.get("page", 0), "section": h.get("section", ""),
                    "score": h.get("score", 0.0),
                }
                for h in hits
            ],
            mode=mode,
            domain=domain,
            fmt=fmt,
        )


# 命令行单题调试
if __name__ == "__main__":
    import argparse, json, sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ap = argparse.ArgumentParser()
    ap.add_argument("--question_file", required=True)
    ap.add_argument("--qid", default=None)
    ap.add_argument("--mode", default="oracle", choices=["oracle", "domain", "global"])
    ap.add_argument("--index_dir", default="data/index")
    args = ap.parse_args()

    text = open(args.question_file, encoding="utf-8").read()
    try:
        qs = json.loads(text)
        if isinstance(qs, dict):
            qs = [qs]
    except Exception:
        qs = [json.loads(line) for line in text.splitlines() if line.strip()]

    if args.qid:
        qs = [q for q in qs if q.get("qid") == args.qid]
        if not qs:
            print(f"找不到 qid={args.qid}", file=sys.stderr)
            sys.exit(2)

    q = qs[0]
    retriever = Retriever(args.index_dir)
    solver = Solver(retriever)
    res = solver.solve(q, mode=args.mode)
    print("=== Question ===")
    print(q.get("question"))
    print("Options:", q.get("options"))
    print("fmt:", q.get("answer_format"))
    print("=== Evidence (top) ===")
    for i, e in enumerate(res.evidence[:8], 1):
        print(f"  [{i}] {e['doc_id']} {e['section'][:30]} p{e['page']} score={e['score']:.2f}")
    print("=== Raw ===")
    print(res.raw)
    print("=== Answer ===", res.answer)
    from agent.llm import get_token_stats
    print("Token:", get_token_stats().to_dict())
