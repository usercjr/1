# AFAC2026 赛题四：方案设计文档

> 金融长文本 Agent · 动态记忆压缩与高效问答
> 目标：在 5M token 预算内做到 ≥50% 准确率（baseline 15%）
> 约束：仅 Qwen 系列 API（百炼/魔搭），禁止 embedding 模型，离线解析不计 token

---

## 1. 评分公式与目标拆解

```
TokenScore = max(0, (5,000,000 - TotalTokens) / 5,000,000)
FinalScore = 100 * Accuracy * (0.7 + 0.3 * TokenScore)
```

| 情形 | Accuracy | TotalTokens | TokenScore | FinalScore |
|---|---|---|---|---|
| Baseline | 15% | 7.5M | 0 | 10.5 |
| 我们的 P0 | 45% | 3.5M | 0.30 | 35.55 |
| 我们的 P1 | 60% | 2.5M | 0.50 | 51.0 |
| 理想 | 75% | 1.5M | 0.70 | 68.25 |

**结论**：准确率权重远大于 token 效率（70% vs 30%），但 baseline 已经用掉 7.5M token，所以**先压成本再提精度**会让每次实验更便宜。优先级：能跑通 > Token 控住 < 5M > 准确率冲刺。

---

## 2. 整体架构

```
┌──────────────────── 离线阶段（不计 token） ────────────────────┐
│ 1. 文档解析    PDF/HTML/TXT → 结构化 JSON（含章节、条款、表格）│
│ 2. 切片        按章节/条款切成 200–500 字 chunk                │
│ 3. 倒排索引    jieba 分词 → BM25                              │
│ 4. 元数据索引  条款号、金额、百分比、日期等正则抽取             │
└────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────── 在线推理（计 token） ─────────────────────┐
│ Step 1  题目解析   规则抽取关键词/数值/实体（本地，0 token）   │
│ Step 2  文档定位   A 榜：直接用 doc_ids                         │
│                    B 榜：BM25 召回 + 标题/章节正则过滤          │
│ Step 3  证据检索   BM25 在限定 doc 内做 chunk 检索 top-K        │
│ Step 4  证据精排   规则打分（关键词命中、章节优先级）          │
│ Step 5  Agent 推理 把题目 + top-3~5 chunk → Qwen-Plus 输出     │
│ Step 6  答案标准化 正则抽字母、去重、排序                       │
└────────────────────────────────────────────────────────────┘
```

**关键设计原则**

- **离线做尽一切能做的事**：解析、切片、索引、关键词词典构造，全部不计 token，越彻底越省钱。
- **检索阶段零 LLM 调用**：靠 BM25 + 规则就到 top-5，禁用 LLM rerank（除非最后 polish 阶段加一档配置）。
- **每题只调一次 LLM**：拒绝多轮自检/反思（token 翻倍），改用规则后处理 + 选项级证据打分。
- **统一 prompt 模板**：按 answer_format（mcq/multi/tf）分发，强约束输出仅一字母串。

---

## 3. 数据情况确认

实际数据集（你本地已下载）：

| 域 | 文档数 | 格式 | 备注 |
|---|---|---|---|
| insurance | 16 | PDF | 保险条款，表格多，条款编号是关键 |
| regulatory | 6 (txt) + html + attachments | TXT/HTML | TXT 最干净，直接用 |
| financial_contracts | 14 | PDF | 含表格、条款 |
| financial_reports | 10 | PDF | 年报，体量最大（几十~上百页） |
| research | 20 | PDF | 研报，图表多 |

> 注：A 榜公开题目目前只放出 regulatory 和 research 两个域的 json，其余三个域的题目应在评测开放后给出（或需要根据已公开题目自行构造验证集）。

---

## 4. 模块设计

### 4.1 文档解析（offline，不计 token）

| 文件类型 | 推荐库 | 备注 |
|---|---|---|
| PDF | `pdfplumber` 主，`pymupdf` 备 | pdfplumber 表格抽取好，pymupdf 速度快 |
| HTML | `BeautifulSoup4` | 去掉导航、广告 |
| TXT | 直接读 | 注意编码（GB18030/UTF-8） |

**输出统一 schema**（每文档一个 JSON）：

```json
{
  "doc_id": "strict_v3_008",
  "title": "金融机构客户受益所有人识别管理办法",
  "domain": "regulatory",
  "sections": [
    {
      "section_id": "ch1_art1",
      "heading": "第一章 总则 第一条",
      "page": 1,
      "text": "为加强金融机构反洗钱...",
      "tables": []
    }
  ],
  "meta": {
    "publish_date": "2025-...",
    "effective_date": "2025-...",
    "issuer": "中国人民银行"
  }
}
```

### 4.2 切片策略（offline）

- **法规/合同**：按"第X条/第X章/第X节"切，每条款一个 chunk。条款号是检索 anchor。
- **年报**：按一级标题切，过长再按段落（500 字一窗口，100 字 overlap）。
- **研报**：按章节标题切，保留图表 caption。
- **保险条款**：按"责任条款/释义/除外责任/退保"等小节切。

每个 chunk 限制在 300–600 字之间，存 chunk_id、doc_id、section_path、text。

### 4.3 检索（online，0 token）

**BM25 实现**：`rank_bm25` 库 + `jieba` 分词。词典里手动加入：
- 金融术语：受益所有人、保单现金价值、特别决议、可疑交易报告...
- 域名实体：比亚迪、宁德时代、招商银行...
- 条款号 token：第四十七条、第八十二条...

**查询构造**：
- 题干 + 所有选项一起作为 query（不是只用题干）→ 显著提升命中
- 选项里的具体数值/条款号 → 额外做正则定位

**召回 K 值**：
- A 榜：在指定 doc_ids 里召回 top-5 chunk
- B 榜：先文档级召回 top-3 doc，再 chunk 级召回 top-5

### 4.4 Prompt 模板

**单选题 (mcq)**：

```
你是金融文档分析专家。基于以下证据片段回答问题，只输出一个大写字母。

【证据片段】
[1] <doc_id> <section>: <text>
[2] ...
[3] ...

【问题】<question>

【选项】
A. <opt_A>
B. <opt_B>
C. <opt_C>
D. <opt_D>

回答规则：只输出 A/B/C/D 中的一个字母，不要解释。

答案：
```

**多选题 (multi)**：

```
... 同上证据 ...

【问题】<question>
【选项】A/B/C/D...

回答规则：选出所有正确选项，按字母顺序输出，无空格无逗号。例如 ABC。
若全部都对输出 ABCD，若都错输出 X（X 不会被算对，但避免空答）。

答案：
```

> 多选题是丢分大头（漏选/多选都算错）。两条增强策略：
> 1. 逐选项独立判断：让模型对每个选项输出 T/F，再拼出最终答案。但这会让 completion token 翻倍。
> 2. 给 few-shot 例子，强化"严格根据证据，没说就不选"的倾向。

**判断题 (tf)**：

```
... 同上 ...

【问题】<question>
【选项】
A. <opt_A>  （通常是"正确"）
B. <opt_B>  （通常是"错误"）

回答规则：只输出 A 或 B。

答案：
```

### 4.5 答案后处理

```python
def normalize(answer: str, fmt: str) -> str:
    letters = re.findall(r'[A-D]', answer.upper())
    if fmt == 'mcq' or fmt == 'tf':
        return letters[0] if letters else 'A'  # 兜底
    elif fmt == 'multi':
        return ''.join(sorted(set(letters))) or 'A'
```

---

## 5. 项目目录结构

```
ali_competition/
├── data/
│   ├── raw/                  # 原始 PDF/HTML/TXT（你已下载）
│   ├── parsed/               # 离线解析后的 JSON
│   └── chunks/               # 切片结果 + BM25 索引
├── agent/
│   ├── __init__.py
│   ├── config.py             # API key、模型名、超参
│   ├── llm.py                # Qwen API 封装（含重试、token 累计）
│   ├── retriever.py          # BM25 + 规则检索
│   ├── prompts.py            # 三类题型的模板
│   ├── solver.py             # 单题求解主流程
│   └── postproc.py           # 答案标准化
├── scripts/
│   ├── 01_parse_docs.py      # 离线解析
│   ├── 02_build_index.py     # 切片 + BM25
│   ├── 03_run_eval.py        # 在线批量答题
│   └── 04_make_submission.py # 生成 answer.csv
├── outputs/
│   ├── answer.csv            # 提交文件
│   ├── evidence.json         # 证据日志
│   └── logs/                 # 每题的 prompt/response/token
├── requirements.txt
└── README.md
```

---

## 6. Token 预算（5M 内）

按 200 题摊销：每题 **平均 25,000 token** 预算。

**单题分配建议（mcq/multi/tf 通用）**：

| 项 | tokens | 说明 |
|---|---|---|
| 系统/任务说明 | ~200 | 模板固定开销 |
| 题干 + 选项 | ~500 | 含 4 个 option |
| 5 个证据片段 | ~3500 | 每片 ~700 字 ≈ ~700 token |
| 完成 (completion) | ~10 | 只输出字母 |
| **小计** | **~4200** | 留 5–6 倍 buffer |

**1M token 试跑（5,000/题）能跑全部 200 题**。剩 4M 留给：
- B 榜需要先做文档级 LLM 筛选时的额外调用
- 多选题做"逐选项判断"加 2× completion 时的开销
- 失败重试

**省 token 的几条狠招**：
1. **chunk 去重**：同 doc 同章节命中两次只保留一次。
2. **强制截断**：单 chunk 超过 800 字强行掐到 800。
3. **不输出 reasoning**：completion 上限设 2 token（"AB"够用）。
4. **DashScope 缓存**：相同 prompt 重试时利用平台缓存，prompt token 半价。
5. **小模型分流**：tf 题（最简单）用 Qwen-Turbo，mcq/multi 用 Qwen-Plus。
6. **失败重试不重新解析**：缓存 retrieval 结果到磁盘。

**Token 监控**：每次调用累加到 `outputs/token_stats.json`，跑到 70% 时报警。

---

## 7. 风险点 & 应对

| 风险 | 影响 | 应对 |
|---|---|---|
| PDF 解析丢失表格 | 财报、合同丢关键数字 | pdfplumber + camelot 双管齐下，表格转 markdown |
| 多选题准确率低 | 占比 1/3，全错丢 33 分 | 逐选项独立判断 + few-shot |
| B 榜文档检索失败 | 召回不到正确 doc 就完蛋 | 文档级先用标题+摘要 BM25，再用条款号/公司名规则强匹配 |
| 题干极长（嵌套场景） | prompt token 暴涨 | 题干截断 + 关键信息抽取 |
| API 限流 / 超时 | 跑批中断 | 指数退避 + checkpoint 续跑 |
| jieba 词典覆盖不全 | BM25 召回差 | 离线从所有文档抽 n-gram 高频词补充词典 |
| qwen3.6-plus 模型名 | 题面写的是 "Qwen3.6-plus"，但目前公开版本是 qwen-plus / qwen2.5-... | 评测前用官方公告/赛题群确认确切 model id |

---

## 8. 时间表（剩余约 41 天）

| 阶段 | 时长 | 目标 | 产出 |
|---|---|---|---|
| W1 (D1–D3) | 3 天 | 文档解析 + 索引就绪 | data/parsed、data/chunks |
| W1 (D4–D5) | 2 天 | 最小 baseline 跑通 5 题 | agent 骨架 + 第一次提交 |
| W2 (D6–D10) | 5 天 | A 榜全量 100 题首版 | answer.csv v1，目标 ≥ 35% |
| W2 (D11–D14) | 4 天 | 多选题专项优化、证据 rerank | v2，目标 ≥ 50% |
| W3 (D15–D21) | 7 天 | 难题 case study、prompt 调优 | v3，目标 ≥ 60% |
| W4 (D22–D28) | 7 天 | A 榜冲分 + B 榜检索预演 | A 榜定稿 |
| W5–W6 (D29–D41) | 13 天 | B 榜评测 + 报告 + 代码整理 | 提交包 submission.zip |

> 关键里程碑：D5 之前必须有一次合法提交（哪怕 20% 准确率），确认提交流程没坑。

---

## 9. 立刻可以开始的下一步

1. **环境准备**：`pip install pdfplumber pymupdf beautifulsoup4 jieba rank_bm25 dashscope pandas tqdm`
2. **scripts/01_parse_docs.py**：先把 5 个域的所有文档解析成 JSON。
3. **scripts/02_build_index.py**：jieba + BM25 索引。
4. **agent/llm.py**：封装 Qwen 调用，统计 token。
5. **scripts/03_run_eval.py**：先跑 5 题验证端到端，再放量。

下一轮我可以直接：
- (a) 写文档解析脚本 `01_parse_docs.py`（recommended，离线不烧钱）
- (b) 写 agent 骨架 `llm.py + solver.py`
- (c) 先做 retrieval 评测脚本（验证 BM25 召回是否够用）

告诉我从哪一项开始。
