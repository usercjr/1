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

---

## 10. 版本演进记录（Changelog）

记录三次迭代的改动与参数。分数为 A 榜 oracle 模式。

### v1 — 初版基线（检索骨架）

- **检索**：BM25（jieba + 金融词典）+ per-doc 公平采样配额；条款号 boost。
- **Prompt**：通用模板，按 mcq/multi/tf 分发，**只要求输出一个字母**，不做推理。
- **模型**：`qwen-plus`。
- **关键参数**：
  - `TOP_K_CHUNKS = 5`，`MAX_CHUNK_CHARS_IN_PROMPT = 700`
  - `MAX_COMPLETION_TOKENS = 8`（只够吐字母，无 CoT）
  - 切片：`MAX_CHARS=600 / MIN_CHARS=80 / OVERLAP=80 / ARTICLE_HARD_CAP=1200`
- **成绩**：A 榜约 **44.5 分**（对应 git commit `v2 baseline`）。
- **瓶颈**：模型直接选字母，专业题（法规阈值、保险计算、跨文档对比）易错，多选题漏选/错选严重。

### v2 — 加入 Prompt 工程（精度主升级）

- **新增 5 个域专用 system prompt**（`prompts/{domain}.txt`）：针对保险/法规/合同/财报/研报各自的术语、计算公式与常见陷阱（如法规题的"应当/必须＞可以＞不得"规范词强弱、绝对词陷阱、阈值显式比对）。
- **JSON 结构化 CoT**：强制模型按 `{"analysis":{"A":"…","B":"…"},"answer":"AC"}` 逐选项分析后再给答案 → 不漏判、可追溯、`answer` 字段稳定便于后处理。
- **选项增强检索**：每个选项单独跑一轮 BM25（top-2，权重 0.7）融入主检索池，提升多选/对比题证据覆盖。
- **后处理升级**：三重 JSON 解析兜底 + 正则字母抽取。
- **参数变化**：`MAX_COMPLETION_TOKENS 8 → 400`（给 JSON CoT 留输出空间）。
- **模型**：升级到 `qwen3.7-plus` / `qwen-max`。
- **成绩**：A 榜约 **71 分**（对应 git commit `v3.7 baseline`）。这一版是分数的主要跃升。

### v3 — 检索参数调优（topsize + chunk）

- **检索 top size 调大**：`TOP_K_CHUNKS 5 → 8`（经 `.env` 覆盖）→ 每题喂入更多证据，提升 all-doc 召回。
- **切片粒度调大**（重建索引）：
  - `MAX_CHARS 600 → 1000`
  - `MIN_CHARS 80 → 120`
  - `OVERLAP 80 → 120`
  - `ARTICLE_HARD_CAP 1200 → 1800`
- **效果**：更长的 chunk 减少跨片段证据被切断的问题，对法规长条文 / 财报表格段落更友好。重建后总 chunk 数 17,309。
- **Token 状况**：v3（qwen-max）100 题总计 ~529K token，仅占 5M 预算 **10.6%**，TokenScore≈0.89，加权系数≈0.97 —— token 效率已几乎不拖分，后续应全力提准确率。

> ⚠️ **待修复的不一致**：当前已构建的索引（`data/index/index_summary.json`）记录的切片参数是 `1000/120/120/1800`，但源码 `scripts/02_build_index.py` 仍写着 `600/80/80/1200`。说明 v3 的切片改动**没有写回源码**（或源码被回退）。如果现在重跑 `02_build_index.py`，会生成与当前索引不一致的 chunk。**修复方式**：把 `02_build_index.py` 顶部的 4 个常量改成 `1000/120/120/1800`，使源码与已构建索引、与本 changelog 一致，保证可复现（B 榜 top-15 代码审核要求一键复现）。

### 各版本对照表

| 版本 | 检索 | Prompt | 模型 | TOP_K | completion | 切片(MAX/MIN/OVL/CAP) | A 榜分 |
|---|---|---|---|---|---|---|---|
| v1 | BM25 + per-doc cap | 通用·仅字母 | qwen-plus | 5 | 8 | 600/80/80/1200 | ~44.5 |
| v2 | + 选项增强检索 | 5 域 prompt + JSON CoT | qwen3.7-plus / qwen-max | 5 | 400 | 600/80/80/1200 | ~71 |
| v3 | 同上 | 同上 | qwen-max | 8 | 400 | 1000/120/120/1800 | — |

### 下一步候选（v4 / B 榜）

- **域路由**（B 榜必做）：关键词/条款号/实体规则预测 domain，避开全域检索噪声（global 模式 all-doc 召回仅 55%）。
- **两阶段检索**：`search_docs()` 先锁候选文档，再在候选内做 chunk 检索。
- **multi 题 self-verification**：36 道分歧题里 26 道是多选，自检性价比最高。
- **financial_contracts 跨文档对比题**：强化"逐文档分别核验"提示。

---

## 11. v4 → v6 改进记录（错误分析驱动）

### 方法论：从"猜"到"测"

前期靠"改一版、提一次榜 / 看小样本"判断好坏，结果在噪声里来回打转。两个教训定调了后续打法：

1. **模型快照会漂移**：`qwen-max` 是别名，指向的快照随时间变化。同一份代码 + 索引 + prompt，v4 当时跑 12/15、后来复跑只有 2/15 —— 多选输出方差极大（部分是别名漂移，部分是 400 token 截断放大的抖动）。
2. **噪声地板 ≈ ±5 题**：同配置连跑两遍，100 题里也差约 11 题。任何小于 ~5 题的"提升"都分不清是改动还是运气。

应对：**自建人工真值尺子 + 错误分析**。
- `outputs/analysis/ground_truth_manual.json`：人工核对 35 道题的标准答案（覆盖 reg/research/insurance/fc/fin 五域 × mcq/multi/tf 三型，高置信 24 道）。
- `scripts/score_truth.py`：0-token 本地打分器，对任意 `answer.csv` 在真值上评分。
- 判据：**改动必须有"可观察、可泛化的错误机制"支撑**（如亲眼看到跨文档张冠李戴），而不是"小样本分涨了"（那是噪声）。最终仍以榜单为准。

### 三个版本

| 版本 | 定义 | A 榜 | 35 道尺子(高置信) | token |
|---|---|---|---|---|
| **v4** | v3.7 代码 + 1000 索引 + qwen-max | **71.65** | 19/24 | 529K |
| **v5** | v4 + 学队友三件套（800 token / 每文档证据预算 / retry）+ enable_thinking 全关 | 67.89 | — | 503K |
| **v6** | v5 + 文档身份头注入 + enable_thinking 仅 qwen3 | 待提交 | **22/24** | 508K |

> v5 比 v4 低：不是 token 问题（503K 反而更少），是 `enable_thinking=False` 误伤了 qwen-max 上的推理单选题（退保计算、日期比较），多选涨、单选跌，净降。v6 修复后高置信反超 v4 +3。

### 六处代码改动（改哪里 / 为什么 / 方法 / 怎么观察到）

**① completion token 400 → 800**（`agent/config.py: MAX_COMPLETION_TOKENS`）
JSON 逐项分析常被 400 token 截断、答案字段没写完 → 多选漏选、答案不稳。翻倍留足空间。观察来源：学队友（用 900）+ 同配置方差极大定位到截断。

**② 每文档证据字符预算**（`agent/solver.py: _budget_by_doc` + `config.EVIDENCE_TOTAL_CHAR_BUDGET=9000`）
多文档对比题里长文档（财报）挤占证据，另一篇没料。每文档分 `max(1500, 9000/篇数)` 字符、每篇至少 1 chunk。观察来源：学队友 per-doc budget。

**③ MAX_CHUNK_CHARS 700 → 1000**（`agent/config.py`）
证据片段被截到 700 字丢信息，放足（配合 1000 字索引）。

**④ 解析失败 retry**（`agent/postproc.py: extract_or_none` + `solver.py` 重答）
抽不到答案时原来直接兜底"A"；改为 `extract_or_none` 返回 `None` → 强制格式重答一次 → 仍失败才兜底。观察来源：学队友 retry。

**⑤ 文档身份头注入**（`agent/solver.py: _inject_doc_headers` + `agent/retriever.py: doc_head`）
fc 跨文档比对题"张冠李戴"（把 text07 海峡股份/重组报告当成安克/募集说明书）。把每份文档封面首段（发行人/证券简称/文件类型，取前 220 字）置顶进证据。**观察来源：错误分析——给 v4 打分发现 fc 域最弱（3/6），逐题读错答，看到根因是文档身份信息没进上下文。** 修后 fc_a_011、fc_a_020 立即修对。

**⑥ enable_thinking 仅对 qwen3 生效**（`agent/llm.py`）
先前给所有模型关思考，把 qwen-max 上的推理单选题也"口算"化 → 单选回归。改为 `if "qwen3" in model` 才传 `extra_body={"enable_thinking": False}`，qwen-max 恢复默认（= v4）。**观察来源：标了单选/判断真值，发现 v5 在 mcq/tf 上比 v4 净少对 2 道（fc_a_008、ins_a_002），均为推理题。**

## 12. v7 改动（错误机制驱动，2026-07-03）

对 v6 在 63 道真值上的 19 道错题做机制归因后的三处修复（`agent/solver.py` + `agent/prompts.py`）：

**① 文档对照表 + 证据序数标注**（`solver._doc_roster` + `prompts.render_evidence(doc_ord)`）
fc 域 7 道题的选项用"第一份文档/第二份文档"指代，但 prompt 从未告诉模型 doc_ids 的顺序
→ v6 在 fc_a_001/002/004/019/020 全错（fc_a_002 的 B/D 判断与真值完全颠倒，即两份文档张冠李戴）。
修复：doc_ids ≥2 时注入"第 N 份文档 = doc_id：封面身份摘要(80字)"对照表，
且每条证据头标注【第 N 份文档】。

**② 选项保底证据**（`solver._option_augmented_retrieve` 返回 guards + solve 步骤 1.6）
多选漏选 9 题的主因是"该选项的证据根本没进上下文"（模型自己在 analysis 里写
"证据中没有找到相关信息"，如 res_a_002-D、fin_a_012-B）。财报/研报域此前完全关闭
选项级检索，加剧此问题。修复：所有域每个选项取 top-1 命中；若最终证据集中没有它，
截断到 600 字后**追加**（不挤占主检索证据，不参与排名，规避此前"选项噪声助长过选"的教训）。

**③ 多选证据缺口复核**（`solver._recheck_gap_options`）
首答 analysis 里对未选选项明说"未找到/未提及证据"时（正则 `_GAP_RE`），做定向补检
（top-3、排除已展示 chunk），用一次 ~150 token 小验证调用判断补充证据是否支持该选项；
仅当 `support:true` 且给出原文引用才补入字母。单题最多复核 2 个选项。
预计新增 token ~40K/百题，对 TokenScore 影响 <0.5 分。

另：多选 JSON 指令加一行"漏选与错选同罚，证据明确支持的必须全选入"；
数值选项"先找到数字再比对"。

### v7 验证结果（63 道真值，qwen-max，oracle）

| 版本 | 全部 | 高置信 | 备注 |
|---|---|---|---|
| v6 | 44/63 | 41/54 | |
| v7 首跑 | 49/63 | 45/54 | 63 题 359K token |
| v7 + 收紧复核 | ~51/63 | — | fc_a_016/ins_a_016 回归修复 |
| **v8 全量定稿** | **50/63** | **46/54** | 100 题 547,674 token，TokenScore 0.8905，产物 outputs_v8_full/ |

- 由错转对：fc_a_001/002/012/018/019（①文档对照表）、ins_a_001、ins_a_014。
- 首跑复核宽松引入 2 个回归（fc_a_016 误加 D、ins_a_016 误加 A，均为验证调用
  不知道引用属于哪份文档）→ 收紧：复核 prompt 带文档对照表、要求"选项断言某文档
  的内容只有该文档的证据才能支持"、从严判定、quote ≥10 字。复跑两题均修复。
- 仍错（下一步候选）：fin 数值题 fin_a_002/005/012/015/016（表格数字进上下文仍不完整）、
  ins_a_002/019/020（保险计算/条款细节）、res_a_002/004、fc_a_004。
- 已知方差点：fc_a_014 首答偶尔漏 A（多选首答方差，复核可救但非必然触发）。

## 13. v9 改动（fin 证据缺口专项，2026-07-04）

**① 财报关键指标钉入**（`solver._pin_fin_key_chunks` + `retriever.find_doc_chunks`，规则式 0 token）
fin 对比题反复需要营收/归母净利/现金流(+同比)、研发占比、分红，全在年报固定结构表块里，
但 BM25 常召不回（fin_a_002/005/012 的营收/净利证据缺失）。每份年报"主要会计数据"表
（谓词：核心科目+同比字样同现）固定钉入；题目提到研发/分红再钉对应表块（截 700 字）。
fin 域证据预算 +3000 字，钉入块只加不挤占。10 份年报谓词覆盖率验证全通过。

**② _GAP_RE 拓宽**：模型高频写"未**直接**提供/没有**直接**给出/缺乏直接**支持**/未**明确**说明"，
原正则全漏 → fin_a_002-A/fin_a_005-A/fin_a_012-B/ins_a_019-D 的缺口复核都没触发。已补变体。

**③ _OPTION_MIN_CHARS 8 → 4**：ins_a_019-D 选项只有产品名"平安富鸿金生"(6字)，
被 8 字门槛挡掉 → 该选项零专属证据。

### v9 验证

- 15 题受影响子集：v8 7/15 → v9 9/15（fin_a_005、fin_a_012 由错转对，无回归）。
- 全量 100 题：50/63（高置信 46/54）与 v8 打平——fin 的 A 选项证据都补上了，
  但方差把收益抵消（ins_a_001/002、fin_a_005-D 本次随机翻错；fc_a_014/020 随机翻对）。
  603K token，加权系数 0.9638。
- **方差是当前主要瓶颈**（±3-5 题/跑）。已评估并否决：3 次投票（token×3 → 系数-6.6%，
  需 +5.7pp 准确率才回本，投票只能换 ~2-4pp）；跨历史跑 0 成本投票（真值上 50/63 持平，
  且须诚实上报全部跑批 token）。
- 不可修清单：res_a_002-D（"金融信创2500亿"在两份 oracle 文档索引中不存在，研报图表
  解析缺口，除非对 research 域做 OCR 级重解析）；fin_a_015/016、res_a_004（真值本身存疑）。

### 提交建议

提交 `outputs_v9_full/answer.csv`（100 题，603,066 token）。v9 与 v8 真值持平，
但 fin 钉入对尺子外的 8 道 fin 题、序数对照表对尺子外的 fc_a_005/017 应有额外收益。

### v6 仍存在的漏洞（下一步）

- **fc_a_018（日期比较）**：两个文档的关键日期没同时进上下文 → 证据召回问题，非推理问题。
- **financial_reports 数值题**：年报表格解析常把数字打乱（"4.5 2.19 461.9"糊在一起），连人工核对都难；模型失分很可能是"数字没干净地进上下文"。可考虑专门的表格/关键指标抽取。
- **多选仍偶尔差一个选项**（ins_a_019 等）：证据召回或逐项判断的边界问题。
