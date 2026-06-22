# AFAC2026 赛题四 · 金融长文本 Agent

> 阿里云 AFAC2026 挑战组赛题四：金融长文本的动态记忆压缩与高效问答。
> 5 个领域 × 20 题 = 100 道 A 榜题（单选 mcq / 多选 multi / 判断 tf）。

![python](https://img.shields.io/badge/python-3.10%2B-blue)
![model](https://img.shields.io/badge/LLM-Qwen%20系列%20(DashScope)-orange)
![retrieval](https://img.shields.io/badge/retrieval-BM25%20%2B%20jieba-green)
![no-embedding](https://img.shields.io/badge/embedding-禁用-red)

---

## 赛题约束

- **仅可调用 Qwen 系列 API**（阿里云百炼 / 魔搭），不得用其他开闭源模型替代推理。
- **正式检索与推理禁止使用 embedding / 向量模型**；预处理（PDF 解析、版面还原）允许非 Qwen 工具。
- **Token 预算 5,000,000**，离线解析与索引不计入。评分：
  ```
  TokenScore = max(0, (5_000_000 - TotalTokens) / 5_000_000)
  FinalScore = 100 × Accuracy × (0.7 + 0.3 × TokenScore)
  ```
  准确率权重 70%，token 效率权重 30%。

## 设计要点

- **零向量检索**：jieba 分词（+50 余条金融自定义词典）+ BM25 倒排，条款号命中 ×2.5 加权，多文档题 per-doc 公平采样配额。完全合规且 0 token。
- **选项增强检索**：每个选项单独跑一轮 BM25（top-2，权重 0.7）融入主检索池，提升多选 / 跨文档对比题的证据覆盖。
- **5 个领域专用 Prompt**：保险 / 法规 / 合同 / 财报 / 研报各自的术语、计算公式与陷阱（如法规题的"应当/必须＞可以＞不得"规范词强弱、阈值显式比对）。
- **JSON 结构化 CoT**：强制 `{"analysis":{"A":"…"},"answer":"AC"}` 逐选项分析后给答案，不漏判、可追溯、便于后处理。
- **Token 极致优化**：离线完成全部解析 / 切片 / 索引；100 题约 53 万 token，仅占预算 ~10.6%。

## 系统架构

```
PDF / HTML / TXT
      │  scripts/01_parse_docs.py   (pdfplumber + pymupdf + bs4)
      ▼
data/parsed/{domain}/*.json
      │  scripts/02_build_index.py  (jieba + BM25 倒排，不用 embedding)
      ▼
data/index/{domain}.pkl
      │
      ├── agent/retriever.py   BM25 + 条款号 boost + per-doc 公平采样 + 选项增强检索
      ├── agent/prompts.py     5 域 system prompt + JSON CoT 模板
      ├── agent/llm.py         Qwen 调用（OpenAI 兼容端点 + token 统计 + 重试）
      └── agent/postproc.py    JSON 抽取 + 字母标准化兜底
      ▼
outputs/answer.csv  +  outputs/evidence.json
```

## 项目结构

```
ali_competition/
├── agent/                  # Agent 核心
│   ├── config.py           #   配置（API key / 模型 / 超参）
│   ├── llm.py              #   Qwen 封装（token 统计 + 退避重试）
│   ├── retriever.py        #   BM25 检索（条款 boost / per-doc cap）
│   ├── prompts.py          #   5 域 prompt + JSON CoT
│   ├── postproc.py         #   答案标准化
│   └── solver.py           #   单题主流程（选项增强检索）
├── scripts/
│   ├── 01_parse_docs.py    #   PDF/HTML/TXT → JSON
│   ├── 02_build_index.py   #   切片 + BM25 索引
│   ├── 03_test_retrieval.py#   检索召回评测（0 token，无需 API key）
│   ├── 04_run_eval.py      #   批量答题 → answer.csv（检查点续跑）
│   └── 05_find_suspect.py  #   事后找可疑题（0 token）
├── prompts/                # 5 个域专用 system prompt
├── plan.md                 # 完整方案设计 + 版本演进记录
├── requirements.txt
└── README.md
```

> 数据目录（`data/`、`public_dataset_upload/`、`outputs/`）、`*.pkl`、`*.jsonl`、`.env` 均已在 `.gitignore` 中排除，不入库。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API key

```bash
copy .env.example .env
# 编辑 .env，填入 DASHSCOPE_API_KEY 和 QWEN_MODEL
```

### 3. 解析 + 索引（一次性，几分钟，不计 token）

```bash
python scripts/01_parse_docs.py --raw_dir public_dataset_upload/raw --out_dir data/parsed --workers 4
python scripts/02_build_index.py
```

### 4. 验证检索质量（0 token，无需 API key）

```bash
python scripts/03_test_retrieval.py
```

期望 domain mode total all_doc% ≥ 80%。

### 5. 批量答题 + 提交

```bash
python scripts/04_run_eval.py
# 产物：outputs/answer.csv 直接提交天池
```

### 6. 事后分析可疑题

```bash
python scripts/05_find_suspect.py
```

> 推荐流程：先看 `03`（文档召回够不够）→ 再 `04 --dry_run`（喂进去的 chunk 对不对）→ 满意后正式跑 `04`（烧 token）。

## 提交格式

`answer.csv` 含一行 `summary`（汇总 token）+ 每题一行：

```csv
qid,answer,prompt_tokens,completion_tokens,total_tokens
summary,,512401,16610,529011
ins_a_001,B,...,...,...
```

单选 / 判断输出单字母；多选去重并按字母升序，无分隔符（如 `AC`）。

## 版本演进

| 版本 | 检索 | Prompt | 模型 | TOP_K | 切片(MAX/MIN/OVL/CAP) | A 榜 |
|---|---|---|---|---|---|---|
| v1 | BM25 + per-doc cap | 通用·仅字母 | qwen-plus | 5 | 600/80/80/1200 | ~44.5 |
| v2 | + 选项增强检索 | 5 域 prompt + JSON CoT | qwen3.7-plus / qwen-max | 5 | 600/80/80/1200 | ~71 |
| v3 | 同上 | 同上 | qwen-max | 8 | 1000/120/120/1800 | — |

详见 [`plan.md`](plan.md) 第 10 节。

## 合规性

- ✅ 仅使用 Qwen 系列 API（qwen-plus / qwen-max / qwen3.7-plus）
- ✅ 检索零向量模型，纯统计 BM25
- ✅ 预处理仅用 pdfplumber / pymupdf / bs4（非模型工具）
- ✅ 无 Qwen 离线生成的语义摘要参与正式答题
