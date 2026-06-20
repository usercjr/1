# AFAC2026 赛题四：金融长文本 Agent

> 阿里云 AFAC2026 挑战组赛题四，金融长文本动态记忆压缩与高效问答。
> 5 个域 × 20 题 = 100 道 A 榜题（mcq/multi/tf）。
> 约束：仅 Qwen 系列 API，禁止 embedding 模型；离线解析免费，在线 token 5M 预算。

## 当前成绩

| 版本 | 改动 | FinalScore | Acc | Total Tokens |
|---|---|---|---|---|
| baseline | 出题方裸跑 qwen-plus | 40.18 | 49% | 3.0M |
| v1 | BM25 + top_k=5 | 44.49 | 45% | 188K |
| v2 | top_k=8 | 44.49 | 45% | 287K |
| **v3** | **5 域 prompt + JSON CoT + 选项增强 + 条款 boost** | **55.65** | **57%** | 395K |
| v3.7 | 切 qwen3.7-plus + thinking mode | TBD | TBD | 455K |

## 系统架构

```
PDF / HTML / TXT  ──┐
                    ▼
        [01_parse_docs.py]  离线解析（pdfplumber + pymupdf + bs4）
                    ▼
        data/parsed/{domain}/*.json
                    ▼
        [02_build_index.py]  jieba + BM25 倒排（不用 embedding）
                    ▼
        data/index/{domain}.pkl
                    ▼
   ┌────────────────────────────────────────┐
   │ agent/retriever.py                     │
   │   - BM25 + 条款号 boost (×2.5)         │
   │   - per-doc 公平采样                    │
   │   - 选项增强检索（每选项 top-2，权重 0.7）│
   └────────────────────────────────────────┘
                    ▼
   ┌────────────────────────────────────────┐
   │ agent/prompts.py                       │
   │   - 5 个域专用 system prompt           │
   │   - JSON 结构化 CoT：                  │
   │     {analysis:{A:"...",B:"..."},       │
   │      answer:"AC"}                       │
   └────────────────────────────────────────┘
                    ▼
        agent/llm.py  (OpenAI 兼容模式 → DashScope)
                    ▼
        agent/postproc.py  (JSON 抽取 + 字母兜底)
                    ▼
        outputs/answer.csv
```

## 项目结构

```
ali_competition/
├── README.md
├── plan.md                    # 完整方案设计文档
├── requirements.txt
├── .env.example               # 配置模板（真 key 放 .env 不入 git）
├── .gitignore
│
├── agent/                     # Agent 核心代码
│   ├── config.py              # API key / 模型 / 超参
│   ├── llm.py                 # Qwen 调用（OpenAI 兼容客户端 + token 统计 + 重试）
│   ├── retriever.py           # BM25 检索（含条款 boost / per-doc cap）
│   ├── prompts.py             # 5 域 prompt + JSON CoT 模板
│   ├── postproc.py            # 答案标准化（JSON 抽取）
│   └── solver.py              # 单题主流程（含选项增强检索）
│
├── scripts/                   # 离线 + 评测脚本
│   ├── 01_parse_docs.py       # PDF/HTML/TXT → JSON
│   ├── 02_build_index.py      # 切片 + BM25 索引
│   ├── 03_test_retrieval.py   # 检索召回率测试（0 token）
│   ├── 04_run_eval.py         # 批量答题，生成 answer.csv
│   └── 05_find_suspect.py     # 找可疑题（0 token，事后分析）
│
└── prompts/                   # 5 个域专用 system prompt
    ├── insurance.txt
    ├── regulatory.txt
    ├── financial_contracts.txt
    ├── financial_reports.txt
    └── research.txt
```

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

### 3. 解析 + 索引（一次性，几分钟）

```bash
python scripts/01_parse_docs.py --raw_dir public_dataset_upload/raw --out_dir data/parsed --workers 4
python scripts/02_build_index.py
```

### 4. 验证检索质量（0 token）

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

## 关键设计

### 不用 embedding 也能做高质量检索
- jieba 分词 + 自定义金融词典（50+ 术语）
- BM25 + 条款号 boost（query 含"第 X 条"时含此条款的 chunk × 2.5）
- per-doc 公平采样（多 GT 文档题强制每 doc 至少 N 个 slot）
- 选项增强检索（每选项单独 BM25，权重 0.7 合并入池）

### Token 极致优化
- 离线全部完成：解析、切片、索引（不计预算）
- 单题平均 ~3-5K token（baseline 30K，省 6×）
- max_completion_tokens = 400（JSON CoT 输出空间）
- 100 题总 ~400K token，仅占 5M 预算 8%

### JSON 结构化 CoT
强制模型按 `{analysis:{A:"...",B:"...",C:"...",D:"..."},answer:"AC"}` 输出：
- 逐选项分析 → 不漏判
- 引用证据编号 → 可追溯
- answer 字段稳定 → 后处理简单

### 5 个域专用 system prompt
针对保险/法规/合同/财报/研报各自的术语、计算公式、常见陷阱，避免通用 prompt 在专业题上失误。

## 后续路线

- **v4 候选**（A 榜冲分）：
  - 多选题 self-verification（第二轮自检）
  - Few-shot 示例注入 prompt
  - 难题升级 qwen-max
- **B 榜准备**（7/22 开榜前）：
  - Domain router（关键词预测 domain，缓解 global 检索准确率低）
  - 文档级摘要索引

## 合规性

- ✅ 仅使用 Qwen 系列 API（qwen-plus / qwen3.7-plus）
- ✅ 检索零向量模型，纯统计 BM25
- ✅ 预处理仅用 pdfplumber/pymupdf/bs4（非模型工具，合规）
- ✅ 无 Qwen 离线生成的语义摘要参与正式答题

详见 `plan.md` 完整方案设计文档。
