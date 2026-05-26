# Vision FinAgent：多模态金融报告分析 Agent

> 面向金融财报 / 年报场景的多模态 RAG 问答系统。系统支持 PDF / 图片上传、页面级解析、视觉向量建库、财报证据页召回、VLM 回答生成、多轮会话管理和 evidence 复用，目标是让用户能够基于原始财报页面进行可追溯的问答分析。

---

## 面试官快速查看

### 项目定位

Vision FinAgent 不是普通文本 RAG，而是一个面向财报页面的 **多模态检索增强问答系统**。财报中的关键信息大量存在于表格、版式、图片和跨页结构中，因此项目采用 ColPali 对页面图像进行多向量表示，并通过 MaxSim 完成 query-page 相关性计算，最后把 evidence、page_num、maxsim_score 等证据信息返回给 VLM 和前端。

### 核心能力

- **多模态入库**：支持 PDF / 图片上传，完成页面渲染、文本抽取、图像编码、向量入库和任务状态查询。
- **页面级证据召回**：基于 ColPali + Milvus 实现财报页面级检索，返回 evidence、page_num、maxsim_score 等结构化证据信息。
- **VLM 证据问答**：将检索到的财报页面与证据上下文传入 VLM，生成基于原始页面的回答。
- **多轮会话与证据复用**：基于 Redis 管理 session history 与 evidence cache，支持追问、刷新恢复和检索失败兜底。
- **二次检索增强**：当首轮回答缺少数字、引用或证据不足时，扩大召回范围并重新生成回答，提升数值类金融问答稳定性。
- **工程化部署**：基于 FastAPI 提供服务接口，支持健康检查、schema 检查、超时控制、降级回答和 GPU 加载校验。

### 当前评测口径

当前已构建 bank-class 财报问答冒烟评测集，用于验证多模态 RAG 主链路，包括数值查询、证据页召回与答案引用等场景。

| 指标 | 含义 | 当前结果 |
| --- | --- | --- |
| Evidence Recall@5 | Top-5 召回页面中是否包含标注证据页 | 87.5% |
| Answer Accuracy | 回答是否命中正确财报数值 / 结论 | 91.7% |

> 说明：当前指标用于小规模冒烟验证和回归观察，不等价于大规模公开 benchmark。后续可继续扩展到更多公司、更多年份和更多问题类型。

---

## 技术栈

| 模块 | 技术 |
| --- | --- |
| Web 服务 | FastAPI, Uvicorn, Pydantic Settings |
| 多模态检索 | ColPali, MaxSim, PyTorch, Transformers |
| 向量数据库 | Milvus Lite / Zilliz Cloud / Milvus Standalone |
| 会话与缓存 | Redis session history, evidence cache |
| 文档处理 | PyMuPDF, PDF page rendering, text extraction |
| 回答生成 | OpenAI-compatible VLM API |
| 工程化 | health check, schema check, timeout, degraded fallback, GPU fail-fast |

---

## 系统架构

```text
User / Web UI
    │
    ▼
FastAPI
    ├── /reports/upload              # 文件上传与异步入库
    ├── /reports/tasks/{task_id}      # 入库任务状态查询
    ├── /reports/query                # 多模态检索问答
    ├── /reports/sessions             # 会话列表
    ├── /ready, /schema-health        # 健康检查与 schema 检查
    │
    ▼
Ingestion Pipeline
    ├── PDF / Image validation
    ├── PyMuPDF page rendering
    ├── page text extraction
    ├── ColPali page embedding
    └── Milvus page vector collection
    │
    ▼
Query Pipeline
    ├── query embedding
    ├── MaxSim page retrieval
    ├── evidence construction
    ├── Redis evidence cache
    ├── VLM answer generation
    └── degraded fallback if timeout / failure
```

---

## 核心流程

### 1. 财报入库

```text
PDF / Image Upload
→ 页面渲染与文本抽取
→ ColPali 页面级视觉向量生成
→ Milvus 写入页面向量、页码、报告 ID、图片路径和文本元数据
→ 返回 task_id，前端轮询任务状态
```

### 2. 财报问答

```text
Question
→ ColPali query encoding
→ Milvus candidate retrieval
→ MaxSim relevance scoring
→ Top-K evidence page construction
→ VLM answer generation
→ 返回 answer + evidence + retrieved_pages + maxsim_score
```

### 3. 多轮追问

```text
session_id
→ Redis 读取历史对话与上一轮 evidence
→ 支持 use_retrieval=false 复用上一轮证据
→ 支持证据不足时重新检索
```

---

## 项目目录

```text
vision_finagent/
├── src/
│   ├── main.py                         # FastAPI 入口与 lifespan 初始化
│   ├── config.py                       # 配置项与 fail-fast 校验
│   ├── routers/
│   │   ├── reports.py                  # 上传、查询、会话历史、管理接口
│   │   └── health.py                   # 健康检查与 schema 检查
│   ├── services/
│   │   ├── ingestion_service.py        # PDF / 图片解析与向量化入库
│   │   ├── retrieval_service.py        # ColPali + MaxSim 检索逻辑
│   │   └── vlm_service.py              # VLM 回答生成与降级逻辑
│   └── core/
│       └── milvus_client.py            # Milvus collection 管理
├── static/
│   └── index.html                      # 简易上传与问答前端
├── start.py                            # 推荐启动入口
└── README.md
```

---

## 快速启动

### 1. 启动 Redis

```bash
redis-server --daemonize yes
```

### 2. 配置环境变量

项目从 `.env` 读取配置。常用配置如下：

```env
MILVUS_COLLECTION=fin_vision_reports_v2
REDIS_URL=redis://localhost:6379/0

MODEL_PATH=/root/autodl-tmp/colpali-v1.2
BASE_MODEL_PATH=/root/autodl-tmp/colpaligemma-3b-pt-448-base

TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1

VLM_API_BASE=https://your-openai-compatible-endpoint/v1
VLM_MODEL=your-vlm-model
VLM_API_KEY=your-key

VLM_TIMEOUT=120
VLM_QUERY_TIMEOUT=90
INGEST_TIMEOUT=600
QUERY_TIMEOUT=15
REQUIRE_RETRIEVAL_GPU=true
```

Milvus 默认使用本地 Lite 文件模式；如果需要切换到 Zilliz Cloud 或自建 Milvus，可配置：

```env
MILVUS_URI=https://your-zilliz-or-milvus-endpoint
MILVUS_TOKEN=your-token
MILVUS_DB_NAME=default
```

### 3. 启动服务

```bash
python start.py
```

后台运行：

```bash
nohup python start.py > server.log 2>&1 &
```

### 4. 检查服务状态

```bash
curl http://localhost:8000/ready
curl http://localhost:8000/schema-health
nvidia-smi
```

浏览器访问：

```text
http://<server-ip>:8000
```

---

## API 示例

### 上传财报

```bash
curl -X POST http://localhost:8000/reports/upload \
  -F "file=@/path/to/bank_of_america_2024.pdf" \
  -F "report_id=bank_of_america_2024"
```

返回：

```json
{
  "report_id": "bank_of_america_2024",
  "task_id": "task_xxx",
  "status": "PENDING"
}
```

轮询任务：

```bash
curl http://localhost:8000/reports/tasks/task_xxx
```

### 查询财报

```bash
curl -X POST http://localhost:8000/reports/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Bank of America 2024 年净利润是多少？",
    "target_companies": ["bank_of_america"],
    "top_k": 3,
    "use_retrieval": true,
    "session_id": "demo-session-001"
  }'
```

典型返回字段：

```json
{
  "session_id": "demo-session-001",
  "answer": "...",
  "degraded": false,
  "degrade_reason": "none",
  "evidence_source": "new_retrieval",
  "evidence": [
    {
      "report_id": "bank_of_america_2024",
      "page_num": 12,
      "maxsim_score": 8.73
    }
  ],
  "retrieved_pages": [
    {
      "report_id": "bank_of_america_2024",
      "page_num": 12,
      "maxsim_score": 8.73,
      "image_base64": "..."
    }
  ]
}
```

复用上一轮 evidence：

```bash
curl -X POST http://localhost:8000/reports/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "继续基于上一次证据，总结核心结论",
    "session_id": "demo-session-001",
    "use_retrieval": false
  }'
```

---

## 健康检查与管理接口

| Method | Path | 说明 |
| --- | --- | --- |
| GET | `/health` | 存活检查 |
| GET | `/ready` | Redis + Milvus + 检索模型就绪检查 |
| GET | `/schema-health` | Milvus schema 漂移检查 |
| POST | `/reports/upload` | 上传 PDF / 图片并异步向量化 |
| GET | `/reports/tasks/{task_id}` | 查询入库任务状态 |
| POST | `/reports/query` | 财报检索问答 |
| GET | `/reports/sessions` | 获取会话列表 |
| GET | `/reports/sessions/{session_id}/history` | 获取历史对话 |
| GET | `/reports/sessions/{session_id}/evidence-status` | 获取 evidence 可复用状态 |
| GET | `/reports/admin/inventory` | 枚举已入库报告 |
| POST | `/reports/admin/clear-collections` | 清空并重建 Milvus 集合 |

---

## 工程化设计要点

- **GPU fail-fast**：在 GPU 服务器上若检索模型未真正加载到 CUDA，服务直接启动失败，避免静默 CPU fallback。
- **超时分层**：区分 VLM HTTP 超时、查询 pipeline 超时和入库超时，避免前端长时间挂起。
- **降级返回**：当 VLM 超时或失败时仍返回 evidence 摘要，保证用户能看到检索证据。
- **Redis 状态持久化**：会话历史与 evidence cache 不保存在内存，支持刷新恢复与多轮追问。
- **schema-health**：提供向量库 schema 漂移检查，便于定位 collection 字段不一致问题。

---

## 可向面试官说明的项目亮点

- 针对财报表格和复杂版式场景，引入 ColPali 页面级视觉检索，避免纯文本切分丢失关键信息。
- 使用 MaxSim late-interaction 计算 query 与页面多向量相关性，提升页面级 evidence 定位能力。
- 通过 evidence cache、多轮 session 和 degraded fallback 提升系统可用性。
- 通过 Recall@K、Answer Accuracy、Citation Accuracy 等指标评估检索和回答链路，而不是只看模型生成效果。
