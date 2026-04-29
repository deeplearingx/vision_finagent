# Vision-FinAgent

基于 ColPali + Milvus Lite + FastAPI 的多模态财报检索与对话系统，提供：

- PDF 财报上传与异步向量化
- 基于 ColPali MaxSim 的页面级检索
- Web 对话界面
- 会话历史保存接口
- OpenAI 兼容私有 VLM 连通能力

---

## 当前运行形态

当前项目已经调整为 **单机 AutoDL 运行模式**，不再以 Docker Milvus standalone 为主路径，而是默认使用：

- **Milvus Lite 本地文件模式**
- **Redis** 用于幂等控制
- **FastAPI + Uvicorn** 提供 API 与前端静态页面
- **ColPali 本地模型目录** 做检索编码

默认前端页面由 FastAPI 直接挂载，访问根路径即可打开界面。

---

## 目录说明

```text
vision_finagent/
├── src/
│   ├── main.py                 # FastAPI 入口
│   ├── config.py               # 配置项
│   ├── routers/reports.py      # 上传/查询/会话历史接口
│   ├── routers/health.py       # 健康检查接口
│   ├── services/retrieval_service.py   # ColPali 检索逻辑
│   ├── services/ingestion_service.py   # PDF 向量化入库
│   └── core/milvus_client.py   # Milvus Lite / collection 管理
├── static/
│   └── index.html              # 前端对话 + 上传界面
├── milvus_local.db             # Milvus Lite 本地数据库文件
└── README.md
```

---

## 环境要求

- Python 3.10+
- Redis
- 可访问的本地 ColPali 模型目录
- Linux / AutoDL 环境

---

## 关键配置

项目从 [`.env`](.env) 读取运行配置。常用项如下：

### Milvus 连接模式

**本地 Lite 模式（默认）**：不设置 `MILVUS_URI`，项目自动使用本地文件 `milvus_local.db`。

**Zilliz Cloud 云端模式**：在 `.env` 中设置以下三项：

```env
MILVUS_URI=https://in03-xxxxxxxxxxxxxxxxx.serverless.gcp-us-west1.cloud.zilliz.com
MILVUS_TOKEN=your_zilliz_api_token_here
MILVUS_DB_NAME=default
```

- `MILVUS_URI`：Zilliz Cloud 控制台 → Cluster → Public Endpoint（保留 `https://`）
- `MILVUS_TOKEN`：Zilliz Cloud 控制台 → API Keys → 生成或复制 token
- `MILVUS_DB_NAME`：默认填 `default`；若创建了独立 database 则填对应名称
- 切换到云端后，原本地 `milvus_local.db` 中的数据**不会自动迁移**，需重新上传 PDF

**自建 Milvus Standalone（无认证）**：

```env
MILVUS_URI=http://your-host:19530
# 不设置 MILVUS_TOKEN
```

---

```env
MILVUS_COLLECTION=fin_vision_reports_v2
REDIS_URL=redis://localhost:6379/0

# MODEL_PATH 可为 LoRA adapter 目录或完整模型目录
MODEL_PATH=/root/autodl-tmp/colpali-v1.2
# 当 MODEL_PATH 是 LoRA adapter 目录时，必须设置 BASE_MODEL_PATH
BASE_MODEL_PATH=/root/autodl-tmp/colpaligemma-3b-pt-448-base

TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1

VLM_API_BASE=https://ark.cn-beijing.volces.com/api/coding/v3
VLM_MODEL=Kimi-K2.6
VLM_API_KEY=your-key

# OpenAI client 层超时
VLM_TIMEOUT=120
# query pipeline 层超时，必须小于 VLM_TIMEOUT
VLM_QUERY_TIMEOUT=90
INGEST_TIMEOUT=600
QUERY_TIMEOUT=15

# GPU 服务器默认强制检索模型驻留 GPU
REQUIRE_RETRIEVAL_GPU=true
```

说明：

- **不要**在 [`.env`](.env) 中设置 `MILVUS_URI=./milvus_local.db`
- 本项目会通过 [`src/config.py`](src/config.py) 自动使用绝对路径形式的 Milvus Lite 文件
- 当 [`MODEL_PATH`](src/config.py) 指向 **LoRA adapter 目录** 时，[`BASE_MODEL_PATH`](src/config.py) 必须指向 **完整 base model 目录**；当前版本已在 [`settings.validate_model_paths()`](src/config.py:43) 中做 fail-fast 校验
- 当 [`MODEL_PATH`](src/config.py) 本身就是完整模型目录时，[`BASE_MODEL_PATH`](src/config.py) 可以留空
- [`VLM_QUERY_TIMEOUT`](src/config.py:43) 必须小于 [`VLM_TIMEOUT`](src/config.py:43)，当前版本已在启动阶段做 fail-fast 校验，避免前端已超时但底层 VLM HTTP 连接仍长时间占用线程
- 在 GPU 服务器上，默认会强制要求检索模型真正加载到 GPU；若未成功上 GPU，服务会直接启动失败，而不是静默退回 CPU

---

## 启动方式

### 1. 启动 Redis

```bash
redis-server --daemonize yes
```

### 2. 启动服务

#### 推荐启动方式

当前项目统一使用 [`start.py`](start.py) 作为**标准启动入口**。它负责：

- 在导入 [`torch`](start.py:1) / [`uvicorn`](start.py:13) 之前执行兼容性保护
- 通过 [`src.main:app`](src/main.py:69) 统一触发 [`lifespan()`](src/main.py:39) 启动流程
- 在主线程内完成 [`warmup_retrieval_model()`](src/main.py:50)，确保检索模型、Redis、Milvus 都在启动阶段完成初始化
- 固定监听 `0.0.0.0:8000`

```bash
cd /root/autodl-tmp/vision_finagent
nohup python start.py > server.log 2>&1 &
```

前台调试可直接运行：

```bash
cd /root/autodl-tmp/vision_finagent
python start.py
```

#### 不作为首选的启动方式

以下命令不是当前文档推荐路径，因为它绕过了 [`start.py`](start.py) 这个统一入口，不利于复现实验结果与排查启动日志：

```bash
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000
```

### 3. 验证服务状态

```bash
curl http://localhost:8000/ready
curl http://localhost:8000/schema-health
nvidia-smi
```

GPU 正常加载成功时返回：

```json
{"status":"ready"}
```

如启动失败，请第一时间看日志：

```bash
tail -f server.log
```

重点关注以下日志字段：

- `cuda_available`
- `first_param_device`
- `hf_device_map`
- `has_cuda_placement`
- `cpu_fallback`
- `model_ready`

如果你在 GPU 服务器上看到模型没有任何层落在 CUDA，服务现在会直接启动失败，这是预期行为。

### 4. 打开前端

浏览器访问：

```text
http://<服务器IP>:8000
```

### 5. 推荐启动后回归顺序

```bash
curl http://localhost:8000/ready
curl http://localhost:8000/schema-health
curl http://localhost:8000/reports/sessions
```

若要做完整回归，建议按以下顺序：

1. 启动 Redis
2. 使用 [`python start.py`](start.py) 启动服务
3. 查看 [`server.log`](server.log)
4. 查看 `nvidia-smi`
5. 检查 [`/ready`](src/routers/health.py:17)
6. 检查 [`/schema-health`](src/routers/health.py:58)
7. 上传真实 PDF / 图片
8. 轮询任务状态
9. 执行一次真实 [`/reports/query`](src/routers/reports.py:341)

---

## 停止与排查

### 停止服务

```bash
pkill -f "python start.py"

```

### 查看日志

```bash
cd /root/autodl-tmp/vision_finagent
tail -f server.log
```

### Redis 检查

```bash
redis-cli ping
```

返回 `PONG` 表示正常。

---

## 常见问题

### 1. `Open local milvus failed`

原因：[`milvus_local.db`](milvus_local.db) 被旧进程占用。

处理：

```bash
pkill -f "python start.py"

```

然后重新启动服务。

---

### 2. 服务启动失败，日志里出现 GPU / device_map / CPU fallback

如果日志中出现以下特征：

- `CUDA is available but no model layer is on CUDA`
- `hf_device_map={'': 'cpu'}`
- `retrieval.model_device_mismatch`

说明当前检索模型没有真正加载到 GPU。当前版本在 GPU 服务器上会把这视为启动失败，而不是静默降级。

优先检查：

1. [`.env`](.env) 中的 [`MODEL_PATH`](src/config.py) 是否指向 LoRA adapter 目录
2. [`.env`](.env) 中的 [`BASE_MODEL_PATH`](src/config.py) 是否指向完整 base model 目录
3. 是否使用了推荐启动命令 [`python start.py`](start.py)
4. GPU 显存是否足够

检查命令：

```bash
nvidia-smi
tail -n 100 server.log
```

正常情况下，启动后 `nvidia-smi` 应看到明显显存占用，而不是只有几 MiB。

---

### 3. 前端对话一直卡住

已修复的根因：

- 检索模型预热不再阻塞服务启动
- 前端查询已加入超时控制
- 后端查询已加入 `QUERY_TIMEOUT`

如果仍然很慢，请先确认：

```bash
curl http://localhost:8000/ready
```

---

### 4. 上传失败，提示缺少 `company_ticker / fiscal_year / form_type`

这是旧接口形式导致的问题。当前上传接口已改成：

- 必填：`file`
- 可选：`report_id`

---

## API 一览

### 健康检查

| Method | Path | 说明 |
|--------|------|------|
| GET | `/health` | 存活检查 |
| GET | `/ready` | 就绪检查，探测 Redis + Milvus + 检索模型就绪状态 |
| GET | `/schema-health` | schema 漂移检查 |

### 报告接口

| Method | Path | 说明 |
|--------|------|------|
| POST | `/reports/upload` | 上传 PDF 并异步向量化 |
| GET | `/reports/tasks/{task_id}` | 轮询上传任务状态 |
| POST | `/reports/query` | 页面检索查询 |
| GET | `/reports/sessions` | 获取会话列表 |
| GET | `/reports/sessions/{session_id}/history` | 获取对话历史 |
| GET | `/reports/sessions/{session_id}/evidence-status` | 获取当前会话 evidence 可复用状态 |
| POST | `/reports/admin/clear-collections` | 清空并重建当前 Milvus 集合 |
| GET | `/reports/{report_id}` | 报告状态占位接口 |

---

## 上传示例

```bash
curl -X POST http://localhost:8000/reports/upload \
  -F "file=@/root/autodl-tmp/vidore_raw/pdfs/bank_of_america_2024.pdf" \
  -F "report_id=bank_of_america_2024"
```

返回示例：

```json
{
  "report_id": "bank_of_america_2024",
  "task_id": "task_xxx",
  "status": "PENDING"
}
```

轮询任务状态：

```bash
curl http://localhost:8000/reports/tasks/task_xxx
```

---

## 查询示例

```bash
curl -X POST http://localhost:8000/reports/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Bank of America 2024年净利润是多少？",
    "target_companies": ["bank_of_america"],
    "top_k": 3,
    "use_retrieval": true,
    "session_id": "demo-session-001"
  }'
```

返回示例：

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

复用上一轮 evidence 追问：

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

## 会话历史示例

```bash
curl http://localhost:8000/reports/sessions/demo-session-001/history
```

---

## 当前查询链路说明

当前 [`/reports/query`](src/routers/reports.py:341) 的主链路为：

1. [`retrieve()`](src/services/retrieval_service.py:227) 做 ColPali 页面级检索
2. 将轻量 evidence 缓存到 Redis
3. 调用 [`generate_answer()`](src/services/vlm_service.py:64) 生成回答
4. 若 VLM 超时或失败，则降级为 evidence 摘要，但仍返回检索结果
5. 将历史、多轮状态、evidence 可复用状态持久化到 Redis

也就是说：

- 当前主路径是 **检索 + VLM + evidence 返回**
- VLM 不是硬依赖；超时会返回 `degraded=true` 与 `degrade_reason`
- 当前会话状态不再保存在内存，而是保存在 Redis 中，支持刷新恢复与 evidence 复用

---

## 架构概览

```text
FastAPI
├── /health, /ready
├── /schema-health
├── /reports/upload
├── /reports/tasks/{task_id}
├── /reports/query
├── /reports/admin/clear-collections
├── /reports/sessions
├── /reports/sessions/{session_id}/history
├── /reports/sessions/{session_id}/evidence-status
└── Static Frontend (/)

Retrieval Flow
├── PDF upload
├── ingestion_service
├── Milvus Lite patch/page collections
├── retrieval_service
├── vlm_service
├── Redis session/evidence cache
└── Frontend chat results
```

---

## 备注

- [`src/graph/`](src/graph/) 中保留了 LangGraph 编排代码
- 当前线上主路径已经是稳定的 [`/reports/query`](src/routers/reports.py:341) 直连链路
- [`probe_triton_spec.py`](probe_triton_spec.py) 用于验证 [`triton.__spec__`](start.py:7) 问题是否真实存在；当前环境结论是“无法证实”，但兼容补丁仍可保留为防御性代码
