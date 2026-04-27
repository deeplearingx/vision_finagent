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

```env
MILVUS_COLLECTION=fin_vision_reports_v2
REDIS_URL=redis://localhost:6379/0
MODEL_PATH=/root/autodl-tmp/colpali-v1.2

TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1

VLM_API_BASE=https://ark.cn-beijing.volces.com/api/coding/v3
VLM_MODEL=Kimi-K2.6
VLM_API_KEY=your-key

VLM_TIMEOUT=120
INGEST_TIMEOUT=600
QUERY_TIMEOUT=20
```

说明：

- **不要**在 [`.env`](.env) 中设置 `MILVUS_URI=./milvus_local.db`
- 本项目会通过 [`src/config.py`](src/config.py) 自动使用绝对路径形式的 Milvus Lite 文件

---

## 启动方式

### 1. 启动 Redis

```bash
redis-server --daemonize yes
```

### 2. 启动服务

```bash
cd /root/autodl-tmp/vision_finagent
nohup python -m uvicorn src.main:app --host 0.0.0.0 --port 8080 > server.log 2>&1 &
```

### 3. 验证服务状态

```bash
curl http://localhost:8080/ready
```

成功时返回：

```json
{"status":"ready"}
```

### 4. 打开前端

浏览器访问：

```text
http://<服务器IP>:8080
```

---

## 停止与排查

### 停止服务

```bash
pkill -f "python -m uvicorn src.main:app"
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
pkill -f "python -m uvicorn src.main:app"
```

然后重新启动服务。

---

### 2. 前端对话一直卡住

已修复的根因：

- 检索模型预热不再阻塞服务启动
- 前端查询已加入超时控制
- 后端查询已加入 `QUERY_TIMEOUT`

如果仍然很慢，请先确认：

```bash
curl http://localhost:8080/ready
```

---

### 3. 上传失败，提示缺少 `company_ticker / fiscal_year / form_type`

这是旧接口形式导致的问题。当前上传接口已改成：

- 必填：`file`
- 可选：`report_id`

---

## API 一览

### 健康检查

| Method | Path | 说明 |
|--------|------|------|
| GET | `/health` | 存活检查 |
| GET | `/ready` | 就绪检查，探测 Redis + Milvus |

### 报告接口

| Method | Path | 说明 |
|--------|------|------|
| POST | `/reports/upload` | 上传 PDF 并异步向量化 |
| POST | `/reports/query` | 页面检索查询 |
| GET | `/reports/sessions/{session_id}/history` | 获取对话历史 |
| GET | `/reports/{report_id}` | 报告状态占位接口 |

---

## 上传示例

```bash
curl -X POST http://localhost:8080/reports/upload \
  -F "file=@/root/autodl-tmp/vidore_raw/pdfs/bank_of_america_2024.pdf" \
  -F "report_id=bank_of_america_2024"
```

返回示例：

```json
{
  "report_id": "bank_of_america_2024",
  "status": "PENDING"
}
```

---

## 查询示例

```bash
curl -X POST http://localhost:8080/reports/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Bank of America 2024年净利润是多少？",
    "target_companies": ["bank_of_america"],
    "top_k": 3,
    "session_id": "demo-session-001"
  }'
```

返回示例：

```json
{
  "session_id": "demo-session-001",
  "answer": null,
  "retrieved_pages": [
    {
      "report_id": "bank_of_america_2024",
      "page_num": 12,
      "maxsim_score": 8.73
    }
  ]
}
```

---

## 会话历史示例

```bash
curl http://localhost:8080/reports/sessions/demo-session-001/history
```

---

## 当前查询链路说明

当前 [`/reports/query`](src/routers/reports.py) 为了保证实时性，默认走：

1. [`retrieve()`](src/services/retrieval_service.py) 做 ColPali 检索
2. 返回页面级结果
3. 将对话摘要保存到内存 session

也就是说：

- 当前前端对话偏向 **检索式对话**
- 不强制依赖 VLM 才能返回结果
- VLM 连通性已单独验证正常

---

## 架构概览

```text
FastAPI
├── /health, /ready
├── /reports/upload
├── /reports/query
├── /reports/sessions/{session_id}/history
└── Static Frontend (/)

Retrieval Flow
├── PDF upload
├── ingestion_service
├── Milvus Lite patch/page collections
├── retrieval_service
└── Frontend chat results
```

---

## 备注

- [`src/graph/`](src/graph/) 中保留了 LangGraph 编排代码
- 当前线上查询主路径已经优先切到更稳定的直接检索路径
- 如果后续要恢复“检索 + VLM 回答”的完整智能问答链路，建议把 VLM 调用做成显式可选开关，而不是所有查询默认强依赖
