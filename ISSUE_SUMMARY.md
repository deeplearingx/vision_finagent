# Vision-FinAgent 项目问题排查与修复汇总

本文档汇总本项目在推进过程中遇到的主要问题、问题成因、修复方式与当前状态，便于后续迁移对话、继续开发、做回归验证或上线前复核。

---

## 1. 项目背景

项目目标：构建一个基于多模态财报页面检索与问答的系统，支持：

- PDF / 图片上传
- 财报页面向量化入库
- 基于 ColPali 的页面检索
- 基于 VLM 的证据页问答
- Web 对话界面
- 会话历史、evidence 复用、维护态清库

主要技术栈：

- FastAPI
- Milvus Lite
- Redis
- ColPali / ColPaliProcessor
- OpenAI 兼容 VLM 接口

关键代码位置：

- 服务入口：`src/main.py`
- 检索服务：`src/services/retrieval_service.py`
- 上传与查询路由：`src/routers/reports.py`
- 入库服务：`src/services/ingestion_service.py`
- Milvus 客户端：`src/core/milvus_client.py`
- 健康检查：`src/routers/health.py`
- 启动包装脚本：`start.py`

---

## 2. 环境与基础设施问题

### 2.1 Milvus Lite 路径被误识别为 URI

**遇到的问题**

- 服务启动时报本地 Milvus 打开失败或非法 URI 类错误。

**为什么会发生**

- `pymilvus` 在读取环境变量时，会把相对路径形式的 `MILVUS_URI` 当成网络 URI 处理。
- 使用 `./milvus_local.db` 这类相对路径时，解析行为不符合预期。

**如何解决**

- 不再依赖相对路径 `MILVUS_URI`
- 在配置层统一改为项目内绝对路径的 Milvus Lite 文件
- 相关逻辑最终集中在 `src/config.py`

**结果**

- Milvus Lite 文件模式可稳定启动。

---

### 2.2 Redis / Milvus / FastAPI 启动顺序与就绪语义不一致

**遇到的问题**

- 服务看起来已经启动，但首轮查询超时或失败。

**为什么会发生**

- 启动时基础连接已就绪，但检索模型还没有真正完成 warmup。
- 之前 warmup 一度放到后台，导致“服务启动成功”与“模型真正可查询”之间存在时间窗错位。

**如何解决**

- 调整 `src/main.py` 中的 `lifespan` 行为
- 改为在启动阶段阻塞等待检索 warmup 完成后再进入服务可用状态

**结果**

- 当前语义已经调整为：模型 warmup 完成后，服务才算启动成功。

---

## 3. 上传链路问题

### 3.1 PDF 解析方式错误

**遇到的问题**

- 上传 PDF 时失败，报图像识别类错误。

**为什么会发生**

- 早期实现直接按普通图片方式读取 PDF。
- PDF 需要先渲染成逐页图像，不能直接按普通图片打开。

**如何解决**

- 在 `src/services/ingestion_service.py` 中改为使用 `fitz` / PyMuPDF 渲染 PDF 页面
- 再将页面转成 PIL Image

**结果**

- PDF 上传链路恢复正常。

---

### 3.2 上传任务阻塞 API，导致 `/ready` 无响应或服务发僵

**遇到的问题**

- 上传期间服务响应显著变慢
- 健康检查、前端交互看起来像卡死

**为什么会发生**

- 入库属于重型任务，包含：
  - 文件解析
  - 模型编码
  - Milvus 写入
- 早期直接在请求上下文内执行，阻塞事件循环或工作线程。

**如何解决**

- 在 `src/routers/reports.py` 中将上传改成异步任务模式
- 在 `src/services/task_service.py` 中引入后台任务状态管理
- 前端改成轮询任务状态，而不是等待单个长请求完成

**结果**

- 上传接口返回 `task_id`
- `/reports/tasks/{task_id}` 可查询状态
- API 主线程不再被长时间占住。

---

### 3.3 非法 PDF / 伪图片进入后台任务，失败时机太晚

**遇到的问题**

- 用户上传伪 PDF 或损坏图片，接口表面接受成功，但后台任务才失败。

**为什么会发生**

- 入口只做扩展名判断，没做轻量内容校验。

**如何解决**

- 在 `src/routers/reports.py` 增加：
  - PDF 头尾标记检查
  - 图片 magic bytes 检查
- 对解析阶段异常在 `src/services/ingestion_service.py` 中补更明确的错误信息

**结果**

- 明显无效文件在入口就会被 4xx 拒绝
- 进入后台任务的文件基本都是结构有效文件。

---

### 3.4 上传失败后，同一 `report_id` 无法再次上传

**遇到的问题**

- 同一个 `report_id` 上传失败后，再次上传会收到 `409 Duplicate upload`。

**为什么会发生**

- 幂等 token 在上传开始时创建
- 失败后未释放，导致后续重试仍命中幂等拒绝

**如何解决**

- 在 `src/utils/idempotency.py` 中增加 token 释放逻辑
- 在 `src/services/task_service.py` 中，对失败任务自动释放 `upload:{report_id}` token

**结果**

- 上传失败后，相同 `report_id` 可再次重传。

---

### 3.5 同一 `report_id` 重传导致旧 patch/page 数据累积

**遇到的问题**

- 同一 `report_id` 再次入库时，旧向量残留，造成检索污染。

**为什么会发生**

- pages 集合用 `upsert`，会覆盖
- patches 集合用 `insert`，会追加
- 重传前没有清旧数据

**如何解决**

- 在 `src/core/milvus_client.py` 增加按 `report_id` 清理的能力
- 在 `src/services/ingestion_service.py` 中，正式写入前先清理同 `report_id` 的旧 patch/page 数据

**结果**

- 重传不会再累积旧向量脏数据。

---

## 4. 查询与对话链路问题

### 4.1 查询只返回检索结果，没有真正回答

**遇到的问题**

- 前端只看到“找到若干相关页面”，没有自然语言回答。

**为什么会发生**

- 为规避 VLM 超时，早期版本曾绕开 VLM，只保留纯检索输出。

**如何解决**

- 将 `src/routers/reports.py` 中的查询逻辑改回混合模式：
  - 检索
  - VLM 生成回答
  - 返回 evidence
- 保留降级路径：VLM 超时/异常时仍返回 evidence 摘要

**结果**

- 前端可获得 `answer + evidence`
- 降级场景也不会整体失败。

---

### 4.2 查询超时控制不匹配，前端经常看到“查询超时”

**遇到的问题**

- 前端经常显示查询超时
- 即使 `/ready` 正常，短问题也可能失败

**为什么会发生**

- 前端超时和后端检索/VLM超时预算不协调
- 用户在模型尚未 warmup 完成前就发起请求
- VLM 阶段曾有 20–25 秒的超时窗口

**如何解决**

- 多次调整前后端超时策略
- 增加更清晰的降级输出
- 后续又把模型 warmup 移回启动阻塞阶段，进一步收敛语义

**结果**

- 纯超时问题已显著减少，但这块历史上是反复出现的问题源头之一。

---

### 4.3 evidence 缓存、会话恢复、前后端状态不一致

**遇到的问题**

- 刷新页面后会新建会话
- 历史会话不易切换
- 前端有时误以为存在 evidence，可后端实际已无可复用缓存

**为什么会发生**

- 会话最早只是内存字典
- 后续迁移到 Redis 后，前端仍有基于历史消息错误推断 evidence 的问题

**如何解决**

- 后端增加会话列表、会话 meta、evidence status 接口
- 前端改为：
  - 恢复上次活跃会话
  - 通过后端真实接口判断 `has_evidence`
  - 支持显式新建会话

**结果**

- 会话恢复和会话切换逻辑已基本闭环。

---

### 4.4 每轮都检索太慢，需要支持 evidence 复用

**遇到的问题**

- 用户每问一句都触发新检索，体验太慢

**为什么会发生**

- 查询接口最初默认每轮都走检索

**如何解决**

- 在 `src/routers/reports.py` 的查询请求中扩展检索开关
- 支持：
  - 本轮新检索
  - 关闭检索时复用当前会话最近一次 evidence
- 前端增加显式“本轮检索数据”开关

**结果**

- 用户可按需检索
- 后续追问可复用上次 evidence
- 减少重复慢检索。

---

### 4.5 evidence cache 太重

**遇到的问题**

- Redis 中缓存完整页面对象，含 `image_base64`，单会话 payload 太大

**为什么会发生**

- 早期 evidence cache 直接存完整 `PageResult`

**如何解决**

- 将 evidence cache 轻量化，只保留轻量 meta
- 需要真实图像时再回查 Milvus pages 集合

**结果**

- Redis 压力下降
- 同时新增了“回查失败时的日志和可解释性”。

---

## 5. Milvus schema 与数据一致性问题

### 5.1 pages 集合 `image_base64` 字段长度过小

**遇到的问题**

- 高分辨率页面 base64 可能写入失败或截断

**为什么会发生**

- 最早 schema 为较小的 `VARCHAR` 长度，无法覆盖真实页面大小

**如何解决**

- 在 `src/core/milvus_client.py` 中扩大 `image_base64` 字段长度
- 增加清库与重建集合能力，使新 schema 可真正生效

**结果**

- 结合清库后重新上传，可适应更大的页面图像 payload。

---

### 5.2 schema 漂移无法发现

**遇到的问题**

- collection 已存在但字段结构落后于代码时，系统静默继续运行。

**为什么会发生**

- 只在不存在 collection 时创建，没有对已有 collection 做字段核对。

**如何解决**

- 在 `src/core/milvus_client.py` 中增加 schema drift 检查
- 对 patches / pages 两个集合都补齐检测
- 增加 [`/schema-health`](src/routers/health.py) 接口供运维检查

**结果**

- schema 漂移从“静默问题”变成“可诊断问题”。

---

### 5.3 清库后 Redis 与 Milvus 状态不一致

**遇到的问题**

- 清空 Milvus 后，Redis 中仍可能残留旧 evidence 或 `has_evidence=true`

**为什么会发生**

- 清库只处理 Milvus，不同步处理 session evidence/meta

**如何解决**

- 在 `clear_collections` 中同步：
  - 清 evidence key
  - 扫描 meta，将 `has_evidence` 置 `false`

**结果**

- 清库后的会话状态与 Milvus 状态保持一致。

---

## 6. 入库回滚问题

### 6.1 回滚不精确 / 存在孤立数据

**遇到的问题**

- 入库中途失败后，可能只回滚 patch，不回滚 page
- 或按估算页数回滚，不够精确

**为什么会发生**

- 早期逻辑只记录粗粒度状态
- 没有按真实成功写入的 page_id / patch 主键回滚

**如何解决**

- 在 `src/services/ingestion_service.py` 中：
  - 记录真实写入的 patch PK 列表
  - 记录真实写入的 page_id 列表
  - 失败时精确回滚

**结果**

- 入库失败后脏数据风险显著降低。

---

### 6.2 page / patch 写入顺序风险

**遇到的问题**

- 先写 page 再写 patch 时，失败窗口里可能出现“页面已在库里，但向量不存在”

**为什么会发生**

- Milvus 没有跨集合事务

**如何解决**

- 改为 patch 先写、page 后写
- 并用注释明确失败窗口与回滚契约

**结果**

- 失败模式从“误导性成功”变成“更可观测的降级/失败”。

---

## 7. GPU / 模型加载问题（最新重点）

### 7.1 初始误判：ready 但显存几乎不占

**遇到的问题**

- `/ready` 返回正常
- 查询能跑
- 但 `nvidia-smi` 几乎没有显存占用

**为什么会发生**

- 早期只用“模型对象存在”判断 ready
- 后来又误用“首参数在 CPU 就代表没上 GPU”判断，导致过度误杀
- accelerate 的分层 `device_map` 让问题判断变复杂

**如何解决（阶段性）**

- 先加入：
  - `get_model_device`
  - `get_hf_device_map`
  - `_has_cuda_placement`
  - `_get_input_device`
- 修正：在分层 device_map 下不再依赖 `model.device`

**结果**

- 解决了“首参数在 CPU 导致误判”的一部分问题，但后来又发现当前环境实际并不是分层 GPU，而是真正 fallback 到 CPU。

---

### 7.2 `device_map="auto"` 在当前环境退到 CPU

**遇到的问题**

- 启动日志中 `hf_device_map={'': 'cpu'}`
- 启动阶段直接失败

**为什么会发生**

- `MODEL_PATH` 实际是 LoRA adapter 目录，不是完整 base model
- 直接用 adapter 路径做 `from_pretrained(..., device_map="auto")`，accelerate 无法正常按完整模型推断 GPU 放置
- 再叠加 uvicorn CLI + triton / CUDA lazy init 环境问题

**如何解决**

- 在 `src/config.py` 中新增 `BASE_MODEL_PATH`
- 明确区分：
  - base model 路径
  - adapter 路径
- 在 `src/services/retrieval_service.py` 中优先显式上 `cuda:0`
- 在项目根目录新增 [start.py](start.py) 作为推荐启动方式

**结果**

- 最新子任务声称：
  - 启动后显存占用约 6GB
  - `/ready` 成功
  - 查询成功

**注意**

- 这一块必须由后续接手对话再次做真实运行复验，确认当前工作区文件与子任务报告一致。

---

## 8. 自动化测试与测试基础设施问题

### 8.1 pytest 在当前环境 segfault

**遇到的问题**

- 当前 Python / torch / CUDA 环境下，pytest 收集阶段可能 segfault

**为什么会发生**

- 更偏环境级兼容问题，不是业务代码本身 bug

**如何解决**

- 建立了 `tests/` 目录结构
- 增加了多组关键回归测试
- 在当前环境下优先用 [tests/run_tests.py](tests/run_tests.py) 或 Python 直接执行逻辑验证

**结果**

- 自动化回归基础设施已经存在，但标准 pytest 仍有环境限制。

---

## 9. 文档与启动流程更新

### 9.1 README 与实际启动流程不一致

**遇到的问题**

- README 里还写着老的 `uvicorn` 启动方式

**为什么会发生**

- 启动策略后来改成 [start.py](start.py)，但文档未及时同步

**如何解决**

- 已更新 [README.md](README.md)
- 同步说明：
  - 新启动流程
  - `BASE_MODEL_PATH`
  - `REQUIRE_RETRIEVAL_GPU`
  - GPU / CPU fallback 排障

**结果**

- 文档现已更贴近当前代码状态。

---

## 10. 当前项目进度

### 已完成
- 环境稳定化（Milvus Lite / Redis / 配置）
- 上传任务化
- PDF / 图片上传支持
- 会话历史与 Redis 持久化
- evidence 复用
- 清库维护态与 schema 健康检查
- 多轮关键 bug 修复
- 自动化测试基础设施与关键回归测试
- README 新启动流程更新

### 当前最重要的待确认项
- 真实运行回归：确认最新代码确实能通过 [python start.py](start.py) 稳定上 GPU
- 验证最新子任务声称的显存占用 / ready / query 结果是否与工作区实际代码一致

---

## 11. 当前待完成项

1. 用真实环境重新执行：
   - `python start.py`
   - `nvidia-smi`
   - `curl http://localhost:8080/ready`
   - 一次真实 `POST /reports/query`

2. 清库并重传真实网页端样本，验证：
   - schema 生效
   - 上传链路稳定
   - 查询链路稳定

3. 统一确认当前 `.env` 中这些关键项是否最终正确：
   - `MODEL_PATH`
   - `BASE_MODEL_PATH`
   - `REQUIRE_RETRIEVAL_GPU`
   - `MILVUS_COLLECTION`

4. 若需要上线，建议补做：
   - 一次完整前端回归
   - 一次真实文件上传 + 多轮对话回归
   - 对 [start.py](start.py) 启动路径做发布脚本固化

---

## 12. 当前注意事项

1. **启动必须优先使用 [start.py](start.py)**
   - 不要默认继续用 `python -m uvicorn src.main:app ...`

2. **GPU 服务器应默认强制上 GPU**
   - 不能长期接受 CPU fallback 作为正常上线状态

3. **LoRA adapter 与 base model 路径必须分开配置**
   - `MODEL_PATH`：adapter
   - `BASE_MODEL_PATH`：完整 base model

4. **不要只看 `/ready`，还要看显存与真实查询**
   - 因为历史上就出现过 ready 与真实 GPU 状态错位的问题

5. **pytest 结果在当前环境不一定可信**
   - 优先看已经建立的备用测试运行方式

---

## 13. 后续接手建议顺序

1. 先核实当前代码是否已包含：
   - [start.py](start.py)
   - `BASE_MODEL_PATH`
   - 新版 `retrieval_service`
   - 更新后的 `README`

2. 立刻做一次真实运行回归：
   - 启动
   - 显存
   - `/ready`
   - 一次查询

3. 若启动仍有异常，优先排查：
   - base model 路径
   - adapter 路径
   - GPU strict 配置
   - 当前启动命令是否正确

4. 启动稳定后，再做清库、重上传、前端问答全链路验证

---

## 14. 一句话总括

项目已经从“上传卡死、会话丢失、evidence 状态错乱、schema 不可见、GPU 就绪语义混乱”的状态，推进到了“核心链路基本可用、关键 bug 大多已修、自动化回归基础已具备”的阶段；当前最关键的收尾工作，是**验证最新 GPU 显式加载方案在真实环境下稳定成立，并完成一次完整真实回归。**
