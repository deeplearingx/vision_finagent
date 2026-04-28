# 第二阶段优化：保守的两轮 VLM 证据扩展

## 实施概览

已完成保守的两轮 VLM 证据扩展，保持 `/reports/query` 快路径不变，无 schema 迁移，完全向后兼容。

## 核心改动文件

### 1. [`src/config.py`](autodl-tmp/vision_finagent/src/config.py)
**改动**：新增 4 个第二轮配置项
```python
VLM_SECOND_PASS_ENABLED: bool = True
VLM_SECOND_PASS_TOP_K: int = 10
VLM_SECOND_PASS_CANDIDATE_K: int = 100
VLM_SECOND_PASS_MAX_IMAGES: int = 10
```

**设计权衡**：
- 默认启用第二轮，生产环境可通过 `.env` 关闭
- 第二轮 top_k=10 是第一轮 5 的 2 倍，candidate_k=100 是第一轮 50 的 2 倍
- 独立的 `MAX_IMAGES` 限制第二轮图片数，避免 VLM 超时

---

### 2. [`src/services/vlm_service.py`](autodl-tmp/vision_finagent/src/services/vlm_service.py)
**改动**：
1. 新增常量 `INSUFFICIENT_EVIDENCE_PHRASE = "Insufficient evidence in provided pages"`
2. 新增函数 `is_insufficient_evidence(answer: str | None) -> bool`
3. 在 `_FIN_DOC_SYSTEM` 中使用常量（保持 DRY 原则）

**设计权衡**：
- **保守触发**：只有当 VLM 明确返回该短语时才触发第二轮
- **不触发的情况**：`vlm_timeout`、`vlm_error`、空响应、None 均不触发
- 避免字符串硬编码，统一在 prompt 和判定逻辑中使用同一常量

---

### 3. [`src/services/retrieval_service.py`](autodl-tmp/vision_finagent/src/services/retrieval_service.py)
**改动**：
- `retrieve()` 新增可选参数 `pass_label: str = "first_pass"`
- 在日志中输出 `pass_label`，用于区分第一轮/第二轮检索

**设计权衡**：
- **完全向后兼容**：默认值 `"first_pass"` 保证现有调用无需修改
- **可观测性优先**：日志中明确标记 `first_pass` / `second_pass`，便于生产排查

---

### 4. [`src/routers/reports.py`](autodl-tmp/vision_finagent/src/routers/reports.py:341)
**改动**：核心两轮逻辑实现

**控制流**：
```
第一轮：检索 → VLM
  ↓
判定：is_insufficient_evidence(answer) && VLM_SECOND_PASS_ENABLED?
  ↓ 是
第二轮：扩展检索（top_k=10, candidate_k=100）→ VLM
  ↓
若第二轮成功：使用第二轮答案和页面
若第二轮失败：保留第一轮答案（降级语义）
```

**新增响应字段**（additive，不破坏前端兼容性）：
- `vlm_passes`: int — 实际执行的 VLM 轮数（0/1/2）
- `second_pass_triggered`: bool — 是否触发第二轮
- `evidence_source_detail`: str — 证据来源详情（与现有 `evidence_source` 互补）

**设计权衡**：
- **不覆盖第一轮成功答案**：只有当第一轮返回"证据不足"时才扩展
- **第二轮失败时保留第一轮结果**：避免用户看到更差的体验
- **日志完整性**：记录每轮耗时、页数、VLM 原因，便于性能分析

---

## 风险控制验证

### ✅ 已确认的安全保障

1. **无 schema 迁移**：未修改 Milvus collection 字段
2. **Redis cache 语义不变**：evidence cache 结构未改动
3. **前端兼容性**：新增字段为 additive，现有字段全部保留
4. **接口向后兼容**：`retrieve()` 默认参数保证现有调用无需修改
5. **不破坏快路径**：第一轮逻辑与原实现完全一致

### ✅ 保守触发条件

- ❌ **不触发**：VLM timeout、error、空响应、None
- ❌ **不触发**：第一轮返回正常答案
- ✅ **触发**：第一轮成功返回且包含 `INSUFFICIENT_EVIDENCE_PHRASE`

---

## 测试与回归

### 单元测试
创建 [`tests/test_second_pass.py`](autodl-tmp/vision_finagent/tests/test_second_pass.py)，覆盖：
- `is_insufficient_evidence()` 7 种边界情况
- 配置默认值验证
- 控制流触发条件（正常答案、证据不足、VLM 错误、配置关闭）

### 回归脚本
[`regression_second_pass.py`](autodl-tmp/vision_finagent/regression_second_pass.py) — **32/32 全部通过**

验证项：
- ✓ 6 个 sentinel 判定测试
- ✓ 4 个配置默认值测试
- ✓ 3 个 `retrieve()` 签名兼容性测试
- ✓ 8 个 `reports.py` 控制流标识符测试
- ✓ 11 个响应 schema 字段测试（8 个现有 + 3 个新增）

---

## 可观测性增强

### 日志标记
- `vlm.first_pass_done` — 第一轮完成，含页数、耗时、答案预览
- `vlm.second_pass_triggered` — 触发第二轮，含扩展参数
- `vlm.second_pass_done` — 第二轮完成
- `vlm.second_pass_no_improvement` — 第二轮未改善
- `retrieval.candidates` / `retrieval.done` — 含 `pass_label` 字段

### 响应字段
```json
{
  "vlm_passes": 2,
  "second_pass_triggered": true,
  "evidence_source_detail": "second_pass"
}
```

---

## 生产配置建议

### 默认配置（已生效）
```env
VLM_SECOND_PASS_ENABLED=true
VLM_SECOND_PASS_TOP_K=10
VLM_SECOND_PASS_CANDIDATE_K=100
VLM_SECOND_PASS_MAX_IMAGES=10
```

### 保守调优
若第二轮成本过高，可：
1. 降低 `VLM_SECOND_PASS_TOP_K=8`（减少检索页数）
2. 降低 `VLM_SECOND_PASS_MAX_IMAGES=8`（减少 VLM 输入图片）
3. 关闭 `VLM_SECOND_PASS_ENABLED=false`（完全禁用）

### 激进调优
若召回率仍不足，可：
1. 提高 `VLM_SECOND_PASS_CANDIDATE_K=200`（扩大候选池）
2. 提高 `VLM_SECOND_PASS_TOP_K=15`（增加最终页数）

---

## 设计权衡总结

### ✅ 采纳的方案
1. **保守触发**：只在明确"证据不足"时扩展，避免放大成本
2. **直线快路径**：第一轮逻辑不变，不引入 LangGraph 复杂度
3. **降级语义保留**：第二轮失败时保留第一轮结果
4. **additive schema**：新增字段不破坏前端兼容性
5. **可观测性优先**：日志完整记录每轮状态

### ❌ 未采纳的方案
1. **对所有失败重试**：会放大 timeout/error 成本
2. **切回 LangGraph 主链路**：增加复杂度，违背"保持快路径"原则
3. **修改 Milvus schema**：风险高，回滚困难
4. **覆盖第一轮成功答案**：可能降低用户体验

---

## 下一步建议

1. **生产灰度**：先在 10% 流量启用，观察第二轮触发率与成功率
2. **监控指标**：
   - `second_pass_triggered` 比例
   - 第二轮成功改善答案的比例
   - 第二轮平均耗时
3. **A/B 测试**：对比启用/禁用第二轮的答案质量
4. **成本分析**：统计第二轮带来的 VLM API 调用增量

---

## 验证命令

```bash
# 运行回归脚本
cd autodl-tmp/vision_finagent
python regression_second_pass.py

# 预期输出：32 passed, 0 failed
```
