import json
import time
import structlog
from typing import List

# 双重保障：start.py 顶部是 first-line 补丁；此处是 fallback，
# 防止通过其他入口（如直接 uvicorn CLI）启动时 triton.__spec__ 仍为 None。
try:
    import triton as _triton
    if _triton.__spec__ is None:
        import importlib as _il
        _triton.__spec__ = _il.util.spec_from_file_location("triton", _triton.__file__)
except Exception:
    pass

import torch
import numpy as np

log = structlog.get_logger()
from colpali_engine.models.paligemma.colpali.modeling_colpali import ColPali
from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor
from ..config import settings
from ..core.milvus_client import get_client, get_collection_name, get_pages_collection_name
from ..models.schema import PageResult

_model: ColPali | None = None
_processor: ColPaliProcessor | None = None


def get_model_device() -> str | None:
    """返回模型第一个参数所在设备字符串，模型未加载时返回 None。"""
    if _model is None:
        return None
    try:
        return str(next(_model.parameters()).device)
    except StopIteration:
        return "unknown"


def get_hf_device_map() -> dict | None:
    """返回 accelerate 注入的 hf_device_map，不存在时返回 None。"""
    if _model is None:
        return None
    return getattr(_model, "hf_device_map", None)


def _has_cuda_placement() -> bool:
    """检查模型是否有任何层/参数实际放置在 CUDA 上。

    兼容两种场景：
    1. 单卡全量加载：首参数 device 为 cuda:N
    2. accelerate device_map="auto" 分层加载：hf_device_map 中存在 cuda 设备
    """
    if _model is None:
        return False
    hf_map = getattr(_model, "hf_device_map", None)
    if hf_map:
        return any(str(v).startswith("cuda") for v in hf_map.values())
    device = get_model_device()
    return device is not None and device.startswith("cuda")


def _get_input_device() -> str:
    """解析推理输入 batch 应放置的目标设备。

    accelerate device_map 分层时 model.device 不可靠（可能返回 cpu 的 embedding 层），
    需从 hf_device_map 中找第一个 cuda 设备作为输入目标。

    优先级：
    1. hf_device_map 中第一个 cuda 设备（accelerate 分层场景）
    2. 首参数设备（单卡全量加载）
    3. "cpu"（CPU-only fallback）
    """
    if _model is None:
        return "cpu"
    hf_map = getattr(_model, "hf_device_map", None)
    if hf_map:
        for v in hf_map.values():
            if str(v).startswith("cuda"):
                return str(v)
    device = get_model_device()
    return device if device else "cpu"


def is_cpu_fallback() -> bool:
    """CUDA 可用但模型全在 CPU 上（降级模式）。"""
    return _model is not None and torch.cuda.is_available() and not _has_cuda_placement()


def is_model_ready() -> bool:
    """模型已加载且满足设备要求。

    strict 模式（REQUIRE_RETRIEVAL_GPU=true）：CUDA 可用时必须有 CUDA 放置。
    optional 模式（REQUIRE_RETRIEVAL_GPU=false）：CPU fallback 也视为 ready。
    """
    if _model is None:
        return False
    if torch.cuda.is_available() and not _has_cuda_placement():
        return not settings.REQUIRE_RETRIEVAL_GPU
    return True


def _is_lora_adapter_dir(path: str) -> bool:
    """判断路径是否为 LoRA adapter 目录（有 adapter_config.json 但无 config.json）。"""
    import os
    return (os.path.isfile(os.path.join(path, "adapter_config.json"))
            and not os.path.isfile(os.path.join(path, "config.json")))


def _load_model():
    global _model, _processor
    if _model is not None:
        return _model, _processor

    cuda_available = torch.cuda.is_available()
    is_lora = _is_lora_adapter_dir(settings.MODEL_PATH)
    base_path = settings.BASE_MODEL_PATH if settings.BASE_MODEL_PATH else settings.MODEL_PATH

    log.info("retrieval.model_loading",
             model_path=settings.MODEL_PATH,
             base_model_path=base_path,
             is_lora_adapter=is_lora,
             cuda_available=cuda_available)

    if cuda_available:
        # GPU strict 路径：用 device_map="cuda:0" 直接在 from_pretrained 里指定设备，
        # 避免 .to("cuda:0") 触发 uvicorn worker 里 triton.__spec__=None 的 DeferredCudaCallError
        try:
            _model = ColPali.from_pretrained(
                base_path, torch_dtype=torch.bfloat16, device_map="cuda:0"
            ).eval()

            if is_lora:
                from peft import PeftModel
                _model = PeftModel.from_pretrained(_model, settings.MODEL_PATH).eval()

            explicit_gpu = True
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            log.error("retrieval.gpu_load_failed",
                      error=str(exc),
                      msg="Explicit cuda:0 load failed — not falling back (GPU server requires GPU)")
            raise RuntimeError(
                f"Failed to load model on cuda:0: {exc}. "
                "Check GPU memory. Set REQUIRE_RETRIEVAL_GPU=false to allow CPU fallback."
            ) from exc
    else:
        # CPU-only 环境
        explicit_gpu = False
        _model = ColPali.from_pretrained(
            base_path, torch_dtype=torch.float32
        ).eval()
        if is_lora:
            from peft import PeftModel
            _model = PeftModel.from_pretrained(_model, settings.MODEL_PATH).eval()

    _processor = ColPaliProcessor.from_pretrained(settings.MODEL_PATH)

    first_param_device = get_model_device()
    hf_device_map = getattr(_model, "hf_device_map", None)
    on_cuda = _has_cuda_placement()

    log.info("retrieval.model_placement",
             cuda_available=cuda_available,
             explicit_gpu_path=explicit_gpu,
             first_param_device=first_param_device,
             hf_device_map=hf_device_map,
             has_cuda_placement=on_cuda,
             is_lora_adapter=is_lora)

    if cuda_available and not on_cuda:
        if settings.REQUIRE_RETRIEVAL_GPU:
            log.error("retrieval.model_device_mismatch",
                      cuda_available=True,
                      first_param_device=first_param_device,
                      hf_device_map=hf_device_map,
                      require_gpu=True,
                      msg="CUDA available but no layer on GPU — failing (REQUIRE_RETRIEVAL_GPU=true)")
            raise RuntimeError(
                f"CUDA is available but no model layer is on CUDA "
                f"(first_param={first_param_device}, hf_device_map={hf_device_map}). "
                "Check GPU memory or device_map config."
            )
        else:
            log.warning("retrieval.model_cpu_fallback",
                        cuda_available=True,
                        first_param_device=first_param_device,
                        hf_device_map=hf_device_map,
                        require_gpu=False,
                        msg="CUDA available but model on CPU — degraded mode (REQUIRE_RETRIEVAL_GPU=false)")
    elif not cuda_available:
        log.warning("retrieval.model_cpu_fallback",
                    cuda_available=False,
                    first_param_device=first_param_device,
                    msg="CUDA not available, running on CPU (degraded performance)")
    else:
        log.info("retrieval.model_ready",
                 model_path=settings.MODEL_PATH,
                 first_param_device=first_param_device,
                 hf_device_map=hf_device_map,
                 cuda_available=True,
                 explicit_gpu_path=explicit_gpu)

    return _model, _processor


def warmup_retrieval_model() -> None:
    _load_model()


def _maxsim_score(q_vecs: np.ndarray, page_vecs: np.ndarray) -> float:
    """Σ_i max_j dot(q_i, p_j)  — standard ColPali late-interaction score."""
    # q_vecs: (nq, 128), page_vecs: (np, 128)
    scores = q_vecs @ page_vecs.T  # (nq, np)
    return float(scores.max(axis=1).sum())


def _build_company_filter(target_companies: List[str]) -> str | None:
    """Build a Milvus filter expression for company pre-filtering.

    Semantics preserved from post-filter:  rid.startswith(t + "_") or rid == t
    Milvus supports `like` for prefix matching and `==` for exact match.
    Multiple companies are joined with `or`.

    Returns None when target_companies is empty (no filter needed).
    """
    if not target_companies:
        return None
    clauses: list[str] = []
    for t in target_companies:
        import json as _json
        # prefix match: report_id starts with "<ticker>_"
        prefix = t + "_"
        clauses.append(f'report_id like {_json.dumps(prefix + "%")}')
        # exact match: report_id == "<ticker>"
        clauses.append(f'report_id == {_json.dumps(t)}')
    return "(" + " or ".join(clauses) + ")"


def retrieve(
    query: str,
    target_companies: List[str],
    top_k: int = 5,   # aligned with frontend default Top-5
    candidate_k: int = 50,
    pass_label: str = "first_pass",  # observability: "first_pass" | "second_pass"
    report_ids: List[str] | None = None,
) -> List[PageResult]:
    import threading
    log.info("retrieval.entered", pass_label=pass_label, thread=threading.current_thread().name, query=query[:80])
    model, processor = _load_model()

    input_device = _get_input_device()
    log.info("retrieval.query_device", input_device=input_device,
             hf_device_map=getattr(model, "hf_device_map", None))

    # Stage 0: query embedding
    t0 = time.monotonic()
    with torch.no_grad():
        batch = processor.process_queries([query]).to(input_device)
        q_emb = model(**batch)  # (1, nq, 128)
    q_vecs = q_emb[0].cpu().float().numpy()  # (nq, 128)
    # Mean-pool query vectors for ANN search to reduce Milvus round-trips
    # (nq vectors → 1 vector; MaxSim re-ranking still uses full q_vecs)
    q_vecs_search = q_vecs.mean(axis=0, keepdims=True)  # (1, 128)
    log.info("retrieval.timing", stage="query_embedding", elapsed=round(time.monotonic() - t0, 3),
             pass_label=pass_label, nq=len(q_vecs))

    client = get_client()
    name = get_collection_name()

    # report_ids takes priority over target_companies for exact filtering
    if report_ids:
        import json as _json
        clauses = [f'report_id == {_json.dumps(r)}' for r in report_ids]
        milvus_filter: str | None = "(" + " or ".join(clauses) + ")"
        log.info("retrieval.report_ids_filter", filter=milvus_filter, report_ids=report_ids)
    else:
        milvus_filter = _build_company_filter(target_companies)
        log.info("retrieval.company_filter", filter=milvus_filter,
                 target_companies=target_companies)

    search_kwargs: dict = dict(
        collection_name=name,
        anns_field="colpali_embeddings",
        search_params={"metric_type": "IP", "params": {"nprobe": 10}},
        limit=candidate_k,
        output_fields=["report_id", "page_num"],
    )
    if milvus_filter:
        search_kwargs["filter"] = milvus_filter

    # Stage 1: ANN search
    t1 = time.monotonic()
    _NQ_BATCH = settings.MILVUS_NQ_BATCH
    all_vecs = q_vecs.tolist()
    search_vecs = q_vecs_search.tolist()  # 1 mean-pooled vector
    log.info("retrieval.ann_start", total_vecs=len(all_vecs), search_vecs=len(search_vecs), pass_label=pass_label)
    results = []
    for i in range(0, len(search_vecs), _NQ_BATCH):
        tb = time.monotonic()
        results.extend(client.search(data=search_vecs[i:i + _NQ_BATCH], **search_kwargs))
        log.info("retrieval.ann_batch", batch=i // _NQ_BATCH, elapsed=round(time.monotonic() - tb, 3), pass_label=pass_label)
    log.info("retrieval.timing", stage="milvus_ann_search", elapsed=round(time.monotonic() - t1, 3), pass_label=pass_label)

    candidates: set[tuple[str, int]] = set()
    for hits in results:
        for hit in hits:
            rid = hit["entity"]["report_id"]
            if report_ids and rid not in report_ids:
                continue
            if not report_ids and target_companies and not any(rid.startswith(t + "_") or rid == t for t in target_companies):
                continue
            candidates.add((rid, hit["entity"]["page_num"]))

    log.info("retrieval.candidates", count=len(candidates), query=query,
             filter_applied=milvus_filter is not None, pass_label=pass_label)
    if not candidates:
        log.warning("retrieval.no_candidates", query=query)
        return []

    # Stage 2a: fetch patch vectors
    from collections import defaultdict
    rid_to_pnums: dict[str, list[int]] = defaultdict(list)
    for r, p in candidates:
        rid_to_pnums[r].append(p)

    _MILVUS_QUERY_LIMIT = 16384
    t2 = time.monotonic()
    patch_rows: list = []
    for r, pnums in rid_to_pnums.items():
        for pnum in pnums:
            clause = f"(report_id == {json.dumps(r)} and page_num == {pnum})"
            patch_rows.extend(client.query(
                collection_name=name,
                filter=clause,
                output_fields=["report_id", "page_num", "colpali_embeddings"],
                limit=_MILVUS_QUERY_LIMIT,
            ))
    log.info("retrieval.timing", stage="patch_vector_query", elapsed=round(time.monotonic() - t2, 3), pass_label=pass_label)

    pages: dict[tuple[str, int], list] = {}
    for row in patch_rows:
        key = (row["report_id"], row["page_num"])
        if key not in candidates:
            continue
        pages.setdefault(key, []).append(row["colpali_embeddings"])

    # Stage 2b: fetch page metadata
    t3 = time.monotonic()
    page_ids = [f"{r}_{p}" for r, p in pages]
    meta_rows = client.query(
        collection_name=get_pages_collection_name(),
        filter=f"page_id in {json.dumps(page_ids)}",
        output_fields=["report_id", "page_num", "image_base64"],
        limit=len(page_ids) + 1,
    )
    log.info("retrieval.timing", stage="page_meta_query", elapsed=round(time.monotonic() - t3, 3), pass_label=pass_label)
    page_meta: dict[tuple[str, int], str] = {
        (r["report_id"], r["page_num"]): r["image_base64"] for r in meta_rows
    }

    # Score each page with MaxSim
    scored: List[PageResult] = []
    for (rid, pnum), vecs in pages.items():
        b64 = page_meta.get((rid, pnum), "")
        page_vecs = np.array(vecs, dtype=np.float32)
        score = _maxsim_score(q_vecs, page_vecs)
        scored.append(PageResult(
            report_id=rid,
            page_num=pnum,
            image_base64=b64,
            maxsim_score=score,
        ))

    scored.sort(key=lambda x: x.maxsim_score, reverse=True)
    result = scored[:top_k]
    log.info("retrieval.done", returned=len(result), pass_label=pass_label,
             top_score=result[0].maxsim_score if result else None)
    return result
