import json
import re
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
from ..utils.image import encode_image_file_for_vlm
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


def _select_search_vectors(
    q_vecs: np.ndarray,
    has_filter: bool,
) -> tuple[np.ndarray, bool]:
    """Return (search_vecs, use_multi_vec).

    Degrades to mean-pool when there is no filter (full-collection scan would be too expensive).
    """
    if not has_filter or settings.RETRIEVAL_USE_MEAN_POOL_ANN:
        return q_vecs.mean(axis=0, keepdims=True), False
    max_vecs = settings.RETRIEVAL_MAX_QUERY_VECS
    if len(q_vecs) > max_vecs:
        idx = np.linspace(0, len(q_vecs) - 1, max_vecs, dtype=int)
        return q_vecs[idx], True
    return q_vecs, True


def retrieve(
    query: str,
    target_companies: List[str],
    top_k: int = 5,
    candidate_k: int = 80,
    pass_label: str = "first_pass",
    report_ids: List[str] | None = None,
) -> List[PageResult]:
    import threading
    from collections import defaultdict
    total_t0 = time.monotonic()
    log.info("retrieval.start", pass_label=pass_label,
             thread=threading.current_thread().name, query=query[:80],
             top_k=top_k, candidate_k=candidate_k,
             report_ids=report_ids, target_companies=target_companies)

    model, processor = _load_model()
    input_device = _get_input_device()

    # Stage 0: query embedding
    t0 = time.monotonic()
    with torch.no_grad():
        batch = processor.process_queries([query]).to(input_device)
        q_emb = model(**batch)
    q_vecs = q_emb[0].cpu().float().numpy()  # (nq, 128)
    log.info("retrieval.query_encoded", nq=len(q_vecs), dim=q_vecs.shape[-1],
             elapsed=round(time.monotonic() - t0, 3), pass_label=pass_label)

    client = get_client()
    name = get_collection_name()

    # Build Milvus filter
    if report_ids:
        clauses = [f'report_id == {json.dumps(r)}' for r in report_ids]
        milvus_filter: str | None = "(" + " or ".join(clauses) + ")"
    else:
        milvus_filter = _build_company_filter(target_companies)
    has_filter = milvus_filter is not None
    log.info("retrieval.filter", filter=milvus_filter, has_filter=has_filter, pass_label=pass_label)

    # Select search vectors (degrade to mean-pool when no filter)
    search_vecs_np, use_multi_vec = _select_search_vectors(q_vecs, has_filter)
    per_vec_limit = settings.RETRIEVAL_PER_VEC_LIMIT if use_multi_vec else candidate_k
    log.info("retrieval.search_vecs", use_multi_vec=use_multi_vec,
             search_vecs=len(search_vecs_np), per_vec_limit=per_vec_limit, pass_label=pass_label)

    search_kwargs: dict = dict(
        collection_name=name,
        anns_field="colpali_embeddings",
        search_params={"metric_type": "IP", "params": {"nprobe": 10}},
        limit=per_vec_limit,
        output_fields=["report_id", "page_num"],
    )
    if milvus_filter:
        search_kwargs["filter"] = milvus_filter

    # Stage 1: ANN search (batched by MILVUS_NQ_BATCH)
    ann_t0 = time.monotonic()
    search_vecs = search_vecs_np.tolist()
    _NQ_BATCH = settings.MILVUS_NQ_BATCH
    page_ann_stats: dict[tuple[str, int], dict] = defaultdict(
        lambda: {"hit_count": 0, "best_score": float("-inf")}
    )
    for batch_i in range(0, len(search_vecs), _NQ_BATCH):
        tb = time.monotonic()
        vec_batch = search_vecs[batch_i: batch_i + _NQ_BATCH]
        batch_results = client.search(data=vec_batch, **search_kwargs)
        for one_query_hits in batch_results:
            for hit in one_query_hits:
                entity = hit.get("entity") or {}
                rid = entity.get("report_id")
                pnum = entity.get("page_num")
                if rid is None or pnum is None:
                    continue
                # post-filter (safety net when Milvus filter is absent)
                if report_ids and rid not in report_ids:
                    continue
                if not report_ids and target_companies and not any(
                    rid.startswith(t + "_") or rid == t for t in target_companies
                ):
                    continue
                key = (rid, int(pnum))
                score = float(hit.get("distance", 0.0))
                page_ann_stats[key]["hit_count"] += 1
                page_ann_stats[key]["best_score"] = max(page_ann_stats[key]["best_score"], score)
        log.info("retrieval.ann_batch", batch=batch_i // _NQ_BATCH,
                 nq=len(vec_batch), unique_pages=len(page_ann_stats),
                 elapsed=round(time.monotonic() - tb, 3), pass_label=pass_label)

    log.info("retrieval.ann_done", unique_pages=len(page_ann_stats),
             elapsed=round(time.monotonic() - ann_t0, 3), pass_label=pass_label)

    if not page_ann_stats:
        log.warning("retrieval.no_candidates", query=query[:80], pass_label=pass_label)
        return []

    # Truncate candidates: sort by (hit_count desc, best_score desc), cap at MAX_CANDIDATE_PAGES
    candidate_items = sorted(
        page_ann_stats.items(),
        key=lambda kv: (kv[1]["hit_count"], kv[1]["best_score"]),
        reverse=True,
    )[:settings.RETRIEVAL_MAX_CANDIDATE_PAGES]
    candidate_pages = [key for key, _ in candidate_items]
    log.info("retrieval.candidate_pages_selected",
             before=len(page_ann_stats), after=len(candidate_pages),
             cap=settings.RETRIEVAL_MAX_CANDIDATE_PAGES, pass_label=pass_label)

    # Stage 2a: fetch patch vectors for rerank candidates only
    rerank_pages = candidate_pages[:settings.RETRIEVAL_RERANK_PAGE_CAP]
    _MILVUS_QUERY_LIMIT = 16384
    t2 = time.monotonic()
    patch_rows: list = []
    rid_to_pnums: dict[str, list[int]] = defaultdict(list)
    for r, p in rerank_pages:
        rid_to_pnums[r].append(p)
    for r, pnums in rid_to_pnums.items():
        for pnum in pnums:
            clause = f"(report_id == {json.dumps(r)} and page_num == {pnum})"
            patch_rows.extend(client.query(
                collection_name=name,
                filter=clause,
                output_fields=["report_id", "page_num", "colpali_embeddings"],
                limit=_MILVUS_QUERY_LIMIT,
            ))
    log.info("retrieval.timing", stage="patch_vector_query",
             pages=len(rerank_pages), rows=len(patch_rows),
             elapsed=round(time.monotonic() - t2, 3), pass_label=pass_label)

    rerank_set = set(rerank_pages)
    pages_vecs: dict[tuple[str, int], list] = {}
    for row in patch_rows:
        key = (row["report_id"], row["page_num"])
        if key not in rerank_set:
            continue
        pages_vecs.setdefault(key, []).append(row["colpali_embeddings"])

    # Stage 2b: fetch page metadata (batched to avoid gRPC 4MB limit)
    t3 = time.monotonic()
    page_ids = [f"{r}_{p}" for r, p in pages_vecs]
    _META_BATCH_SIZE = 30  # each page ~40KB base64; 30 pages ≈ 1.2MB < 4MB limit
    meta_rows: list = []
    for i in range(0, len(page_ids), _META_BATCH_SIZE):
        batch_ids = page_ids[i : i + _META_BATCH_SIZE]
        batch_rows = client.query(
            collection_name=get_pages_collection_name(),
            filter=f"page_id in {json.dumps(batch_ids)}",
            output_fields=["report_id", "page_num", "image_base64", "image_path", "page_text"],
            limit=len(batch_ids) + 1,
        )
        meta_rows.extend(batch_rows)
    log.info("retrieval.timing", stage="page_meta_query",
             pages_queried=len(page_ids), batches=(len(page_ids) + _META_BATCH_SIZE - 1) // _META_BATCH_SIZE,
             elapsed=round(time.monotonic() - t3, 3), pass_label=pass_label)

    page_meta: dict[tuple[str, int], str] = {}
    page_texts: dict[tuple[str, int], str] = {}
    for r in meta_rows:
        key = (r["report_id"], r["page_num"])
        page_texts[key] = r.get("page_text", "")
        image_path = r.get("image_path")
        if image_path:
            try:
                page_meta[key] = encode_image_file_for_vlm(image_path)
                continue
            except Exception as exc:
                log.warning("retrieval.highres_image_load_failed",
                            report_id=r["report_id"], page_num=r["page_num"],
                            image_path=image_path, error=str(exc))
        page_meta[key] = r.get("image_base64", "")

    # Stage 3: MaxSim rerank
    rerank_t0 = time.monotonic()
    log.info("retrieval.rerank_start", pages=len(pages_vecs), pass_label=pass_label)
    scored: List[PageResult] = []
    for (rid, pnum), vecs in pages_vecs.items():
        b64 = page_meta.get((rid, pnum), "")
        page_vecs = np.array(vecs, dtype=np.float32)
        score = _maxsim_score(q_vecs, page_vecs)
        scored.append(PageResult(report_id=rid, page_num=pnum, image_base64=b64, maxsim_score=score, page_text=page_texts.get((rid, pnum), "")))

    scored.sort(key=lambda x: x.maxsim_score, reverse=True)

    # Stage 3b: Keyword-weighted re-rank (conservative boost)
    # Only boost pages already in the visual candidate set — do NOT pull in
    # new pages from keyword matching alone, as that introduces noise.
    # Visual rank weight is 3x keyword rank weight to preserve ColPali signal.
    query_tokens = set(re.findall(r'[a-zA-Z0-9]+', query.lower()))
    _STOP_WORDS = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                   "have", "has", "had", "do", "does", "did", "will", "would", "could",
                   "should", "may", "might", "shall", "can", "need", "dare", "ought",
                   "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
                   "as", "into", "through", "during", "before", "after", "above", "below",
                   "between", "out", "off", "over", "under", "again", "further", "then",
                   "once", "what", "which", "who", "whom", "this", "that", "these", "those",
                   "and", "but", "or", "nor", "not", "so", "yet", "both", "either", "neither",
                   "each", "every", "all", "any", "few", "more", "most", "other", "some",
                   "such", "no", "only", "own", "same", "than", "too", "very", "just",
                   "how", "where", "when", "while", "about", "against", "it", "its"}
    query_keywords = {t for t in query_tokens if len(t) >= 3 and t not in _STOP_WORDS}

    if query_keywords and page_texts:
        # Compute keyword hit count per page (only for pages already in scored list)
        keyword_scores: dict[tuple[str, int], int] = {}
        for s in scored:
            key = (s.report_id, s.page_num)
            text = page_texts.get(key, "")
            if not text:
                continue
            text_lower = text.lower()
            hits = sum(1 for kw in query_keywords if kw in text_lower)
            if hits > 0:
                keyword_scores[key] = hits

        if keyword_scores:
            # Rank by keyword hits (descending) among visual candidates only
            keyword_ranked = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
            keyword_rank_map = {key: rank for rank, (key, _) in enumerate(keyword_ranked, 1)}

            # Weighted RRF: visual weight=3, keyword weight=1
            # Pages without keyword hits keep their original visual order
            _RRF_K = 60
            _VIS_WEIGHT = 3.0
            _KW_WEIGHT = 1.0
            n_scored = len(scored)
            rrf_scores: dict[tuple[str, int], float] = {}
            for i, s in enumerate(scored):
                key = (s.report_id, s.page_num)
                v_rank = i + 1
                k_rank = keyword_rank_map.get(key, n_scored + 1)
                rrf_scores[key] = _VIS_WEIGHT / (_RRF_K + v_rank) + _KW_WEIGHT / (_RRF_K + k_rank)

            scored.sort(key=lambda s: rrf_scores.get((s.report_id, s.page_num), 0), reverse=True)
            log.info("retrieval.keyword_boost", query_keywords=len(query_keywords),
                     keyword_hits=len(keyword_scores), boost_applied=True, pass_label=pass_label)
        else:
            log.info("retrieval.keyword_boost", query_keywords=len(query_keywords),
                     keyword_hits=0, boost_applied=False, pass_label=pass_label)
    else:
        log.info("retrieval.keyword_boost", query_keywords=len(query_keywords),
                 boost_applied=False, pass_label=pass_label)

    result = scored[:top_k]
    log.info("retrieval.rerank_done", reranked=len(scored), returned=len(result),
             elapsed=round(time.monotonic() - rerank_t0, 3), pass_label=pass_label)
    log.info("retrieval.done", returned=len(result),
             top_score=result[0].maxsim_score if result else None,
             total_elapsed=round(time.monotonic() - total_t0, 3), pass_label=pass_label)
    return result
