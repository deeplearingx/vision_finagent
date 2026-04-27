import json
import structlog
from typing import List
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


def is_model_ready() -> bool:
    return _model is not None


def _load_model():
    global _model, _processor
    if _model is None:
        log.info("retrieval.model_loading", model_path=settings.MODEL_PATH)
        _model = ColPali.from_pretrained(
            settings.MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
        ).eval()
        _processor = ColPaliProcessor.from_pretrained(settings.MODEL_PATH)
        log.info("retrieval.model_ready", model_path=settings.MODEL_PATH)
    return _model, _processor


def warmup_retrieval_model() -> None:
    _load_model()


def _maxsim_score(q_vecs: np.ndarray, page_vecs: np.ndarray) -> float:
    """Σ_i max_j dot(q_i, p_j)  — standard ColPali late-interaction score."""
    # q_vecs: (nq, 128), page_vecs: (np, 128)
    scores = q_vecs @ page_vecs.T  # (nq, np)
    return float(scores.max(axis=1).sum())


def retrieve(
    query: str,
    target_companies: List[str],
    top_k: int = 3,
    candidate_k: int = 50,
) -> List[PageResult]:
    model, processor = _load_model()

    with torch.no_grad():
        batch = processor.process_queries([query]).to(model.device)
        q_emb = model(**batch)  # (1, nq, 128)

    q_vecs = q_emb[0].cpu().float().numpy()  # (nq, 128) — no mean pooling

    client = get_client()
    name = get_collection_name()
    # Stage 1: per-query-token ANN recall → candidate pages
    results = client.search(
        collection_name=name,
        data=q_vecs.tolist(),
        anns_field="colpali_embeddings",
        search_params={"metric_type": "IP", "params": {"nprobe": 10}},
        limit=candidate_k,
        output_fields=["report_id", "page_num"],
    )

    # Collect unique (report_id, page_num) candidates, applying ticker filter post-search
    candidates: set[tuple[str, int]] = set()
    for hits in results:
        for hit in hits:
            rid = hit["entity"]["report_id"]
            if target_companies and not any(rid.startswith(t + "_") or rid == t for t in target_companies):
                continue
            candidates.add((rid, hit["entity"]["page_num"]))

    log.info("retrieval.candidates", count=len(candidates), query=query)
    if not candidates:
        log.warning("retrieval.no_candidates", query=query)
        return []

    # Stage 2a: fetch patch vectors (no image_base64)
    # Build per-report filter to avoid cross-product false positives
    # e.g. report_A page 3 and report_B page 3 are different pages
    from collections import defaultdict
    rid_to_pnums: dict[str, list[int]] = defaultdict(list)
    for r, p in candidates:
        rid_to_pnums[r].append(p)

    clauses = [
        f"(report_id == {json.dumps(r)} and page_num in {json.dumps(pnums)})"
        for r, pnums in rid_to_pnums.items()
    ]
    filter_expr = " or ".join(clauses)

    patch_rows = client.query(
        collection_name=name,
        filter=filter_expr,
        output_fields=["report_id", "page_num", "colpali_embeddings"],
        limit=len(candidates) * 1024 + 1,
    )

    # Group patch vectors by page
    pages: dict[tuple[str, int], list] = {}
    for row in patch_rows:
        key = (row["report_id"], row["page_num"])
        if key not in candidates:
            continue
        pages.setdefault(key, []).append(row["colpali_embeddings"])

    # Stage 2b: fetch page metadata (image_base64) from pages collection
    page_ids = [f"{r}_{p}" for r, p in pages]
    meta_rows = client.query(
        collection_name=get_pages_collection_name(),
        filter=f"page_id in {json.dumps(page_ids)}",
        output_fields=["report_id", "page_num", "image_base64"],
        limit=len(page_ids) + 1,
    )
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
    log.info("retrieval.done", returned=len(result),
             top_score=result[0].maxsim_score if result else None)
    return result
