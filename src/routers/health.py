from fastapi import APIRouter
from fastapi.responses import JSONResponse
from ..core.milvus_client import get_client, check_schema_health
from ..core.redis_client import get_redis
from ..services.retrieval_service import is_model_ready, is_cpu_fallback, get_model_device, get_hf_device_map
from ..config import settings
import torch

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/ready")
async def ready():
    errors = {}
    try:
        get_client().list_collections()
    except Exception as e:
        errors["milvus"] = str(e)
    try:
        r = await get_redis()
        await r.ping()
    except Exception as e:
        errors["redis"] = str(e)

    if not is_model_ready():
        device = get_model_device()
        if device is None:
            errors["retrieval_model"] = "ColPali model not loaded"
        elif torch.cuda.is_available():
            hf_map = get_hf_device_map()
            errors["retrieval_model"] = (
                f"ColPali: CUDA available but no layer on GPU "
                f"(first_param={device}, hf_device_map={hf_map})"
            )
        else:
            errors["retrieval_model"] = f"ColPali model not ready (device={device})"

    if errors:
        return JSONResponse(status_code=503, content={"status": "unavailable", "errors": errors})

    # optional 模式下 CPU fallback 允许 ready，但标记 degraded
    if is_cpu_fallback() and not settings.REQUIRE_RETRIEVAL_GPU:
        return JSONResponse(status_code=200, content={
            "status": "ready",
            "retrieval_degraded": True,
            "degraded": True,
            "reason": "cpu_fallback",
            "detail": f"CUDA available but model on CPU (first_param={get_model_device()}, hf_device_map={get_hf_device_map()})",
        })

    return {"status": "ready", "retrieval_degraded": False}


@router.get("/schema-health")
async def schema_health():
    """运维接口：检查两个 collection 的 schema 是否与代码期望一致。"""
    results = check_schema_health()
    drift_detected = any(r.get("missing") or r.get("extra") for r in results)
    return {"drift_detected": drift_detected, "collections": results}
