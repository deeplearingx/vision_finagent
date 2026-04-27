from fastapi import APIRouter
from fastapi.responses import JSONResponse
from ..core.milvus_client import get_client, check_schema_health
from ..core.redis_client import get_redis
from ..services.retrieval_service import is_model_ready

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
        errors["retrieval_model"] = "ColPali model is still warming up"
    if errors:
        return JSONResponse(status_code=503, content={"status": "unavailable", "errors": errors})
    return {"status": "ready"}


@router.get("/schema-health")
async def schema_health():
    """运维接口：检查两个 collection 的 schema 是否与代码期望一致。"""
    results = check_schema_health()
    drift_detected = any(r.get("missing") or r.get("extra") for r in results)
    return {"drift_detected": drift_detected, "collections": results}
