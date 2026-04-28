import uuid, time, asyncio, structlog, logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from .core.exception import VisionFinAgentException
from .core.milvus_client import connect_milvus, disconnect_milvus, ensure_collection
from .core.redis_client import get_redis, close_redis
from .services.retrieval_service import (
    warmup_retrieval_model, get_model_device, get_hf_device_map,
    is_model_ready, is_cpu_fallback, _has_cuda_placement, _is_lora_adapter_dir,
)
from .routers import health, reports
from .config import settings

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)
logging.getLogger().setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
log = structlog.get_logger()


def _fix_triton_spec():
    """修复 uvicorn worker 线程里 triton.__spec__=None 导致的 DeferredCudaCallError。

    torch 在模块导入时将 _register_triton_kernels 加入 torch.cuda._queued_calls，
    该 callback 调用 importlib.util.find_spec("triton")，而 uvicorn 线程池里
    triton.__spec__ 已被清空，触发 ValueError → DeferredCudaCallError。

    修复策略：
    1. 先修复 triton.__spec__
    2. 再强制触发 torch.cuda._lazy_init()，消费掉 _queued_calls，
       使后续线程池里不再重复执行有问题的 callback。
    """
    try:
        import triton as _triton, importlib as _il
        if _triton.__spec__ is None:
            _triton.__spec__ = _il.util.spec_from_file_location("triton", _triton.__file__)
    except Exception:
        pass
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            _torch.cuda.init()  # 消费 _queued_calls，避免线程池里重复触发
            log.info("startup", msg="cuda pre-initialized, triton spec fixed")
    except Exception as e:
        log.warning("startup", msg=f"cuda pre-init failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    connect_milvus()
    ensure_collection()
    await get_redis()
    import torch
    _fix_triton_spec()
    cuda_available = torch.cuda.is_available()
    log.info("startup", msg="waiting for retrieval warmup", cuda_available=cuda_available)
    try:
        await asyncio.to_thread(warmup_retrieval_model)
    except Exception:
        log.exception("startup", msg="retrieval warmup failed", cuda_available=cuda_available)
        raise
    model_device = get_model_device()
    ready = is_model_ready()
    cpu_fb = is_cpu_fallback()
    log.info("startup", msg="retrieval warmup completed",
             cuda_available=cuda_available,
             explicit_gpu_path=cuda_available,
             is_lora_adapter=_is_lora_adapter_dir(settings.MODEL_PATH),
             hf_device_map=get_hf_device_map(),
             has_cuda_placement=_has_cuda_placement(),
             cpu_fallback=cpu_fb,
             model_device=model_device,
             model_ready=ready)
    log.info("startup", msg="connections ready")
    yield
    disconnect_milvus()
    await close_redis()
    log.info("shutdown", msg="connections released")


app = FastAPI(title="Vision-FinAgent", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def request_middleware(request: Request, call_next):
    rid = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    t0 = time.time()
    response = await call_next(request)
    log.info("request", rid=rid, path=request.url.path, ms=round((time.time()-t0)*1000, 1), status=response.status_code)
    response.headers["X-Request-ID"] = rid
    return response


@app.exception_handler(VisionFinAgentException)
async def agent_exc_handler(request: Request, exc: VisionFinAgentException):
    return JSONResponse(status_code=400, content={"error_code": exc.error_code, "detail": exc.detail})


app.include_router(health.router)
app.include_router(reports.router)

_static = Path(__file__).resolve().parent.parent / "static"
if _static.exists():
    app.mount("/", StaticFiles(directory=str(_static), html=True), name="static")
