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
from .services.retrieval_service import warmup_retrieval_model
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    connect_milvus()
    ensure_collection()
    await get_redis()
    app.state.retrieval_warmup_task = asyncio.create_task(asyncio.to_thread(warmup_retrieval_model))
    log.info("startup", msg="retrieval warmup scheduled")
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
