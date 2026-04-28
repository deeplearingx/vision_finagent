import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MILVUS_LITE_PATH = str(PROJECT_ROOT / "milvus_local.db")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    MILVUS_URI: str = DEFAULT_MILVUS_LITE_PATH  # local file mode; set to "http://localhost:19530" for standalone
    MILVUS_COLLECTION: str = "fin_vision_reports"

    REDIS_URL: str = "redis://localhost:6379/0"

    # MODEL_PATH: LoRA adapter 目录（含 adapter_config.json 但无 config.json）时，
    #   必须同时设置 BASE_MODEL_PATH 指向完整 base model 目录，否则启动 fail-fast。
    # MODEL_PATH: 完整模型目录（含 config.json）时，BASE_MODEL_PATH 留空即可。
    MODEL_PATH: str = "vidore/colpali-v1.2"
    BASE_MODEL_PATH: str = ""

    VLM_API_BASE: str = "https://ark.cn-beijing.volces.com/api/coding/v3"
    VLM_MODEL: str = "Kimi-K2.6"
    VLM_API_KEY: str = "EMPTY"

    REQUIRE_RETRIEVAL_GPU: bool = False  # False = allow CPU fallback; True = fail if no CUDA placement

    MAX_BATCH_SIZE: int = 4
    LOG_LEVEL: str = "INFO"
    IDEMPOTENCY_TTL: int = 86400  # seconds
    # VLM_TIMEOUT: OpenAI client 级别超时（连接+响应），应 >= VLM_QUERY_TIMEOUT
    VLM_TIMEOUT: int = 120
    # VLM_QUERY_TIMEOUT: asyncio.wait_for 层超时，必须 < VLM_TIMEOUT 才有意义
    VLM_QUERY_TIMEOUT: int = 90
    # MAX_VLM_IMAGES: images sent to VLM per query.
    # Must match frontend default top_k (currently 5) so retrieved pages are
    # not silently truncated before VLM sees them.  If you change the frontend
    # default, update this value in sync.
    MAX_VLM_IMAGES: int = 5        # aligned with frontend default top_k=5

    # Second-pass VLM evidence expansion (conservative two-round mode)
    VLM_SECOND_PASS_ENABLED: bool = True
    VLM_SECOND_PASS_TOP_K: int = 10       # expanded top_k for second retrieval
    VLM_SECOND_PASS_CANDIDATE_K: int = 100  # expanded candidate_k for second retrieval
    VLM_SECOND_PASS_MAX_IMAGES: int = 10  # max images sent to VLM in second pass

    # VLM outbound image compression (applied only before sending to VLM, not stored)
    VLM_IMG_MAX_SIDE: int = 1024   # resize longest side to this (pixels); 0 = no resize
    VLM_IMG_JPEG_QUALITY: int = 75 # JPEG re-encode quality 1-95; 0 = skip re-encode
    VLM_IMG_MAX_BYTES: int = 0     # hard byte budget per image (0 = disabled)

    # VLM connection-level retry (only for transient network errors, not timeouts/empty responses)
    VLM_RETRY_ENABLED: bool = True
    VLM_RETRY_MAX_ATTEMPTS: int = 2   # total attempts (1 original + 1 retry)
    VLM_RETRY_BACKOFF_SECONDS: float = 1.0  # sleep between attempts

    INGEST_TIMEOUT: int = 300      # seconds for ingestion task
    INGEST_WORKERS: int = 2        # background thread pool for ingestion
    QUERY_TIMEOUT: int = 15        # seconds for retrieval API call

    def validate_model_paths(self) -> None:
        """Fail fast if MODEL_PATH is a LoRA adapter dir but BASE_MODEL_PATH is not set."""
        is_lora = (
            os.path.isfile(os.path.join(self.MODEL_PATH, "adapter_config.json"))
            and not os.path.isfile(os.path.join(self.MODEL_PATH, "config.json"))
        )
        if is_lora and not self.BASE_MODEL_PATH:
            raise ValueError(
                f"MODEL_PATH '{self.MODEL_PATH}' looks like a LoRA adapter directory "
                "(has adapter_config.json but no config.json). "
                "You must set BASE_MODEL_PATH to the full base model directory in .env."
            )
        if self.VLM_QUERY_TIMEOUT >= self.VLM_TIMEOUT:
            raise ValueError(
                f"VLM_QUERY_TIMEOUT ({self.VLM_QUERY_TIMEOUT}s) must be less than "
                f"VLM_TIMEOUT ({self.VLM_TIMEOUT}s). "
                "asyncio.wait_for cancels at VLM_QUERY_TIMEOUT; "
                "the OpenAI client must have a larger budget to avoid orphaned connections."
            )


settings = Settings()
# Run path/timeout consistency checks at import time so misconfiguration is caught on startup.
settings.validate_model_paths()
