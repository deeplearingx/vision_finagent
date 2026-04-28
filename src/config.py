from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MILVUS_LITE_PATH = str(PROJECT_ROOT / "milvus_local.db")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    MILVUS_URI: str = DEFAULT_MILVUS_LITE_PATH  # local file mode; set to "http://localhost:19530" for standalone
    MILVUS_COLLECTION: str = "fin_vision_reports"

    REDIS_URL: str = "redis://localhost:6379/0"

    MODEL_PATH: str = "vidore/colpali-v1.2"
    # 若 MODEL_PATH 是 LoRA adapter 目录（含 adapter_config.json 但无 config.json），
    # 则必须设置 BASE_MODEL_PATH 指向完整 base model 目录。
    # 留空表示 MODEL_PATH 本身是完整模型（直接加载，无 LoRA）。
    BASE_MODEL_PATH: str = ""
    VLM_API_BASE: str = "https://ark.cn-beijing.volces.com/api/coding/v3"
    VLM_MODEL: str = "Kimi-K2.6"
    VLM_API_KEY: str = "EMPTY"

    REQUIRE_RETRIEVAL_GPU: bool = False  # False = allow CPU fallback; True = fail if no CUDA placement

    MAX_BATCH_SIZE: int = 4
    LOG_LEVEL: str = "INFO"
    IDEMPOTENCY_TTL: int = 86400  # seconds
    VLM_TIMEOUT: int = 60          # seconds for VLM API call (OpenAI client level)
    VLM_QUERY_TIMEOUT: int = 20    # seconds for VLM call inside query pipeline
    MAX_VLM_IMAGES: int = 4        # max images sent to VLM per query
    INGEST_TIMEOUT: int = 300      # seconds for ingestion task
    INGEST_WORKERS: int = 2        # background thread pool for ingestion
    QUERY_TIMEOUT: int = 15        # seconds for retrieval API call


settings = Settings()
