from enum import Enum


class ReportStatus(str, Enum):
    PENDING = "pending"
    INGESTING = "ingesting"
    READY = "ready"
    FAILED = "failed"


class TaskType(str, Enum):
    INGEST = "ingest"
    QUERY = "query"
    VALIDATE = "validate"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class DegradeReason(str, Enum):
    NONE = "none"
    VLM_TIMEOUT = "vlm_timeout"
    VLM_ERROR = "vlm_error"
    VLM_NO_CONFIG = "vlm_no_config"
    RETRIEVAL_TIMEOUT = "retrieval_timeout"
    RETRIEVAL_ERROR = "retrieval_error"
    NO_EVIDENCE = "no_evidence"


class AuditAction(str, Enum):
    UPLOAD = "upload"
    INGEST_START = "ingest_start"
    INGEST_SUCCESS = "ingest_success"
    INGEST_ROLLBACK = "ingest_rollback"
    QUERY = "query"
    VALIDATE_PASS = "validate_pass"
    VALIDATE_FAIL = "validate_fail"
