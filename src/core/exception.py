class VisionFinAgentException(Exception):
    def __init__(self, error_code: str, detail: str):
        self.error_code = error_code
        self.detail = detail
        super().__init__(detail)


class IngestionError(VisionFinAgentException):
    def __init__(self, detail: str):
        super().__init__("INGESTION_ERROR", detail)


class RetrievalError(VisionFinAgentException):
    def __init__(self, detail: str):
        super().__init__("RETRIEVAL_ERROR", detail)


class ValidationError(VisionFinAgentException):
    def __init__(self, detail: str):
        super().__init__("VALIDATION_ERROR", detail)


class RollbackError(VisionFinAgentException):
    def __init__(self, detail: str):
        super().__init__("ROLLBACK_ERROR", detail)
