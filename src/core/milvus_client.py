import logging
import os
os.environ.pop("MILVUS_URI", None)  # prevent pymilvus from parsing it at import time
from pymilvus import MilvusClient, DataType
from ..config import settings

_client: MilvusClient | None = None
_log = logging.getLogger(__name__)

# Expected field sets per collection (field_name -> DataType name)
_PATCHES_EXPECTED_FIELDS = {"pk", "report_id", "page_num", "colpali_embeddings"}
_PAGES_EXPECTED_FIELDS = {"page_id", "report_id", "page_num", "image_base64", "_vec"}


def get_client() -> MilvusClient:
    global _client
    if _client is None:
        if settings.MILVUS_TOKEN:
            # Zilliz Cloud: TLS endpoint + token authentication
            _client = MilvusClient(
                uri=settings.MILVUS_URI,
                token=settings.MILVUS_TOKEN,
                db_name=settings.MILVUS_DB_NAME,
            )
        else:
            # Local Lite file or self-hosted Milvus (no auth)
            _client = MilvusClient(settings.MILVUS_URI)
    return _client


def connect_milvus():
    get_client()


def disconnect_milvus():
    global _client
    if _client:
        _client.close()
        _client = None


def get_collection_name() -> str:
    return settings.MILVUS_COLLECTION


def get_pages_collection_name() -> str:
    return settings.MILVUS_COLLECTION + "_pages"


def get_configured_collection_names() -> list[str]:
    return [get_collection_name(), get_pages_collection_name()]


def _ensure_patches_collection(client: MilvusClient, name: str) -> None:
    """patch-level vectors, no image_base64."""
    if not client.has_collection(name):
        schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field("pk", DataType.INT64, is_primary=True)
        schema.add_field("report_id", DataType.VARCHAR, max_length=128)
        schema.add_field("page_num", DataType.INT64)
        schema.add_field("colpali_embeddings", DataType.FLOAT_VECTOR, dim=128)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="colpali_embeddings",
            index_type="AUTOINDEX",
            metric_type="IP",
        )
        client.create_collection(collection_name=name, schema=schema, index_params=index_params)
        return

    _check_schema_drift(client, name, _PATCHES_EXPECTED_FIELDS)
    indexes = client.list_indexes(collection_name=name, field_name="colpali_embeddings")
    if not indexes:
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="colpali_embeddings",
            index_type="AUTOINDEX",
            metric_type="IP",
        )
        client.create_index(collection_name=name, index_params=index_params)


def _check_schema_drift(client: MilvusClient, name: str, expected_fields: set[str]) -> dict:
    """Return drift info: missing/extra fields vs expected. Empty dicts = no drift."""
    try:
        desc = client.describe_collection(collection_name=name)
        actual = {f["name"] for f in desc.get("fields", [])}
        missing = expected_fields - actual
        extra = actual - expected_fields
        if missing or extra:
            _log.warning(
                "milvus_schema_drift detected collection=%s missing=%s extra=%s",
                name, missing, extra,
            )
        return {"collection": name, "missing": list(missing), "extra": list(extra)}
    except Exception as exc:
        _log.error("milvus_schema_drift_check_failed collection=%s error=%s", name, exc)
        return {"collection": name, "error": str(exc)}


def check_schema_health() -> list[dict]:
    """Check both collections for schema drift. Returns list of drift reports."""
    client = get_client()
    results = []
    patches_name = settings.MILVUS_COLLECTION
    pages_name = get_pages_collection_name()
    for name, expected in (
        (patches_name, _PATCHES_EXPECTED_FIELDS),
        (pages_name, _PAGES_EXPECTED_FIELDS),
    ):
        if client.has_collection(name):
            results.append(_check_schema_drift(client, name, expected))
        else:
            results.append({"collection": name, "missing": list(expected), "extra": [], "note": "collection_absent"})
    return results


def _ensure_pages_collection(client: MilvusClient, name: str) -> None:
    """page-level metadata: one row per page, stores image_base64.

    A dummy 1-dim float vector field `_vec` is required by Zilliz Cloud
    (every collection must have a vector field).  It is never searched.
    """
    if client.has_collection(name):
        _check_schema_drift(client, name, _PAGES_EXPECTED_FIELDS)
        return
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("page_id", DataType.VARCHAR, max_length=256, is_primary=True)
    schema.add_field("report_id", DataType.VARCHAR, max_length=128)
    schema.add_field("page_num", DataType.INT64)
    schema.add_field("image_base64", DataType.JSON)
    schema.add_field("_vec", DataType.FLOAT_VECTOR, dim=2)  # required by Zilliz Cloud (min dim=2)

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="_vec", index_type="FLAT", metric_type="L2")
    client.create_collection(collection_name=name, schema=schema, index_params=index_params)


def ensure_collection() -> None:
    client = get_client()
    _ensure_patches_collection(client, settings.MILVUS_COLLECTION)
    _ensure_pages_collection(client, get_pages_collection_name())


def delete_report_data(report_id: str) -> None:
    """Delete all patches and pages for a given report_id (pre-ingest cleanup for retries)."""
    client = get_client()
    client.delete(
        collection_name=get_collection_name(),
        filter=f'report_id == "{report_id}"',
    )
    client.delete(
        collection_name=get_pages_collection_name(),
        filter=f'report_id == "{report_id}"',
    )


def delete_report_data_by_prefix(prefix: str) -> dict:
    """Delete all patches and pages whose report_id starts with prefix."""
    client = get_client()
    # Query matching report_ids first
    col_pages = get_pages_collection_name()
    rows = client.query(
        collection_name=col_pages,
        filter="",
        output_fields=["report_id"],
        limit=16000,
    )
    matched = {r["report_id"] for r in rows if r.get("report_id", "").startswith(prefix)}
    if not matched:
        return {"prefix": prefix, "deleted_reports": 0, "matched_ids": []}
    for rid in matched:
        client.delete(collection_name=get_collection_name(), filter=f'report_id == "{rid}"')
        client.delete(collection_name=col_pages, filter=f'report_id == "{rid}"')
    return {"prefix": prefix, "deleted_reports": len(matched), "matched_ids": sorted(matched)}


# Maximum rows fetched when enumerating report inventory.
# Zilliz Cloud hard-caps query limit at 16384; use a safe value.
_INVENTORY_PAGE_SIZE = 16_000


def list_reports_inventory(max_rows: int = _INVENTORY_PAGE_SIZE) -> dict:
    """Read-only: enumerate all report_ids in the pages collection and count pages per report."""
    client = get_client()
    col = get_pages_collection_name()

    if not client.has_collection(col):
        return {
            "collection": col,
            "reports": [],
            "total_reports": 0,
            "total_rows_fetched": 0,
            "truncated": False,
            "max_rows": max_rows,
        }

    rows = client.query(
        collection_name=col,
        filter="",
        output_fields=["report_id", "page_num"],
        limit=max_rows,
    )

    groups: dict[str, list[int]] = {}
    for r in rows:
        groups.setdefault(r.get("report_id", ""), []).append(r.get("page_num", 0))

    reports = [
        {"report_id": rid, "page_count": len(pages), "page_nums": sorted(pages)}
        for rid, pages in sorted(groups.items())
    ]

    return {
        "collection": col,
        "reports": reports,
        "total_reports": len(reports),
        "total_rows_fetched": len(rows),
        "truncated": len(rows) >= max_rows,
        "max_rows": max_rows,
    }


def clear_report_collections() -> dict:
    """Drop and recreate only the configured report collections."""
    client = get_client()
    targets = get_configured_collection_names()
    dropped: list[str] = []

    for name in targets:
        if client.has_collection(name):
            client.drop_collection(name)
            dropped.append(name)

    ensure_collection()
    schema_health = check_schema_health()
    drift_detected = any(r.get("missing") or r.get("extra") for r in schema_health)
    return {
        "dropped": dropped,
        "recreated": targets,
        "schema_health": schema_health,
        "schema_drift_detected": drift_detected,
    }
