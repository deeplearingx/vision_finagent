"""Regression tests: idempotency token lifecycle & task failure release."""
import json
import pytest
import asyncio
from unittest.mock import AsyncMock, patch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# check_and_set: first call returns True, duplicate returns False
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_check_and_set_first_call_true(fake_redis):
    from src.utils.idempotency import check_and_set
    result = await check_and_set("upload:rpt_new")
    assert result is True


@pytest.mark.asyncio
async def test_check_and_set_duplicate_false(fake_redis):
    from src.utils.idempotency import check_and_set
    await check_and_set("upload:rpt_dup2")
    result = await check_and_set("upload:rpt_dup2")
    assert result is False


# ---------------------------------------------------------------------------
# release: after release, token can be re-acquired
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_release_allows_retry(fake_redis):
    from src.utils.idempotency import check_and_set, release
    await check_and_set("upload:rpt_retry")
    assert await check_and_set("upload:rpt_retry") is False  # blocked

    await release("upload:rpt_retry")
    assert await check_and_set("upload:rpt_retry") is True   # now allowed


# ---------------------------------------------------------------------------
# task_service: on FAILED, idempotency token is released
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_failed_task_releases_idempotency_token(fake_redis):
    from src.utils.idempotency import check_and_set
    from src.services import task_service

    report_id = "rpt_fail_release"
    # Simulate token already set (as upload endpoint would do)
    await fake_redis.set(f"upload:{report_id}", "1", nx=True)

    # Patch ingest_report to raise so task fails
    with patch(
        "src.services.task_service.ingest_report",
        new_callable=AsyncMock,
        side_effect=RuntimeError("ingest boom"),
    ):
        task_id = await task_service.submit_ingest_task(report_id, "/tmp/fake.pdf")
        # Give the background task time to run
        await asyncio.sleep(0.1)

    # Token must be gone so the same report_id can be retried
    can_retry = await check_and_set(f"upload:{report_id}")
    assert can_retry is True, "idempotency token must be released after task failure"


# ---------------------------------------------------------------------------
# task_service: on SUCCESS, idempotency token is NOT released
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_success_task_keeps_idempotency_token(fake_redis):
    from src.utils.idempotency import check_and_set
    from src.services import task_service

    report_id = "rpt_success_keep"
    await fake_redis.set(f"upload:{report_id}", "1", nx=True)

    with patch(
        "src.services.task_service.ingest_report",
        new_callable=AsyncMock,
        return_value=None,
    ):
        await task_service.submit_ingest_task(report_id, "/tmp/fake.pdf")
        await asyncio.sleep(0.1)

    # Token must still be set — duplicate upload should be blocked
    can_resubmit = await check_and_set(f"upload:{report_id}")
    assert can_resubmit is False, "idempotency token must be kept after task success"
