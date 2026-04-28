"""Regression script for conservative two-pass VLM evidence expansion.

Verifies:
  1. is_insufficient_evidence() sentinel detection
  2. Config defaults and boundary values
  3. retrieve() backward-compatible signature
  4. reports.py control-flow identifiers
  5. Response schema: new fields additive, no existing field removed
"""
import sys, ast

PASS = "✓"
FAIL = "✗"
results = []


def assert_(cond):
    if not cond:
        raise AssertionError("assertion failed")


def check(name, fn):
    try:
        fn()
        results.append((PASS, name))
    except Exception as e:
        results.append((FAIL, f"{name}: {e}"))


# ── 1. Sentinel detection ────────────────────────────────────────────────────
from src.services.vlm_service import is_insufficient_evidence, INSUFFICIENT_EVIDENCE_PHRASE

check("sentinel: exact match",
      lambda: assert_(is_insufficient_evidence(INSUFFICIENT_EVIDENCE_PHRASE)))
check("sentinel: case-insensitive",
      lambda: assert_(is_insufficient_evidence("insufficient evidence in provided pages")))
check("sentinel: embedded in text",
      lambda: assert_(is_insufficient_evidence(f"Note: {INSUFFICIENT_EVIDENCE_PHRASE}.")))
check("sentinel: normal answer → False",
      lambda: assert_(not is_insufficient_evidence("净利润 12 亿")))
check("sentinel: None → False",
      lambda: assert_(not is_insufficient_evidence(None)))
check("sentinel: vlm_error text → False",
      lambda: assert_(not is_insufficient_evidence("VLM 生成失败（vlm_timeout）")))


# ── 2. Config defaults ───────────────────────────────────────────────────────
from src.config import settings

check("config: SECOND_PASS_ENABLED default True",
      lambda: assert_(settings.VLM_SECOND_PASS_ENABLED is True))
check("config: SECOND_PASS_TOP_K > MAX_VLM_IMAGES",
      lambda: assert_(settings.VLM_SECOND_PASS_TOP_K >= settings.MAX_VLM_IMAGES))
check("config: SECOND_PASS_CANDIDATE_K > first-pass default",
      lambda: assert_(settings.VLM_SECOND_PASS_CANDIDATE_K > 50))
check("config: SECOND_PASS_MAX_IMAGES > 0",
      lambda: assert_(settings.VLM_SECOND_PASS_MAX_IMAGES > 0))


# ── 3. retrieve() backward-compatible signature ──────────────────────────────
import inspect
from src.services.retrieval_service import retrieve

sig = inspect.signature(retrieve)
params = sig.parameters
check("retrieve: pass_label param exists",
      lambda: assert_("pass_label" in params))
check("retrieve: pass_label default == 'first_pass'",
      lambda: assert_(params["pass_label"].default == "first_pass"))
check("retrieve: original params unchanged",
      lambda: assert_(all(p in params for p in ["query", "target_companies", "top_k", "candidate_k"])))


# ── 4. reports.py control-flow identifiers ───────────────────────────────────
with open("src/routers/reports.py") as f:
    src = f.read()

for ident in ["vlm_passes", "second_pass_triggered", "evidence_source_detail",
              "is_insufficient_evidence", "VLM_SECOND_PASS_ENABLED",
              "second_pass", "first_pass_elapsed"]:
    check(f"reports.py: '{ident}' present",
          lambda i=ident: assert_(i in src))

# import check
tree = ast.parse(src)
imports_ok = any(
    isinstance(n, ast.ImportFrom) and "vlm_service" in (n.module or "")
    and any(a.name == "is_insufficient_evidence" for a in n.names)
    for n in ast.walk(tree)
)
check("reports.py: is_insufficient_evidence imported", lambda: assert_(imports_ok))


# ── 5. Response schema: new fields additive ──────────────────────────────────
new_fields = {"vlm_passes", "second_pass_triggered", "evidence_source_detail"}
existing_fields = {"session_id", "answer", "degraded", "degrade_reason",
                   "evidence", "evidence_source", "image_fetch_incomplete", "retrieved_pages"}

for f in existing_fields:
    check(f"response schema: existing field '{f}' preserved",
          lambda field=f: assert_(field in src))
for f in new_fields:
    check(f"response schema: new field '{f}' added",
          lambda field=f: assert_(field in src))


# ── Summary ──────────────────────────────────────────────────────────────────
print("\n=== Second-Pass Regression Results ===")
passed = sum(1 for r in results if r[0] == PASS)
failed = sum(1 for r in results if r[0] == FAIL)
for status, name in results:
    print(f"  {status} {name}")
print(f"\n  {passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
