#!/usr/bin/env python3
"""
小样本冒烟评测脚本
用法见脚本末尾 __main__ 块或 README 说明。
"""
import argparse
import json
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# 公司名 → report_id 映射（独立实现，不依赖 FastAPI 导入）
# ---------------------------------------------------------------------------

_COMPANY_REPORT_ALIASES = {
    "jpmorganchase": "jpm_2024", "jpmorgan chase": "jpm_2024",
    "jpmorgan": "jpm_2024", "jpm": "jpm_2024",
    "citigroup": "citi_2024", "citi": "citi_2024",
    "goldman sachs": "gs_2024", "goldman": "gs_2024", "gs": "gs_2024",
    "morgan stanley": "ms_2024", "ms": "ms_2024",
    "bank of america": "boa_2024", "bofa": "boa_2024", "boa": "boa_2024", "bac": "boa_2024",
    "wells fargo": "wf_2024", "wf": "wf_2024",
}


def _infer_report_ids(question: str) -> list[str]:
    q = question.lower()
    matched = []
    for alias, rid in _COMPANY_REPORT_ALIASES.items():
        if alias in q and rid not in matched:
            matched.append(rid)
    return matched


# ---------------------------------------------------------------------------
# 自动评分
# ---------------------------------------------------------------------------

def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", text.lower()))


def _numeric_match_rate(ref: str, sys_ans: str) -> float:
    nums_ref = set(re.findall(r"\d+\.?\d*", ref))
    if not nums_ref:
        return 1.0
    nums_sys = set(re.findall(r"\d+\.?\d*", sys_ans))
    return len(nums_ref & nums_sys) / len(nums_ref)


def _bool_polarity(text: str) -> str | None:
    t = text.lower()
    if re.search(r"\byes\b|\btrue\b|\bdid\b|\bwas\b|\bwere\b|\bhas\b|\bhave\b", t):
        return "positive"
    if re.search(r"\bno\b|\bnot\b|\bnever\b|\bfalse\b|\bdid not\b|\bwas not\b", t):
        return "negative"
    return None


def auto_score(query_types: list[str], ref: str, sys_ans: str, evidence_source: str) -> tuple[bool, str]:
    """返回 (passed, reason)"""
    insuf = bool(re.search(r"insufficient evidence|未找到相关|no relevant|cannot answer", sys_ans, re.I))

    # 规则 1：参考可答但系统答 insufficient evidence
    if insuf and len(ref.strip()) > 10:
        return False, "insufficient_evidence_but_reference_answerable"

    # 规则 2：布尔题极性冲突
    if "boolean" in query_types:
        rp, sp = _bool_polarity(ref), _bool_polarity(sys_ans)
        if rp and sp and rp != sp:
            return False, "boolean_polarity_mismatch"

    # 规则 3：数值题关键数字匹配率
    if "numerical" in query_types or "extraction" in query_types:
        rate = _numeric_match_rate(ref, sys_ans)
        if rate < 0.5:
            return False, f"numeric_mismatch(rate={rate:.2f})"

    # 规则 4：内容词 token F1 / recall
    ref_tok, sys_tok = _tokens(ref), _tokens(sys_ans)
    if ref_tok:
        recall = len(ref_tok & sys_tok) / len(ref_tok)
        precision = len(ref_tok & sys_tok) / len(sys_tok) if sys_tok else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        if f1 >= 0.33 or recall >= 0.45:
            return True, f"token_f1={f1:.2f},recall={recall:.2f}"
        return False, f"low_semantic_overlap(f1={f1:.2f},recall={recall:.2f})"

    return True, "no_ref_tokens"


# ---------------------------------------------------------------------------
# 核心评测循环
# ---------------------------------------------------------------------------

def run_eval(
    metadata_path: str,
    n: int,
    api_base: str,
    refresh_retrieval: bool,
    output_path: str,
) -> None:
    meta = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    queries = meta["queries"][:n]

    api_url = api_base.rstrip("/") + "/reports/query"
    results = []

    for item in queries:
        qid = item["query_id"]
        question = item["query"]
        ref = item.get("answer") or (item.get("raw_answers") or [""])[0]
        query_types = item.get("query_types", [])

        session_id = f"smokeeval-q{qid}-{uuid.uuid4().hex[:6]}"
        report_ids = _infer_report_ids(question)
        payload = {
            "question": question,
            "session_id": session_id,
            "use_retrieval": True,
            "refresh_retrieval": refresh_retrieval,
            "top_k": 10,
            "candidate_k": 300,
            "report_ids": report_ids,
        }

        try:
            resp = requests.post(api_url, json=payload, timeout=600)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            results.append({
                "query_id": qid,
                "question": question,
                "reference_answer": ref,
                "system_answer": None,
                "degraded": None,
                "degrade_reason": None,
                "evidence_source": None,
                "evidence_pages": [],
                "vlm_passes": None,
                "passed": False,
                "score_reason": f"request_error: {exc}",
            })
            print(f"[{qid}] ERROR: {exc}", file=sys.stderr)
            continue

        sys_ans = data.get("answer") or ""
        evidence_source = data.get("evidence_source", "unknown")
        passed, reason = auto_score(query_types, ref, sys_ans, evidence_source)

        results.append({
            "query_id": qid,
            "question": question,
            "reference_answer": ref,
            "system_answer": sys_ans,
            "degraded": data.get("degraded"),
            "degrade_reason": data.get("degrade_reason"),
            "evidence_source": evidence_source,
            "evidence_pages": [
                {"report_id": e.get("report_id"), "page_num": e.get("page_num")}
                for e in data.get("evidence", [])
            ],
            "vlm_passes": data.get("vlm_passes"),
            "inferred_report_ids": report_ids,
            "passed": passed,
            "score_reason": reason,
        })
        status = "PASS" if passed else "FAIL"
        print(f"[{qid}] {status} | src={evidence_source} | {reason}")

    _write_outputs(results, output_path)


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------

def _write_outputs(results: list[dict], output_path: str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # 逐题 JSON
    json_path = out.with_suffix(".json")
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # 摘要 Markdown
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    rate = passed / total * 100 if total else 0

    src_dist: dict[str, int] = {}
    fail_dist: dict[str, int] = {}
    for r in results:
        src = r.get("evidence_source") or "unknown"
        src_dist[src] = src_dist.get(src, 0) + 1
        if not r["passed"]:
            reason = r.get("score_reason", "unknown").split("(")[0]
            fail_dist[reason] = fail_dist.get(reason, 0) + 1

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    md_lines = [
        f"# 冒烟评测摘要 — {ts}",
        "",
        f"- 样本数：{total}",
        f"- 自动判通过：{passed}",
        f"- 自动判失败：{total - passed}",
        f"- 自动语义一致率：{rate:.1f}%",
        "",
        "## evidence_source 分布",
        "",
    ]
    for k, v in sorted(src_dist.items(), key=lambda x: -x[1]):
        md_lines.append(f"- `{k}`: {v}")
    md_lines += ["", "## 失败模式分布", ""]
    for k, v in sorted(fail_dist.items(), key=lambda x: -x[1]):
        md_lines.append(f"- `{k}`: {v}")
    md_lines += ["", f"详细结果见：`{json_path.name}`"]

    md_path = out.with_suffix(".md")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"\n结果已写出：\n  JSON → {json_path}\n  摘要 → {md_path}")
    print(f"自动语义一致率：{passed}/{total} = {rate:.1f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="小样本冒烟评测脚本")
    parser.add_argument("--metadata", default="autodl-tmp/eval_metadata.json", help="eval_metadata.json 路径")
    parser.add_argument("--n", type=int, default=10, help="评测前 N 条 query（默认 10）")
    parser.add_argument("--api-base", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--refresh", action="store_true", help="强制 refresh_retrieval=true")
    parser.add_argument("--output", default="autodl-tmp/eval_smoke_results", help="输出文件路径前缀（不含扩展名）")
    args = parser.parse_args()

    run_eval(
        metadata_path=args.metadata,
        n=args.n,
        api_base=args.api_base,
        refresh_retrieval=args.refresh,
        output_path=args.output,
    )
