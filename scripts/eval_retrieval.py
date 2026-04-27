"""
Retrieval quality evaluation against eval_metadata.json.

Metrics:
  - answer_recall@k : fraction of ground-truth answer tokens found in VLM response
  - avg_maxsim      : mean top-1 MaxSim score across queries
  - retrieved_count : mean number of pages returned

Usage:
  python scripts/eval_retrieval.py \
      --meta ../../eval_metadata.json \
      --api  http://localhost:8000 \
      --n    50 \
      --top_k 3
"""
import argparse, json, re, sys
import httpx

def token_recall(pred: str, gold: str) -> float:
    """Fraction of gold tokens present in pred (case-insensitive)."""
    gold_tokens = set(re.findall(r"\w+", gold.lower()))
    if not gold_tokens:
        return 0.0
    pred_tokens = set(re.findall(r"\w+", pred.lower()))
    return len(gold_tokens & pred_tokens) / len(gold_tokens)


def run(meta_path: str, api_base: str, n: int, top_k: int, candidate_k: int):
    with open(meta_path, encoding="utf-8") as f:
        queries = json.load(f)["queries"][:n]

    recalls, maxsims, counts = [], [], []
    client = httpx.Client(base_url=api_base, timeout=120)

    for item in queries:
        qid = item["query_id"]
        gold = item.get("answer") or (item.get("raw_answers") or [""])[0]
        try:
            resp = client.post("/reports/query", data={
                "query": item["query"],
                "top_k": top_k,
                "candidate_k": candidate_k,
            })
            resp.raise_for_status()
            body = resp.json()
        except Exception as exc:
            print(f"[WARN] query_id={qid} failed: {exc}", file=sys.stderr)
            continue

        pred = body.get("answer") or ""
        pages = body.get("retrieved_pages", [])

        recall = token_recall(pred, gold)
        top_score = pages[0]["maxsim_score"] if pages else 0.0

        recalls.append(recall)
        maxsims.append(top_score)
        counts.append(len(pages))
        print(f"qid={qid:4d}  recall={recall:.3f}  maxsim={top_score:.4f}  pages={len(pages)}")

    if not recalls:
        print("No results collected.")
        return

    print(f"\n--- Summary (n={len(recalls)}) ---")
    print(f"answer_recall@{top_k} : {sum(recalls)/len(recalls):.4f}")
    print(f"avg_top1_maxsim      : {sum(maxsims)/len(maxsims):.4f}")
    print(f"avg_retrieved_pages  : {sum(counts)/len(counts):.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--meta",        default="../../eval_metadata.json")
    p.add_argument("--api",         default="http://localhost:8000")
    p.add_argument("--n",           type=int, default=50)
    p.add_argument("--top_k",       type=int, default=3)
    p.add_argument("--candidate_k", type=int, default=50)
    args = p.parse_args()
    run(args.meta, args.api, args.n, args.top_k, args.candidate_k)
