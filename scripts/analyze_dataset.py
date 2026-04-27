"""Analyze vidore/vidore_v3_finance_en dataset structure."""
from datasets import load_dataset
import json

def main():
    print("Loading vidore/vidore_v3_finance_en ...")
    ds = load_dataset("vidore/vidore_v3_finance_en", split="train")

    print(f"\nDataset size: {len(ds)}")
    print(f"Features: {ds.features}")
    print(f"Column names: {ds.column_names}")

    # Show first sample
    sample = ds[0]
    print(f"\n--- First sample keys ---")
    for k, v in sample.items():
        vtype = type(v).__name__
        if hasattr(v, 'shape'):
            print(f"  {k}: {vtype}, shape={v.shape}, dtype={v.dtype}")
        elif hasattr(v, '__len__') and not isinstance(v, (str, bytes)):
            print(f"  {k}: {vtype}, len={len(v)}")
        else:
            preview = str(v)[:100]
            print(f"  {k}: {vtype} = {preview}...")

    # Show unique docids / sources
    if "docid" in ds.column_names:
        unique_docs = set(ds["docid"])
        print(f"\nUnique documents: {len(unique_docs)}")
    if "uri" in ds.column_names:
        unique_uris = set(ds["uri"])
        print(f"Unique URIs: {len(unique_uris)}")

    # Query statistics
    if "query" in ds.column_names:
        qlens = [len(q) for q in ds["query"]]
        print(f"\nQuery lengths: min={min(qlens)}, max={max(qlens)}, avg={sum(qlens)/len(qlens):.1f}")

    # Show a few queries
    print(f"\n--- Sample queries ---")
    for i in range(min(5, len(ds))):
        print(f"  [{i}] {ds[i]['query'][:120]}...")


if __name__ == "__main__":
    main()
