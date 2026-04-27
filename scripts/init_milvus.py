import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.core.milvus_client import ensure_collection, get_collection_name

if __name__ == "__main__":
    ensure_collection()
    print(f"Collection '{get_collection_name()}' ready.")
