import sys
import types

# Fix: bitsandbytes 0.44.x unconditionally imports triton.ops.matmul_perf_model
# which no longer exists in triton 3.x. Inject a minimal stub before any real import.
if "triton.ops.matmul_perf_model" not in sys.modules:
    triton_mod = sys.modules.get("triton")
    if triton_mod is None:
        triton_mod = types.ModuleType("triton")
        sys.modules["triton"] = triton_mod

    ops_mod = sys.modules.get("triton.ops")
    if ops_mod is None:
        ops_mod = types.ModuleType("triton.ops")
        ops_mod.__name__ = "triton.ops"
        sys.modules["triton.ops"] = ops_mod
        triton_mod.ops = ops_mod

    stub = types.ModuleType("triton.ops.matmul_perf_model")
    stub.early_config_prune = lambda configs, *a, **k: configs
    stub.estimate_matmul_time = lambda **k: 0
    sys.modules["triton.ops.matmul_perf_model"] = stub
    ops_mod.matmul_perf_model = stub
