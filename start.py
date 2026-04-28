# triton.__spec__ 补丁必须在任何 import torch / uvicorn.run 之前执行，
# 否则 torch/cuda/__init__.py 模块级 _lazy_call(_register_triton_kernels)
# 会在 import torch 完成时把损坏的 __spec__ 捕获进队列，
# 导致后续 CUDA lazy init 触发 DeferredCudaCallError。
try:
    import triton as _triton
    if _triton.__spec__ is None:
        import importlib as _il
        _triton.__spec__ = _il.util.spec_from_file_location("triton", _triton.__file__)
except Exception:
    pass

import uvicorn
uvicorn.run("src.main:app", host="0.0.0.0", port=8000)
