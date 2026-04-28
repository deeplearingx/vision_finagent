"""启动脚本：在 uvicorn 启动前修复 triton.__spec__，避免 DeferredCudaCallError。

torch 在模块导入时将 _register_triton_kernels 加入 _queued_calls，
该 callback 调用 importlib.util.find_spec("triton")。
uvicorn CLI 启动时 triton.__spec__ 会被 Python 的 frozen importlib 清空为 None，
导致后续 CUDA lazy init 时抛 ValueError → DeferredCudaCallError。

修复：在 torch 被导入之前修复 triton.__spec__，然后程序化启动 uvicorn。
"""
# Step 1: 修复 triton.__spec__（必须在 import torch 之前）
try:
    import triton as _triton
    if _triton.__spec__ is None:
        import importlib as _il
        _triton.__spec__ = _il.util.spec_from_file_location("triton", _triton.__file__)
except Exception:
    pass

# Step 2: 预初始化 CUDA，消费 _queued_calls 里的 triton callback
import torch
if torch.cuda.is_available():
    torch.cuda.init()

# Step 3: 程序化启动 uvicorn
import uvicorn
uvicorn.run("src.main:app", host="0.0.0.0", port=8000)
