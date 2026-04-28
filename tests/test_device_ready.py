"""最小回归测试：设备判定函数与 ready 语义。"""
import sys
import types
import pytest


def _patch_colpali(monkeypatch):
    pkg = types.ModuleType("colpali_engine")
    models = types.ModuleType("colpali_engine.models")
    pali = types.ModuleType("colpali_engine.models.paligemma")
    cp = types.ModuleType("colpali_engine.models.paligemma.colpali")
    mcp = types.ModuleType("colpali_engine.models.paligemma.colpali.modeling_colpali")
    pcp = types.ModuleType("colpali_engine.models.paligemma.colpali.processing_colpali")

    class _FakeColPali:
        pass

    class _FakeProcessor:
        pass

    mcp.ColPali = _FakeColPali
    pcp.ColPaliProcessor = _FakeProcessor

    for name, mod in [
        ("colpali_engine", pkg),
        ("colpali_engine.models", models),
        ("colpali_engine.models.paligemma", pali),
        ("colpali_engine.models.paligemma.colpali", cp),
        ("colpali_engine.models.paligemma.colpali.modeling_colpali", mcp),
        ("colpali_engine.models.paligemma.colpali.processing_colpali", pcp),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)


def _import_service(monkeypatch):
    _patch_colpali(monkeypatch)
    for key in list(sys.modules):
        if "retrieval_service" in key:
            monkeypatch.delitem(sys.modules, key, raising=False)
    import importlib
    return importlib.import_module("autodl_tmp.vision_finagent.src.services.retrieval_service")


class _FakeParam:
    def __init__(self, device_str: str):
        self._device_str = device_str

    @property
    def device(self):
        import torch
        return torch.device(self._device_str)


class _FakeModel:
    def __init__(self, device_str: str, hf_device_map=None):
        self._device_str = device_str
        if hf_device_map is not None:
            self.hf_device_map = hf_device_map

    def parameters(self):
        yield _FakeParam(self._device_str)


# ---------------------------------------------------------------------------
# get_model_device
# ---------------------------------------------------------------------------

def test_get_model_device_none(monkeypatch):
    svc = _import_service(monkeypatch)
    monkeypatch.setattr(svc, "_model", None)
    assert svc.get_model_device() is None


def test_get_model_device_cpu(monkeypatch):
    svc = _import_service(monkeypatch)
    monkeypatch.setattr(svc, "_model", _FakeModel("cpu"))
    assert svc.get_model_device() == "cpu"


def test_get_model_device_cuda(monkeypatch):
    svc = _import_service(monkeypatch)
    monkeypatch.setattr(svc, "_model", _FakeModel("cuda:0"))
    assert svc.get_model_device().startswith("cuda")


# ---------------------------------------------------------------------------
# get_hf_device_map
# ---------------------------------------------------------------------------

def test_get_hf_device_map_none_when_no_model(monkeypatch):
    svc = _import_service(monkeypatch)
    monkeypatch.setattr(svc, "_model", None)
    assert svc.get_hf_device_map() is None


def test_get_hf_device_map_none_when_no_attr(monkeypatch):
    svc = _import_service(monkeypatch)
    monkeypatch.setattr(svc, "_model", _FakeModel("cpu"))  # no hf_device_map attr
    assert svc.get_hf_device_map() is None


def test_get_hf_device_map_returns_map(monkeypatch):
    svc = _import_service(monkeypatch)
    hf_map = {"model.embed_tokens": "cpu", "model.layers.0": "cuda:0"}
    monkeypatch.setattr(svc, "_model", _FakeModel("cpu", hf_device_map=hf_map))
    assert svc.get_hf_device_map() == hf_map


# ---------------------------------------------------------------------------
# _has_cuda_placement
# ---------------------------------------------------------------------------

def test_has_cuda_placement_false_when_no_model(monkeypatch):
    svc = _import_service(monkeypatch)
    monkeypatch.setattr(svc, "_model", None)
    assert svc._has_cuda_placement() is False


def test_has_cuda_placement_via_first_param_cuda(monkeypatch):
    svc = _import_service(monkeypatch)
    monkeypatch.setattr(svc, "_model", _FakeModel("cuda:0"))
    assert svc._has_cuda_placement() is True


def test_has_cuda_placement_false_when_first_param_cpu_no_hf_map(monkeypatch):
    svc = _import_service(monkeypatch)
    monkeypatch.setattr(svc, "_model", _FakeModel("cpu"))
    assert svc._has_cuda_placement() is False


def test_has_cuda_placement_true_via_hf_device_map(monkeypatch):
    """accelerate 分层：首参数在 cpu，但 hf_device_map 中有 cuda 层 → 应判定为有 CUDA 放置。"""
    svc = _import_service(monkeypatch)
    hf_map = {"model.embed_tokens": "cpu", "model.layers.0": "cuda:0", "lm_head": "cuda:0"}
    monkeypatch.setattr(svc, "_model", _FakeModel("cpu", hf_device_map=hf_map))
    assert svc._has_cuda_placement() is True


def test_has_cuda_placement_false_when_hf_map_all_cpu(monkeypatch):
    svc = _import_service(monkeypatch)
    hf_map = {"model.embed_tokens": "cpu", "model.layers.0": "cpu"}
    monkeypatch.setattr(svc, "_model", _FakeModel("cpu", hf_device_map=hf_map))
    assert svc._has_cuda_placement() is False


# ---------------------------------------------------------------------------
# is_model_ready — CUDA 可用场景
# ---------------------------------------------------------------------------

def test_ready_false_when_model_none(monkeypatch):
    svc = _import_service(monkeypatch)
    monkeypatch.setattr(svc, "_model", None)
    monkeypatch.setattr(svc.torch.cuda, "is_available", lambda: True)
    assert svc.is_model_ready() is False


def test_ready_false_when_cuda_available_but_all_cpu(monkeypatch):
    """关键回归：首参数 cpu 且无 hf_device_map → 真实 CPU fallback → not ready。"""
    svc = _import_service(monkeypatch)
    monkeypatch.setattr(svc, "_model", _FakeModel("cpu"))
    monkeypatch.setattr(svc.torch.cuda, "is_available", lambda: True)
    assert svc.is_model_ready() is False


def test_ready_true_when_cuda_available_and_first_param_cuda(monkeypatch):
    svc = _import_service(monkeypatch)
    monkeypatch.setattr(svc, "_model", _FakeModel("cuda:0"))
    monkeypatch.setattr(svc.torch.cuda, "is_available", lambda: True)
    assert svc.is_model_ready() is True


def test_ready_true_when_cuda_available_and_layered_device_map(monkeypatch):
    """核心修复验证：首参数在 cpu，但 hf_device_map 有 cuda 层 → ready。"""
    svc = _import_service(monkeypatch)
    hf_map = {"model.embed_tokens": "cpu", "model.layers.0": "cuda:0"}
    monkeypatch.setattr(svc, "_model", _FakeModel("cpu", hf_device_map=hf_map))
    monkeypatch.setattr(svc.torch.cuda, "is_available", lambda: True)
    assert svc.is_model_ready() is True


# ---------------------------------------------------------------------------
# is_model_ready — CPU-only 场景
# ---------------------------------------------------------------------------

def test_ready_true_when_no_cuda_and_model_on_cpu(monkeypatch):
    svc = _import_service(monkeypatch)
    monkeypatch.setattr(svc, "_model", _FakeModel("cpu"))
    monkeypatch.setattr(svc.torch.cuda, "is_available", lambda: False)
    assert svc.is_model_ready() is True


# ---------------------------------------------------------------------------
# _get_input_device — 推理输入设备解析
# ---------------------------------------------------------------------------

def test_get_input_device_none_model_returns_cpu(monkeypatch):
    svc = _import_service(monkeypatch)
    monkeypatch.setattr(svc, "_model", None)
    assert svc._get_input_device() == "cpu"


def test_get_input_device_single_gpu(monkeypatch):
    """单卡全量加载：首参数在 cuda:0 → 输入设备 cuda:0。"""
    svc = _import_service(monkeypatch)
    monkeypatch.setattr(svc, "_model", _FakeModel("cuda:0"))
    assert svc._get_input_device() == "cuda:0"


def test_get_input_device_cpu_only(monkeypatch):
    """CPU-only fallback：首参数在 cpu，无 hf_device_map → 输入设备 cpu。"""
    svc = _import_service(monkeypatch)
    monkeypatch.setattr(svc, "_model", _FakeModel("cpu"))
    assert svc._get_input_device() == "cpu"


def test_get_input_device_layered_device_map(monkeypatch):
    """accelerate 分层：首参数在 cpu，hf_device_map 有 cuda 层 → 输入设备应为 cuda。"""
    svc = _import_service(monkeypatch)
    hf_map = {"model.embed_tokens": "cpu", "model.layers.0": "cuda:0", "lm_head": "cuda:1"}
    monkeypatch.setattr(svc, "_model", _FakeModel("cpu", hf_device_map=hf_map))
    result = svc._get_input_device()
    assert result.startswith("cuda"), f"expected cuda device, got {result!r}"


def test_get_input_device_layered_all_cpu(monkeypatch):
    """hf_device_map 全 cpu → 输入设备 cpu。"""
    svc = _import_service(monkeypatch)
    hf_map = {"model.embed_tokens": "cpu", "model.layers.0": "cpu"}
    monkeypatch.setattr(svc, "_model", _FakeModel("cpu", hf_device_map=hf_map))
    assert svc._get_input_device() == "cpu"


# ---------------------------------------------------------------------------
# REQUIRE_RETRIEVAL_GPU strict/optional 模式
# ---------------------------------------------------------------------------

def _import_service_with_gpu_flag(monkeypatch, require_gpu: bool):
    """重新导入 retrieval_service，并覆盖 settings.REQUIRE_RETRIEVAL_GPU。"""
    svc = _import_service(monkeypatch)
    monkeypatch.setattr(svc.settings, "REQUIRE_RETRIEVAL_GPU", require_gpu)
    return svc


def test_strict_mode_cpu_fallback_raises(monkeypatch):
    """strict 模式：CUDA 可用但模型全在 CPU → _load_model 应抛 RuntimeError。"""
    svc = _import_service_with_gpu_flag(monkeypatch, require_gpu=True)
    monkeypatch.setattr(svc.torch.cuda, "is_available", lambda: True)

    class _FakeColPaliStrict:
        hf_device_map = {"model": "cpu"}
        def eval(self): return self
        def parameters(self): yield _FakeParam("cpu")

    class _FakeProcessorStrict:
        @classmethod
        def from_pretrained(cls, *a, **kw): return cls()

    _FakeColPaliStrict.from_pretrained = classmethod(lambda cls, *a, **kw: _FakeColPaliStrict())
    monkeypatch.setattr(svc, "_model", None)
    monkeypatch.setattr(svc, "_processor", None)

    import sys
    mcp = sys.modules["colpali_engine.models.paligemma.colpali.modeling_colpali"]
    pcp = sys.modules["colpali_engine.models.paligemma.colpali.processing_colpali"]
    monkeypatch.setattr(mcp, "ColPali", _FakeColPaliStrict)
    monkeypatch.setattr(pcp, "ColPaliProcessor", _FakeProcessorStrict)

    with pytest.raises(RuntimeError, match="no model layer is on CUDA"):
        svc._load_model()


def test_optional_mode_cpu_fallback_no_raise(monkeypatch):
    """optional 模式：CUDA 可用但模型全在 CPU → _load_model 不抛异常。"""
    svc = _import_service_with_gpu_flag(monkeypatch, require_gpu=False)
    monkeypatch.setattr(svc.torch.cuda, "is_available", lambda: True)

    class _FakeColPaliOpt:
        hf_device_map = {"model": "cpu"}
        def eval(self): return self
        def parameters(self): yield _FakeParam("cpu")

    class _FakeProcessorOpt:
        @classmethod
        def from_pretrained(cls, *a, **kw): return cls()

    _FakeColPaliOpt.from_pretrained = classmethod(lambda cls, *a, **kw: _FakeColPaliOpt())
    monkeypatch.setattr(svc, "_model", None)
    monkeypatch.setattr(svc, "_processor", None)

    import sys
    mcp = sys.modules["colpali_engine.models.paligemma.colpali.modeling_colpali"]
    pcp = sys.modules["colpali_engine.models.paligemma.colpali.processing_colpali"]
    monkeypatch.setattr(mcp, "ColPali", _FakeColPaliOpt)
    monkeypatch.setattr(pcp, "ColPaliProcessor", _FakeProcessorOpt)

    svc._load_model()  # must not raise
    assert svc.is_cpu_fallback() is True
    assert svc.is_model_ready() is True


def test_strict_mode_is_model_ready_false_on_cpu_fallback(monkeypatch):
    """strict 模式：CPU fallback → is_model_ready() == False。"""
    svc = _import_service_with_gpu_flag(monkeypatch, require_gpu=True)
    monkeypatch.setattr(svc, "_model", _FakeModel("cpu"))
    monkeypatch.setattr(svc.torch.cuda, "is_available", lambda: True)
    assert svc.is_model_ready() is False


def test_optional_mode_is_model_ready_true_on_cpu_fallback(monkeypatch):
    """optional 模式：CPU fallback → is_model_ready() == True。"""
    svc = _import_service_with_gpu_flag(monkeypatch, require_gpu=False)
    monkeypatch.setattr(svc, "_model", _FakeModel("cpu"))
    monkeypatch.setattr(svc.torch.cuda, "is_available", lambda: True)
    assert svc.is_model_ready() is True


def test_is_cpu_fallback_false_when_cuda_placement(monkeypatch):
    """真 CUDA placement → is_cpu_fallback() == False。"""
    svc = _import_service(monkeypatch)
    monkeypatch.setattr(svc, "_model", _FakeModel("cuda:0"))
    monkeypatch.setattr(svc.torch.cuda, "is_available", lambda: True)
    assert svc.is_cpu_fallback() is False


def test_is_cpu_fallback_false_when_no_cuda(monkeypatch):
    """CPU-only 环境（无 CUDA）→ is_cpu_fallback() == False（非降级，是正常 CPU 模式）。"""
    svc = _import_service(monkeypatch)
    monkeypatch.setattr(svc, "_model", _FakeModel("cpu"))
    monkeypatch.setattr(svc.torch.cuda, "is_available", lambda: False)
    assert svc.is_cpu_fallback() is False


# ---------------------------------------------------------------------------
# _is_lora_adapter_dir
# ---------------------------------------------------------------------------

def test_is_lora_adapter_dir_true(monkeypatch, tmp_path):
    """有 adapter_config.json 无 config.json → LoRA adapter 目录。"""
    (tmp_path / "adapter_config.json").write_text("{}")
    svc = _import_service(monkeypatch)
    assert svc._is_lora_adapter_dir(str(tmp_path)) is True


def test_is_lora_adapter_dir_false_full_model(monkeypatch, tmp_path):
    """有 config.json → 完整模型目录，不是 LoRA adapter。"""
    (tmp_path / "config.json").write_text("{}")
    svc = _import_service(monkeypatch)
    assert svc._is_lora_adapter_dir(str(tmp_path)) is False


def test_is_lora_adapter_dir_false_empty(monkeypatch, tmp_path):
    """空目录 → 不是 LoRA adapter。"""
    svc = _import_service(monkeypatch)
    assert svc._is_lora_adapter_dir(str(tmp_path)) is False


# ---------------------------------------------------------------------------
# _load_model — GPU strict 路径（显式 cuda:0）
# ---------------------------------------------------------------------------

def _make_gpu_model_cls():
    """首参数在 cpu，.to(device) 更新设备 → 模拟显式 cuda:0 加载成功。"""
    class _M(_FakeModel):
        def to(self, device):
            self._device_str = device
            return self
        def eval(self): return self
        @classmethod
        def from_pretrained(cls, *a, **kw): return cls("cpu")
    return _M


class _FakeProcessorCls:
    @classmethod
    def from_pretrained(cls, *a, **kw): return cls()


def test_load_model_explicit_gpu_success(monkeypatch, tmp_path):
    """GPU 可用 + 非 LoRA 目录 → 显式 cuda:0 加载成功 → first_param_device=cuda:0。"""
    (tmp_path / "config.json").write_text("{}")
    svc = _import_service(monkeypatch)
    monkeypatch.setattr(svc.settings, "MODEL_PATH", str(tmp_path))
    monkeypatch.setattr(svc.settings, "BASE_MODEL_PATH", str(tmp_path))
    monkeypatch.setattr(svc.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(svc, "_model", None)
    monkeypatch.setattr(svc, "_processor", None)
    # patch ColPali/ColPaliProcessor 在 svc 模块命名空间（已 import 绑定）
    monkeypatch.setattr(svc, "ColPali", _make_gpu_model_cls())
    monkeypatch.setattr(svc, "ColPaliProcessor", _FakeProcessorCls)

    svc._load_model()
    assert svc.get_model_device() == "cuda:0"


def test_load_model_gpu_oom_raises(monkeypatch, tmp_path):
    """GPU 可用但 OOM → 抛 RuntimeError，不静默 fallback 到 CPU。"""
    (tmp_path / "config.json").write_text("{}")
    svc = _import_service(monkeypatch)
    monkeypatch.setattr(svc.settings, "MODEL_PATH", str(tmp_path))
    monkeypatch.setattr(svc.settings, "BASE_MODEL_PATH", str(tmp_path))
    monkeypatch.setattr(svc.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(svc, "_model", None)
    monkeypatch.setattr(svc, "_processor", None)

    class _OOM:
        def to(self, device): raise svc.torch.cuda.OutOfMemoryError("OOM")
        def eval(self): return self
        @classmethod
        def from_pretrained(cls, *a, **kw): return cls()

    monkeypatch.setattr(svc, "ColPali", _OOM)
    monkeypatch.setattr(svc, "ColPaliProcessor", _FakeProcessorCls)

    with pytest.raises(RuntimeError, match="cuda:0"):
        svc._load_model()


def test_load_model_cpu_only_env(monkeypatch, tmp_path):
    """CPU-only 环境（无 CUDA）→ 加载成功，first_param_device=cpu。"""
    (tmp_path / "config.json").write_text("{}")
    svc = _import_service(monkeypatch)
    monkeypatch.setattr(svc.settings, "MODEL_PATH", str(tmp_path))
    monkeypatch.setattr(svc.settings, "BASE_MODEL_PATH", str(tmp_path))
    monkeypatch.setattr(svc.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(svc, "_model", None)
    monkeypatch.setattr(svc, "_processor", None)

    class _CPU:
        def eval(self): return self
        def parameters(self): yield _FakeParam("cpu")
        @classmethod
        def from_pretrained(cls, *a, **kw): return cls()

    monkeypatch.setattr(svc, "ColPali", _CPU)
    monkeypatch.setattr(svc, "ColPaliProcessor", _FakeProcessorCls)

    svc._load_model()
    assert svc.get_model_device() == "cpu"
