"""
A/B 验证脚本：probe_triton_spec.py
目的：在不修改任何业务文件的前提下，验证 triton.__spec__ 问题是否真实存在。

实验设计：
  Case A（补丁开启）：先打补丁再 import torch，记录 __spec__ 状态与 CUDA 初始化结果
  Case B（补丁绕过）：不打补丁直接 import torch，记录 __spec__ 状态与 CUDA 初始化结果

用法：
  python probe_triton_spec.py          # Case A（默认，补丁开启）
  python probe_triton_spec.py --no-patch  # Case B（补丁绕过）
  python probe_triton_spec.py --both      # 在子进程中分别跑 A/B 并汇总
"""
import sys, os, json, subprocess, textwrap

# ── 单次探测逻辑（被子进程调用时直接执行）──────────────────────────────────
def _run_probe(apply_patch: bool) -> dict:
    result = {
        "apply_patch": apply_patch,
        "triton_available": False,
        "spec_before_patch": None,
        "spec_after_patch": None,
        "torch_imported": False,
        "cuda_available": None,
        "cuda_init_ok": None,
        "cuda_init_error": None,
        "deferred_cuda_error": None,
    }

    # 1. 检查 triton 是否可导入
    try:
        import triton as _triton
        result["triton_available"] = True
        result["spec_before_patch"] = repr(_triton.__spec__)
    except ImportError:
        result["triton_available"] = False
        # triton 不存在时补丁无意义，直接跳过
        apply_patch = False

    # 2. 可选：打补丁
    if apply_patch and result["triton_available"]:
        try:
            import triton as _triton
            if _triton.__spec__ is None:
                import importlib as _il
                _triton.__spec__ = _il.util.spec_from_file_location("triton", _triton.__file__)
            result["spec_after_patch"] = repr(_triton.__spec__)
        except Exception as e:
            result["spec_after_patch"] = f"PATCH_ERROR: {e}"
    else:
        result["spec_after_patch"] = result["spec_before_patch"]

    # 3. import torch（触发 CUDA lazy init 注册）
    try:
        import torch
        result["torch_imported"] = True
        result["cuda_available"] = torch.cuda.is_available()
    except Exception as e:
        result["torch_imported"] = False
        result["cuda_init_error"] = str(e)
        return result

    # 4. 触发真实 CUDA 初始化（等价于模型加载时的第一次 CUDA 调用）
    if result["cuda_available"]:
        try:
            _ = torch.zeros(1, device="cuda:0")
            result["cuda_init_ok"] = True
        except Exception as e:
            err = str(e)
            result["cuda_init_ok"] = False
            result["cuda_init_error"] = err
            if "DeferredCudaCallError" in err or "__spec__" in err:
                result["deferred_cuda_error"] = err
    else:
        result["cuda_init_ok"] = None  # 无 GPU，跳过

    return result


# ── 子进程入口 ─────────────────────────────────────────────────────────────
if __name__ == "__main__" and os.environ.get("_PROBE_SUBPROCESS"):
    apply = os.environ["_PROBE_SUBPROCESS"] == "patch"
    out = _run_probe(apply)
    print(json.dumps(out))
    sys.exit(0)


# ── 主入口 ─────────────────────────────────────────────────────────────────
def _run_subprocess(apply_patch: bool) -> dict:
    env = os.environ.copy()
    env["_PROBE_SUBPROCESS"] = "patch" if apply_patch else "nopatch"
    # 切换到项目目录，确保 .env 与 src 包可被找到
    cwd = os.path.dirname(os.path.abspath(__file__))
    proc = subprocess.run(
        [sys.executable, __file__],
        capture_output=True, text=True, env=env, cwd=cwd
    )
    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()
    try:
        data = json.loads(stdout)
    except Exception:
        data = {"parse_error": stdout, "stderr": stderr}
    data["_stderr"] = stderr
    data["_returncode"] = proc.returncode
    return data


def _print_result(label: str, d: dict):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    for k, v in d.items():
        if k.startswith("_stderr"):
            if v:
                print(f"  [stderr excerpt]\n{textwrap.indent(v[:800], '    ')}")
        else:
            print(f"  {k:30s}: {v}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--patch"

    if mode == "--both":
        print("\n>>> Case A: 补丁开启 (apply_patch=True)")
        a = _run_subprocess(True)
        _print_result("Case A — 补丁开启", a)

        print("\n>>> Case B: 补丁绕过 (apply_patch=False)")
        b = _run_subprocess(False)
        _print_result("Case B — 补丁绕过", b)

        # 汇总判断
        print(f"\n{'='*60}")
        print("  汇总判断")
        print(f"{'='*60}")
        spec_none_without = (b.get("spec_before_patch") == "None")
        deferred_without  = bool(b.get("deferred_cuda_error"))
        cuda_ok_with      = a.get("cuda_init_ok") is True
        cuda_ok_without   = b.get("cuda_init_ok") is True

        print(f"  无补丁时 triton.__spec__ is None : {spec_none_without}")
        print(f"  无补丁时出现 DeferredCudaCallError: {deferred_without}")
        print(f"  有补丁时 CUDA 初始化成功          : {cuda_ok_with}")
        print(f"  无补丁时 CUDA 初始化成功          : {cuda_ok_without}")

        if spec_none_without and deferred_without and cuda_ok_with:
            verdict = "✅ 已证实：triton.__spec__ is None 会导致 DeferredCudaCallError，补丁有效"
        elif spec_none_without and not deferred_without:
            verdict = "⚠️  未证实：triton.__spec__ is None 存在，但未触发 DeferredCudaCallError（可能需要 uvicorn worker 环境才能复现）"
        elif not spec_none_without:
            verdict = "❌ 无法证实：当前环境 triton.__spec__ 不为 None，问题前提条件不满足"
        else:
            verdict = "⚠️  部分证据：需结合完整启动日志进一步判断"

        print(f"\n  结论: {verdict}\n")

    elif mode == "--no-patch":
        b = _run_subprocess(False)
        _print_result("Case B — 补丁绕过", b)
    else:
        a = _run_subprocess(True)
        _print_result("Case A — 补丁开启", a)
