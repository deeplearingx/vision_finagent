#!/usr/bin/env python3
"""Startup script with triton __spec__ fix for PyTorch 2.7."""
import triton
import importlib
if triton.__spec__ is None:
    triton.__spec__ = importlib.util.find_spec('triton')

import uvicorn

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, timeout_keep_alive=300)
