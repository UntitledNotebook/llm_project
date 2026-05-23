#!/usr/bin/env python
"""Environment sanity checks for the LLM project stack."""
from __future__ import annotations

import importlib
import platform
import subprocess
import sys


def run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:  # noqa: BLE001
        return f"FAILED: {exc}"


def main() -> None:
    print("Python:", sys.version.replace("\n", " "))
    print("Platform:", platform.platform())
    print("nvidia-smi:")
    print(run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]))
    for module_name in ["torch", "transformers", "datasets", "deepspeed", "flash_attn", "vllm"]:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "unknown")
            print(f"{module_name}: {version}")
        except Exception as exc:  # noqa: BLE001
            print(f"{module_name}: NOT IMPORTABLE ({exc})")
    try:
        import torch

        print("torch.cuda.is_available:", torch.cuda.is_available())
        print("torch.version.cuda:", torch.version.cuda)
        print("torch._C._GLIBCXX_USE_CXX11_ABI:", torch._C._GLIBCXX_USE_CXX11_ABI)
        if torch.cuda.is_available():
            print("GPU count:", torch.cuda.device_count())
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                print(f"GPU {idx}: {props.name}, {props.total_memory / 2**30:.1f} GiB")
    except Exception as exc:  # noqa: BLE001
        print("Torch CUDA probe failed:", repr(exc))


if __name__ == "__main__":
    main()
