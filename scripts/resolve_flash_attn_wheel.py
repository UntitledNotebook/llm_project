#!/usr/bin/env python
"""Print the flash-attn wheel filename matching the active Python/PyTorch ABI.

Run this *after* installing PyTorch. The resulting filename follows the Dao-AILab release
convention, for example:
flash_attn-2.8.2+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
"""
from __future__ import annotations

import argparse
import platform
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flash-version", default="2.8.2")
    parser.add_argument("--filename-only", action="store_true")
    args = parser.parse_args()

    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        raise SystemExit("Install torch before resolving a flash-attn wheel: " + repr(exc)) from exc

    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    torch_major_minor = ".".join(torch.__version__.split("+")[0].split(".")[:2])
    cuda = torch.version.cuda or "0.0"
    # Dao-AILab flash-attn release wheels use CUDA-major tags such as cu12,
    # even when PyTorch itself was installed from a cu124 or cu126 index.
    cuda_major = cuda.split(".")[0]
    abi = "TRUE" if bool(torch._C._GLIBCXX_USE_CXX11_ABI) else "FALSE"
    machine = platform.machine()
    if machine not in {"x86_64", "aarch64"}:
        raise SystemExit(f"Unsupported machine tag for prebuilt flash-attn wheels: {machine}")
    platform_tag = f"linux_{machine}"
    filename = (
        f"flash_attn-{args.flash_version}+cu{cuda_major}torch{torch_major_minor}"
        f"cxx11abi{abi}-{py_tag}-{py_tag}-{platform_tag}.whl"
    )
    if args.filename_only:
        print(filename)
        return
    print("Detected environment:")
    print(f"  Python tag:        {py_tag}")
    print(f"  torch.__version__: {torch.__version__}")
    print(f"  torch.version.cuda:{cuda}")
    print(f"  CXX11 ABI:         {abi}")
    print(f"  Platform tag:      {platform_tag}")
    print("\nSuggested wheel filename:")
    print(filename)
    print("\nSuggested release URL:")
    print(f"https://github.com/Dao-AILab/flash-attention/releases/download/v{args.flash_version}/{filename}")


if __name__ == "__main__":
    main()
