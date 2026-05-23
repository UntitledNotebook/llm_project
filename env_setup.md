# Environment setup with `uv`

This project targets Choice A Part 1 and Part 2 only: SFT and GRPO. The course PDF recommends Python 3.10, `vllm==0.9.0`, `transformers==4.51.1`, `datasets`, `torchdata`, `deepspeed`, and a pre-built `flash-attn` wheel that matches CUDA/PyTorch/Python/CXX11 ABI.

For this RTX 4090 server, use the CUDA 12.4-compatible PyTorch 2.6 stack instead of the course's PyTorch 2.7 example. `nvidia-smi` reports driver `550.54.14` and CUDA `12.4`; official PyTorch 2.6 wheels include a `cu124` build, while the PyTorch 2.7 wheels do not. Because `vllm==0.9.0` hard-pins `torch==2.7.0`, this environment uses `vllm==0.8.5`, which hard-pins `torch==2.6.0`.

Use `flash-attn==2.8.2` for this machine. The official Dao-AILab `2.8.2` pre-built `cu12torch2.6` wheels were tested here and failed to import against `torch==2.6.0+cu124` with an undefined `c10::Error(...std::__cxx11...)` symbol, so the verified path is to build `2.8.2` locally against the installed PyTorch ABI.

The most important flash-attn rule is that the wheel must match the **actual PyTorch runtime** reported by `torch.version.cuda`, the Python tag, and `torch._C._GLIBCXX_USE_CXX11_ABI`. Flash-attn release wheels use CUDA-major tags such as `cu12`, even when PyTorch was installed from the `cu124` index.

## 1. Create a clean Python 3.10 virtual environment

```bash
# Install uv if it is not already available.
curl -LsSf https://astral.sh/uv/install.sh | sh

# From the project root.
uv python install 3.10
uv venv .venv --python 3.10
source .venv/bin/activate
python --version
```

Expected Python line: `Python 3.10.x`.

## 2. Install PyTorch first

Use PyTorch 2.6.0 from the official CUDA 12.4 wheel index:

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
```

Check the installed PyTorch runtime:

```bash
python - <<'PY'
import sys, torch
print("python:", sys.version)
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
print("cxx11 abi:", torch._C._GLIBCXX_USE_CXX11_ABI)
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY
```

Do not continue until `torch.cuda.is_available()` is `True`.

## 3. Install the project dependencies

```bash
# Avoid optional DeepSpeed op compilation during installation; this framework uses torch AdamW.
DS_BUILD_OPS=0 uv pip install -r requirements/base.txt

# Install this source tree as an editable package.
uv pip install -e . --no-deps
```

`requirements/base.txt` intentionally uses `vllm==0.8.5` for this environment because `vllm==0.9.0` would replace PyTorch 2.6 with PyTorch 2.7.

## 4. Install flash-attn 2.8.2

The helper can still show the official Dao-AILab wheel filename for inspection:

```bash
python scripts/resolve_flash_attn_wheel.py
```

For Python 3.10 + PyTorch 2.6 + CUDA-major-12 + CXX11 ABI false, it prints:

```text
flash_attn-2.8.2+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

On this Ubuntu 22.04 / CUDA 12.4 machine, both official `2.8.2` `cxx11abiFALSE` and `cxx11abiTRUE` wheels were tested and failed with the same `std::__cxx11` PyTorch C++ symbol mismatch. Build from source instead so the extension is compiled against the installed `torch==2.6.0+cu124` ABI:

```bash
FLASH_ATTENTION_FORCE_BUILD=TRUE \
TORCH_CUDA_ARCH_LIST=8.9 \
MAX_JOBS=16 \
NVCC_THREADS=4 \
UV_LINK_MODE=copy \
uv pip install flash-attn==2.8.2 \
  --no-build-isolation \
  --no-deps \
  --reinstall \
  --no-binary flash-attn \
  --no-cache
```

Verify the import:

```bash
python - <<'VERIFY_PY'
import torch, flash_attn, flash_attn_2_cuda
print("torch:", torch.__version__, torch.version.cuda, torch._C._GLIBCXX_USE_CXX11_ABI)
print("flash_attn:", flash_attn.__version__)
print("extension:", flash_attn_2_cuda.__file__)
VERIFY_PY
```

## 5. Verify the full stack

```bash
python scripts/doctor.py

python - <<'PY'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = "Qwen/Qwen2.5-1.5B"
tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    trust_remote_code=True,
).cuda().eval()
inputs = tok("1 + 1 =", return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=4)
print(tok.decode(out[0]))
PY
```

## 6. Common fixes

`CUDA driver version is insufficient for CUDA runtime version`: verify that PyTorch is installed from the `cu124` index. Reinstall with the command in section 2 if another CUDA runtime was selected.

`undefined symbol` or `GLIBCXX` errors from `flash_attn_2_cuda`: the flash-attn binary does not match your PyTorch CXX11 ABI or CUDA tag. On this machine, build `flash-attn==2.8.2` from source with the command in section 4.

`ImportError: flash_attn`: either install the matching wheel or set `model.attn_implementation: eager` in `configs/sft.yaml`, `configs/grpo.yaml`, and `configs/eval.yaml` for a slower no-flash-attn run.

`vllm` tries to upgrade PyTorch to 2.7: make sure `requirements/base.txt` uses `vllm==0.8.5`, not `vllm==0.9.0`.

DeepSpeed launch hangs: verify `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`, `NCCL_P2P_DISABLE=0`, and that no other process is using the GPUs. For debugging, run with `--num_gpus 1` first.
