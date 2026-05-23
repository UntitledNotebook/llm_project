# Environment setup with `uv`

This project targets Choice A Part 1 and Part 2 only: SFT and GRPO. The course PDF recommends Python 3.10, `vllm==0.9.0`, `transformers==4.51.1`, `datasets`, `torchdata`, `deepspeed`, and a pre-built `flash-attn` wheel that matches CUDA/PyTorch/Python/CXX11 ABI.

The server information you provided is an 8 × RTX 4090 machine with NVIDIA driver `550.54.14` and `nvidia-smi` reporting CUDA `12.4`. The most important rule is that the flash-attn wheel must match the **actual PyTorch runtime** reported by `torch.version.cuda`, not just the CUDA version shown by `nvidia-smi`.

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

The course flash-attn example uses a `torch2.7` wheel tag, so start with PyTorch 2.7.0. Official PyTorch 2.7.0 Linux wheels are provided for CUDA 11.8, 12.6, and 12.8; there is no official `cu124` PyTorch 2.7.0 wheel. Because your driver reports CUDA 12.4, the CUDA 12.6 wheel may require a newer driver on some systems. Try CUDA 12.6 first if your cluster supports it; otherwise use CUDA 11.8 or ask the admin to upgrade the driver.

Preferred course-aligned attempt:

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu126 \
  torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0
```

Fallback if CUDA initialization fails with a driver/runtime error:

```bash
uv pip uninstall -y torch torchvision torchaudio
uv pip install --index-url https://download.pytorch.org/whl/cu118 \
  torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0
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

## 4. Determine and install the correct flash-attn wheel

Run the helper after PyTorch is installed:

```bash
python scripts/resolve_flash_attn_wheel.py
```

It prints a filename like one of these:

```text
flash_attn-2.8.2+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
flash_attn-2.8.2+cu11torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

Download and install exactly that wheel from the Dao-AILab release:

```bash
FLASH_ATTN_VERSION=2.8.2
WHEEL=$(python scripts/resolve_flash_attn_wheel.py --filename-only --flash-version ${FLASH_ATTN_VERSION})
wget -c "https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VERSION}/${WHEEL}"
uv pip install "./${WHEEL}" --no-build-isolation
```

The course PDF example for Python 3.10 + PyTorch 2.7 + CUDA-major-12 + CXX11 ABI false is:

```bash
wget -c "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.2/flash_attn-2.8.2+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
uv pip install "./flash_attn-2.8.2+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" --no-build-isolation
```

If the release page does not contain your exact filename, change PyTorch/Python to a tag that has a pre-built wheel. Compiling flash-attn from source is intentionally avoided because it can take a long time.

## 5. Verify the full stack

```bash
python scripts/doctor.py

python - <<'PY'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = "Qwen/Qwen2.5-1.5B-Base"
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

`CUDA driver version is insufficient for CUDA runtime version`: install the `cu118` PyTorch 2.7 wheel or upgrade the NVIDIA driver before using `cu126`/`cu128`.

`undefined symbol` or `GLIBCXX` errors from `flash_attn_2_cuda`: the flash-attn wheel does not match your PyTorch CXX11 ABI or CUDA tag. Re-run `python scripts/resolve_flash_attn_wheel.py` and install the exact wheel.

`ImportError: flash_attn`: either install the matching wheel or set `model.attn_implementation: eager` in `configs/sft.yaml`, `configs/grpo.yaml`, and `configs/eval.yaml` for a slower no-flash-attn run.

DeepSpeed launch hangs: verify `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`, `NCCL_P2P_DISABLE=0`, and that no other process is using the GPUs. For debugging, run with `--num_gpus 1` first.
