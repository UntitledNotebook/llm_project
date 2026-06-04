from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_project.distributed import barrier, is_main_process, print_rank0


def save_hf_checkpoint(model_engine: Any, tokenizer: Any, output_dir: str | Path) -> None:
    """Save a Hugging Face-compatible checkpoint from a DeepSpeed engine.

    This is straightforward for ZeRO-1/2. The provided ZeRO-3 config enables parameter gathering on
    save, but for large custom modifications you may still prefer DeepSpeed's zero_to_fp32.py flow.
    """
    output_dir = Path(output_dir)
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        model_engine.module.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
        print_rank0(f"Saved HF checkpoint to {output_dir}")
    barrier()


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    if is_main_process():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    barrier()
