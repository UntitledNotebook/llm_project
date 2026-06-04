from __future__ import annotations

import argparse
import os
from typing import Any

import torch

from llm_project.config import Config
from llm_project.models import load_tokenizer
from llm_project.rollout.formatting import encode_prompt_token_ids
from llm_project.rollout.types import RolloutSamplingConfig
from llm_project.rollout.vllm_common import (
    flatten_vllm_outputs,
    generate_with_vllm,
    make_sampling_params,
    vllm_init_kwargs,
)
from llm_project.rollout.weight_sync import (
    PyncclReceiver,
    load_tmp_weights_into_vllm,
    reset_prefix_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small vLLM rollout server for GRPO training")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--no_trust_remote_code", action="store_false", dest="trust_remote_code")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--max_num_seqs", type=int, default=64)
    parser.add_argument("--download_dir", type=str, default=None)
    parser.add_argument("--pynccl_device", type=str, default="cuda")
    return parser.parse_args()


def build_app(args: argparse.Namespace):
    from fastapi import FastAPI, HTTPException
    from vllm import LLM

    tokenizer_name = args.tokenizer or args.model
    tokenizer = load_tokenizer(
        tokenizer_name,
        trust_remote_code=bool(args.trust_remote_code),
        padding_side="left",
    )
    cfg = Config(
        {
            "model": {
                "trust_remote_code": bool(args.trust_remote_code),
                "dtype": args.dtype,
            },
            "rollout": {},
        }
    )
    vllm_cfg = {
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "download_dir": args.download_dir
        or os.path.join(os.environ.get("HF_HOME", "/mnt/zhangchenming/.cache/huggingface"), "hub"),
    }
    llm = LLM(**vllm_init_kwargs(args.model, cfg, vllm_cfg))
    pynccl_receiver = PyncclReceiver(device=torch.device(args.pynccl_device))
    app = FastAPI(title="llm_project vLLM rollout server")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "model": args.model}

    @app.post("/generate")
    def generate(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            prompts = [str(item) for item in payload.get("prompts", [])]
            sampling = RolloutSamplingConfig(
                max_prompt_length=int(payload["max_prompt_length"]),
                max_new_tokens=int(payload["max_new_tokens"]),
                group_size=int(payload["group_size"]),
                temperature=float(payload["temperature"]),
                top_p=float(payload["top_p"]),
                do_sample=bool(payload.get("do_sample", True)),
                request_logprobs=bool(payload.get("request_logprobs", False)),
            )
            prompt_token_ids = encode_prompt_token_ids(tokenizer, prompts, sampling.max_prompt_length)
            sampling_params = make_sampling_params(sampling)
            outputs = generate_with_vllm(
                llm,
                prompt_token_ids=prompt_token_ids,
                sampling_params=sampling_params,
            )
            completions = flatten_vllm_outputs(outputs)
            return {"completions": completions}
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/sync/tmp")
    def sync_tmp(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            load_tmp_weights_into_vllm(llm, str(payload["path"]))
            return {"ok": True, "step": payload.get("step")}
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/sync/pynccl/init")
    def sync_pynccl_init(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            pynccl_receiver.start_init(host=str(payload["host"]), port=int(payload["port"]))
            return {"ok": True}
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/sync/pynccl/prepare")
    def sync_pynccl_prepare(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            weight_specs = [dict(item) for item in payload["weights"]]
            pynccl_receiver.prepare_receive(llm, weight_specs=weight_specs)
            return {"ok": True, "step": payload.get("step")}
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/sync/pynccl/commit")
    def sync_pynccl_commit(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            pynccl_receiver.commit_receive()
            return {"ok": True, "step": payload.get("step")}
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/reset_prefix_cache")
    def reset_cache() -> dict[str, Any]:
        try:
            reset_prefix_cache(llm)
            return {"ok": True}
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def main() -> None:
    import uvicorn

    args = parse_args()
    app = build_app(args)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
