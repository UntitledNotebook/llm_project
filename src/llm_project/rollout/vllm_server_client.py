from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

import torch
import torch.distributed as dist

from llm_project.distributed import barrier, global_rank, is_main_process, world_size
from llm_project.rollout.formatting import format_rollout_batch
from llm_project.rollout.types import RolloutBackend, RolloutBatch, RolloutSamplingConfig
from llm_project.rollout.weight_sync import PyncclBroadcastClient, iter_model_weights, save_tmp_weights


def _post_json(url: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url} failed with HTTP {exc.code}: {detail}") from exc
    if not raw:
        return {}
    return json.loads(raw)


def _get_json(url: str, *, timeout: float) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GET {url} failed with HTTP {exc.code}: {detail}") from exc
    if not raw:
        return {}
    return json.loads(raw)


class VLLMServerRolloutBackend(RolloutBackend):
    name = "vllm_server"

    def __init__(
        self,
        *,
        cfg: Any,
        tokenizer: Any,
        sampling: RolloutSamplingConfig,
        device: torch.device,
    ) -> None:
        self.tokenizer = tokenizer
        self.sampling = sampling
        self.device = device
        vllm_cfg = cfg.rollout.get("vllm", {})
        self.server_url = str(vllm_cfg.get("server_url", "http://127.0.0.1:8000")).rstrip("/")
        self.sync_method = str(vllm_cfg.get("sync_method", "tmp"))
        self.tmp_dir = str(vllm_cfg.get("tmp_dir", "/tmp/llm_project_vllm_rollout"))
        self.pynccl_host = str(vllm_cfg.get("pynccl_host", "127.0.0.1"))
        self.pynccl_port = int(vllm_cfg.get("pynccl_port", 29577))
        self.importance_sampling = bool(vllm_cfg.get("importance_sampling", True))
        self.importance_sampling_cap = float(vllm_cfg.get("importance_sampling_cap", 3.0))
        self._pynccl: PyncclBroadcastClient | None = None
        if self.sync_method not in {"tmp", "pynccl"}:
            raise ValueError(f"Unsupported vLLM server sync_method: {self.sync_method}")

    def health(self) -> None:
        _get_json(f"{self.server_url}/health", timeout=30.0)

    def _all_rank_prompts(self, prompts: list[str]) -> tuple[list[str], list[int]]:
        if not (dist.is_available() and dist.is_initialized()):
            return prompts, [len(prompts)]
        gathered: list[Any] = [None for _ in range(world_size())]
        dist.all_gather_object(gathered, list(prompts))
        counts = [len(items) for items in gathered]
        flat = [prompt for items in gathered for prompt in items]
        return flat, counts

    def _broadcast_rank_completions(
        self,
        completions: list[dict[str, Any]] | None,
        counts: list[int],
    ) -> list[dict[str, Any]]:
        if not (dist.is_available() and dist.is_initialized()):
            return completions or []
        payload: list[Any] = [None]
        if is_main_process():
            assert completions is not None
            offset = 0
            per_rank: list[list[dict[str, Any]]] = []
            for count in counts:
                length = count * self.sampling.group_size
                per_rank.append(completions[offset : offset + length])
                offset += length
            payload[0] = per_rank
        dist.broadcast_object_list(payload, src=0)
        return payload[0][global_rank()]

    def sync_weights(self, model_engine: Any, *, step: int) -> float:
        start = time.perf_counter()
        if self.sync_method == "tmp":
            if is_main_process():
                weight_path = save_tmp_weights(model_engine, self.tmp_dir, step=step)
                _post_json(
                    f"{self.server_url}/sync/tmp",
                    {"path": str(weight_path), "step": int(step)},
                    timeout=1800.0,
                )
            barrier()
            return time.perf_counter() - start

        if is_main_process():
            if self._pynccl is None:
                _post_json(
                    f"{self.server_url}/sync/pynccl/init",
                    {"host": self.pynccl_host, "port": self.pynccl_port},
                    timeout=30.0,
                )
                self._pynccl = PyncclBroadcastClient(
                    host=self.pynccl_host,
                    port=self.pynccl_port,
                    device=self.device,
                )
                self._pynccl.init()
            weights = list(iter_model_weights(model_engine, cpu=False))
            weight_specs = [
                {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
                for name, tensor in weights
            ]
            _post_json(
                f"{self.server_url}/sync/pynccl/prepare",
                {"step": int(step), "weights": weight_specs},
                timeout=60.0,
            )
            self._pynccl.broadcast_model(weights)
            _post_json(
                f"{self.server_url}/sync/pynccl/commit",
                {"step": int(step)},
                timeout=1800.0,
            )
        barrier()
        return time.perf_counter() - start

    def generate(self, prompts: list[str]) -> RolloutBatch:
        all_prompts, counts = self._all_rank_prompts(prompts)
        completions = None
        if is_main_process():
            payload = {
                "prompts": all_prompts,
                "max_prompt_length": self.sampling.max_prompt_length,
                "max_new_tokens": self.sampling.max_new_tokens,
                "group_size": self.sampling.group_size,
                "temperature": self.sampling.temperature,
                "top_p": self.sampling.top_p,
                "do_sample": self.sampling.do_sample,
                "request_logprobs": self.importance_sampling,
            }
            response = _post_json(f"{self.server_url}/generate", payload, timeout=1800.0)
            completions = response["completions"]
        rank_completions = self._broadcast_rank_completions(completions, counts)
        return format_rollout_batch(
            self.tokenizer,
            prompts,
            rank_completions,
            max_prompt_length=self.sampling.max_prompt_length,
            group_size=self.sampling.group_size,
        )

