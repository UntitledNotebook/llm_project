from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Iterable

import torch

from llm_project.dtypes import require_torch_dtype


def iter_model_weights(
    model_engine: Any, *, cpu: bool = False
) -> Iterable[tuple[str, torch.Tensor]]:
    model = model_engine.module
    for name, parameter in model.named_parameters():
        tensor = parameter.detach()
        if cpu:
            tensor = tensor.cpu().contiguous()
        yield name, tensor


def get_vllm_model(llm: Any) -> Any:
    return llm.llm_engine.model_executor.driver_worker.model_runner.model


def reset_prefix_cache(llm: Any) -> None:
    llm.reset_prefix_cache(device=None)


def load_weights_into_vllm(llm: Any, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
    vllm_model = get_vllm_model(llm)
    vllm_model.load_weights(list(weights))
    reset_prefix_cache(llm)


def save_tmp_weights(model_engine: Any, tmp_dir: str | Path, *, step: int) -> Path:
    from safetensors.torch import save_file

    output_dir = Path(tmp_dir) / f"step_{int(step):06d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    weight_path = output_dir / "model.safetensors"
    state = {name: tensor for name, tensor in iter_model_weights(model_engine, cpu=True)}
    save_file(state, weight_path)
    return weight_path


def load_tmp_weights_into_vllm(llm: Any, weight_path: str | Path) -> None:
    from safetensors.torch import load_file

    weights = load_file(str(weight_path), device="cpu")
    load_weights_into_vllm(llm, weights.items())


def _create_stateless_group(host: str, port: int, rank: int, world_size: int):
    from vllm.distributed.utils import StatelessProcessGroup

    return StatelessProcessGroup.create(
        host=host, port=int(port), rank=int(rank), world_size=int(world_size)
    )


def _normalize_cuda_device(device: torch.device | str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    return resolved


def _create_pynccl(pg: Any, device: torch.device):
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    return PyNcclCommunicator(pg, device=device)


def _broadcast(comm: Any, tensor: torch.Tensor, *, src: int) -> None:
    comm.broadcast(tensor, src=src, stream=torch.cuda.current_stream(tensor.device))


class PyncclBroadcastClient:
    def __init__(self, *, host: str, port: int, device: torch.device | str) -> None:
        self.host = host
        self.port = int(port)
        self.device = _normalize_cuda_device(device)
        self._comm: Any | None = None

    def init(self) -> None:
        if self._comm is not None:
            return
        pg = _create_stateless_group(self.host, self.port, rank=1, world_size=2)
        self._comm = _create_pynccl(pg, self.device)

    def broadcast_model(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        if self._comm is None:
            self.init()
        assert self._comm is not None
        for _, tensor in weights:
            send_tensor = tensor.to(self.device)
            if not send_tensor.is_contiguous():
                send_tensor = send_tensor.contiguous()
            _broadcast(self._comm, send_tensor, src=1)
        torch.cuda.synchronize(self.device)


class PyncclReceiver:
    def __init__(self, *, device: torch.device | str) -> None:
        self.device = _normalize_cuda_device(device)
        self._lock = threading.Lock()
        self._comm: Any | None = None
        self._init_thread: threading.Thread | None = None
        self._init_error: BaseException | None = None
        self._receive_thread: threading.Thread | None = None
        self._receive_error: BaseException | None = None

    def start_init(self, *, host: str, port: int) -> None:
        with self._lock:
            if self._comm is not None or self._init_thread is not None:
                return

            def worker() -> None:
                try:
                    pg = _create_stateless_group(host, int(port), rank=0, world_size=2)
                    self._comm = _create_pynccl(pg, self.device)
                except BaseException as exc:  # noqa: BLE001
                    self._init_error = exc

            self._init_thread = threading.Thread(target=worker, name="pynccl-init", daemon=True)
            self._init_thread.start()

    def _wait_init(self, timeout: float = 300.0) -> None:
        deadline = time.monotonic() + timeout
        while self._comm is None and self._init_error is None:
            if time.monotonic() > deadline:
                raise TimeoutError("Timed out waiting for PyNccl receiver initialization")
            time.sleep(0.05)
        if self._init_error is not None:
            raise RuntimeError("PyNccl receiver initialization failed") from self._init_error

    def prepare_receive(
        self,
        llm: Any,
        *,
        weight_specs: list[dict[str, Any]],
    ) -> None:
        self._wait_init()
        with self._lock:
            if self._receive_thread is not None and self._receive_thread.is_alive():
                raise RuntimeError("A PyNccl receive is already in progress")
            self._receive_error = None
            model = get_vllm_model(llm)
            receive_specs = [
                (
                    str(spec["name"]),
                    tuple(int(dim) for dim in spec["shape"]),
                    require_torch_dtype(spec["dtype"]),
                )
                for spec in weight_specs
            ]

            def worker() -> None:
                try:
                    assert self._comm is not None
                    load_error: BaseException | None = None
                    for name, shape, dtype in receive_specs:
                        receive_tensor = torch.empty(shape, dtype=dtype, device=self.device)
                        _broadcast(self._comm, receive_tensor, src=1)
                        if load_error is None:
                            try:
                                model.load_weights([(name, receive_tensor)])
                            except BaseException as exc:  # noqa: BLE001
                                load_error = exc
                    if load_error is not None:
                        raise RuntimeError(
                            "vLLM load_weights failed during PyNccl receive"
                        ) from load_error
                    torch.cuda.synchronize(self.device)
                    reset_prefix_cache(llm)
                except BaseException as exc:  # noqa: BLE001
                    self._receive_error = exc

            self._receive_thread = threading.Thread(
                target=worker, name="pynccl-receive", daemon=True
            )
            self._receive_thread.start()

    def commit_receive(self, timeout: float = 600.0) -> None:
        thread = self._receive_thread
        if thread is None:
            raise RuntimeError("No PyNccl receive has been prepared")
        thread.join(timeout=timeout)
        if thread.is_alive():
            raise TimeoutError("Timed out waiting for PyNccl weight receive")
        if self._receive_error is not None:
            raise RuntimeError("PyNccl weight receive failed") from self._receive_error
