from llm_project.rollout.types import RolloutBackend, RolloutBatch, RolloutSamplingConfig

__all__ = [
    "RolloutBackend",
    "RolloutBatch",
    "RolloutSamplingConfig",
    "create_rollout_backend",
    "rollout_sampling_from_config",
]


def create_rollout_backend(*args, **kwargs):
    from llm_project.rollout.factory import create_rollout_backend as _create_rollout_backend

    return _create_rollout_backend(*args, **kwargs)


def rollout_sampling_from_config(*args, **kwargs):
    from llm_project.rollout.factory import rollout_sampling_from_config as _rollout_sampling_from_config

    return _rollout_sampling_from_config(*args, **kwargs)
