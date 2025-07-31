from .base import Env as BaseEnv
from .mmbase import MMEnv
from .search import SearchEnv
from .vision import VisionEnv
from .reward_rollout_example import RewardRolloutEnv


__all__ = ['BaseEnv', 'SearchEnv', 'RewardRolloutEnv', 'VisionEnv', 'MMEnv']

TOOL_ENV_REGISTRY = {
    'base': BaseEnv,
    'mmbase': MMEnv,
    'search': SearchEnv,
    'reward_rollout': RewardRolloutEnv,
    'vision': VisionEnv
}