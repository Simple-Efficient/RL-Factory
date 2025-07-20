from .base import Env as BaseEnv
from .search import SearchEnv
from .vision import VisionEnv
from .reward_rollout_example import RewardRolloutEnv

__all__ = ['BaseEnv', 'SearchEnv', 'RewardRolloutEnv', 'VisionEnv']

TOOL_ENV_REGISTRY = {
    'base': BaseEnv,
    'search': SearchEnv,
    'reward_rollout': RewardRolloutEnv,
    'vision': VisionEnv
}