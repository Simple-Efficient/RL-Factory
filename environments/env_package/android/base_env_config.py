from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Optional, List, Union
import random


@dataclass
class BaseEnvConfig(ABC):
    format_reward: float = 0.5
    image_placeholder: str = "<image>"
    special_token_list: Optional[List[str]] = field(
        default_factory=lambda: ["<think>", "</think>", "<answer>", "</answer>"])
    action_sep: str = ","

    @abstractmethod
    def config_id(self) -> str:  # config identifier, wandb and mllm rollout manager use this to identify the config
        pass

    def get(self, key, default=None):
        """
        Get the value of a config key.
        Args:
            key: Key to get
            default: Default value if key is not found
        """
        return getattr(self, key, default)

    def generate_seeds(self, size, seed=0, n_candidate: int = 20000, ) -> list:
        # you can define it in your own env_config to support customized seed geenration
        random.seed(seed)
        seeds = random.sample(range(0, n_candidate + size), size)
        return seeds