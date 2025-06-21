from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Dict


class BaseEnv(ABC):
    @abstractmethod
    def step(self, llm_raw_response) -> Tuple[Dict, float, bool, Dict]:
        """
        action is llm raw response
        Args:
            action: Action to take, assume it's raw llm action

        Returns:
            obs, reward, done, info

        obs: {
            'obs_str': "This is the obs template, you see <image> and <image>, you heard <audio> and <audio>",
            'multi_modal_data':{
                '<image>':[list of images],
                '<audio>':[list of audios],
            }
            # num of <image> and <audio> in the obs_str should match len(multi_modal_data['<image>']) and len(multi_modal_data['<audio>'])
        }
        info: {
            "metrics": {
                'success': success,
                'action_is_effective': action_is_effective,
                'action_is_valid': action_is_valid,
            } # metrics you want to log in wandb
            "llm_raw_response": llm_raw_response,
            "llm_response": llm_response, # for update
        }
        """
        pass

    @abstractmethod
    def close(self):
        """Close the environment."""
        pass

    @abstractmethod
    def reset(self, seed=None) -> Tuple[Dict, Dict]:
        """
        Reset the environment.
        NOTE: the environment should be same for the same seed
        Args:
            seed: Seed for the environment

        Returns:
            obs,info

            format should be same as step
        """
        pass

    @abstractmethod
    def system_prompt(self) -> str:
        """
        Get the system prompt for the environment.

        Returns:
            System prompt string.
        """
        pass

    def compute_reward(self) -> float:
        """
        give final reward
        Currently the reward calculation in rollout manager will be: sum(step_rewards)+env.compute_reward()
        In most cases you can set this to 0.0 since the step rewards are already accumulated, but if you want to add some extra reward for the final step, define it here.
        """
        return 0.0