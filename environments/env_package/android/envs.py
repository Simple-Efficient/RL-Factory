import ray
import numpy as np
from environments.env_package.android.android_env import AndroidEnv
from environments.env_package.android.env_config import AndroidEnvConfig
from typing import Dict, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


#  Worker Actor
@ray.remote(num_cpus=0.25)
class AndroidWorker:

    def __init__(self, env_cfg: AndroidEnvConfig):
        """Initialize the Sokoban environment in this worker"""
        self.env = AndroidEnv(env_cfg)

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict, Dict]:
        return self.env.reset(seed)

    def step(self, llm_raw_response: str) -> Tuple[Dict, float, bool, Dict]:
        return self.env.step(llm_raw_response)

    def close(self):
        self.env.close()


class AndroidMultiProcessEnv:

    def __init__(self,
                 seed=0,
                 env_num=4,
                 group_n=2,
                 is_train=True,
                 env_kwargs=None):
        """
        - env_num: Number of different environments
        - group_n: Number of same environments in each group (for GRPO and GiGPO)
        - env_kwargs: Dictionary of parameters for initializing AndroidEnv
        - seed: Random seed for reproducibility
        - num_processes = env_num * group_n
        """
        super().__init__()

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        self.is_train = is_train
        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        np.random.seed(seed)

        if env_kwargs is None:
            env_kwargs = {}

        # Create Ray remote actors instead of processes
        self.workers = []  # 保存所有远程 worker 的句柄
        for _ in range(self.num_processes):
            if isinstance(env_kwargs, AndroidEnvConfig):
                env_cfg = env_kwargs
            else:
                env_cfg = AndroidEnvConfig(**env_kwargs)

            worker = AndroidWorker.remote(env_cfg)
            self.workers.append(worker)

    def step(self, llm_raw_responses):
        assert len(llm_raw_responses) == self.num_processes
        logger.debug(f"[Env] Sending step to {self.num_processes} workers.")
        # Send step commands to all workers
        futures = []

        for i, (worker, llm_raw_response) in enumerate(zip(self.workers, llm_raw_responses)):
            logger.debug(f"[Env] Step -> worker {i}")
            future = worker.step.remote(llm_raw_response)
            futures.append(future)

        results = ray.get(futures)
        logger.debug("[Env] Step completed for all workers.")

        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            logger.debug(f"[Env] Result from worker {i}: reward={reward}, done={done}")
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
        return obs_list, reward_list, done_list, info_list

    def reset(self):
        if self.is_train:
            seeds = np.random.randint(0, 2 ** 16 - 1, size=self.env_num)
        else:
            seeds = np.random.randint(2 ** 16, 2 ** 32 - 1, size=self.env_num)

        # repeat the seeds for each group
        seeds = np.repeat(seeds, self.group_n)
        seeds = seeds.tolist()

        logger.info(f"[Env] Resetting {self.num_processes} environments with seeds: {seeds}")

        # Send reset commands to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            logger.debug(f"[Env] Sending reset to worker {i} with seed {seeds[i]}")
            future = worker.reset.remote(seeds[i])
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        logger.info("[Env] Reset completed for all workers.")

        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)
        return obs_list, info_list

    def close(self):
        """
        Gracefully close all Ray actor environments by calling their .close() method.
        Then kill the actors to free Ray resources.
        """
        logger.info(f"[Env] Closing {self.num_processes} workers...")
        futures = []
        for i, worker in enumerate(self.workers):
            try:
                logger.debug(f"[Env] Sending close to worker {i}")
                futures.append(worker.close.remote())
            except Exception as e:
                logger.warning(f"[RayEnv] Failed to send close to worker {i}: {e}")

        try:
            ray.get(futures)
            logger.info("[Env] Graceful close completed.")
        except Exception as e:
            logger.warning(f"[RayEnv] Error during close wait: {e}")

        for i, worker in enumerate(self.workers):
            try:
                ray.kill(worker)
                logger.debug(f"[Env] Killed worker {i}")
            except Exception as e:
                logger.warning(f"[RayEnv] Failed to kill worker {i}: {e}")

    def __del__(self):
        self.close()


def build_android_envs(
        seed=0,
        env_num=1,
        group_n=1,
        is_train=True,
        env_kwargs=None):
    return AndroidMultiProcessEnv(seed, env_num, group_n, is_train, env_kwargs=env_kwargs)



