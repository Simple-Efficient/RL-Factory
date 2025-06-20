from typing import List, Tuple, Dict, Union, Any
from collections import defaultdict
import torch
import numpy as np
from functools import partial
import os
from environments.base import EnvironmentManagerBase, to_numpy
from PIL import Image


# -----------------------------------------------------------------------------
# Utility helpers ----------------------------------------------------------------

def parse_gamefile(infos: List[Dict[str, Any]]) -> List[Any]:
    """Extract the ``extra.gamefile`` field from each info dict (or ``None``)."""
    return [info.get("extra.gamefile") for info in infos]


def set_gamefile(infos: List[Dict[str, Any]], gamefile: List[Any]) -> List[Dict[str, Any]]:
    """Set/override the ``extra.gamefile`` entry inside each info dict."""
    for i in range(len(infos)):
        infos[i]["extra.gamefile"] = gamefile[i] if i < len(gamefile) else None
    return infos


# -----------------------------------------------------------------------------
# Android vector‑env manager ----------------------------------------------------
class AndroidEnvironmentManager(EnvironmentManagerBase):
    """
    Vector-env manager that wraps a list of AndroidEnv instances.

    Responsibilities
    ----------------
    1. Call underlying envs.reset / envs.step.
    2. Convert each single-env obs_dict to the multimodal format required
       by the LLM: {"text": prompt, "image": np.ndarray, "anchor": np.ndarray}.
    3. Batch rewards / dones / infos and add the `is_action_valid` flag.
    4. Keep a per-env history buffer for building prompts with dialog history.
    """

    def __init__(self, envs, projection_f, env_name):
        # history buffers
        self.buffers: List[List[Dict[str, Any]]] | None = None
        self.pre_text_obs: List[Dict[str, Any]] | None = None

        super().__init__(envs, projection_f, env_name)

        # public attribute often accessed by training loops/scripts

    # ------------------------------------------------------------------ reset
    def reset(self) -> Tuple[Dict[str, np.ndarray | List[str]], List[Dict]]:
        """Reset all underlying envs and return a batched multimodal obs."""
        obs_batch, infos = self.envs.reset()  # list[obs_dict]
        image_batch_np = self._images_to_numpy(obs_batch)  # (B, H, W, C)

        text_list = [obs["obs_str"] for obs in obs_batch]  # prompt for LLM

        observations = {
            "text": text_list,
            "image": image_batch_np,
            "anchor": image_batch_np  # identical copy; can be customised
        }

        # start fresh history buffers
        self.buffers = [[] for _ in range(len(obs_batch))]
        self.pre_text_obs = obs_batch

        return observations, infos

    # ------------------------------------------------------------------- step
    def step(
            self,
            llm_raw_responses: List[str]
    ) -> Tuple[Dict[str, np.ndarray | List[str]],
    np.ndarray,
    np.ndarray,
    List[Dict]]:
        """
        Batched env step driven by raw LLM responses.

        Returns
        -------
        next_obs_dict : multimodal dict with 'text' / 'image' / 'anchor'
        rewards       : np.ndarray, shape (B,)
        dones         : np.ndarray, shape (B,)
        infos         : list[dict], one per env
        """
        next_obs_batch, rewards, dones, infos = self.envs.step(llm_raw_responses)

        # mark whether each action was syntactically valid
        for info in infos:
            info["is_action_valid"] = info.get("metrics", {}) \
                .get("turn_metrics", {}) \
                .get("action_is_valid", False)

        image_batch_np = self._images_to_numpy(next_obs_batch)

        # push previous obs + current action into history
        self._save_to_history_buffer(self.pre_text_obs, llm_raw_responses)
        self.pre_text_obs = next_obs_batch

        text_list = [obs["obs_str"] for obs in next_obs_batch]

        next_observations = {
            "text": text_list,
            "image": image_batch_np,
            "anchor": image_batch_np
        }

        return next_observations, to_numpy(rewards), to_numpy(dones), infos

    def close(self):
        """
        Close all underlying AndroidEnv instances gracefully.
        """
        if hasattr(self.envs, "__iter__"):
            for env in self.envs:
                if hasattr(env, "close"):
                    env.close()
        elif hasattr(self.envs, "close"):
            self.envs.close()

    # ----------------------------------------------------------- helpers ----
    def _images_to_numpy(self, obs_batch: List[Dict]) -> np.ndarray:
        """
        Extract <image> token from each obs and convert to numpy array.
        If missing, return a placeholder black image of shape (720, 1280, 3).
        """
        img_list = []
        for obs in obs_batch:
            img = obs.get("multi_modal_data", {}).get("<image>", [])
            if img and isinstance(img[0], Image.Image):
                img_np = np.array(img[0].convert("RGB"), dtype=np.uint8)
            else:
                img_np = np.zeros((720, 1280, 3), dtype=np.uint8)
            img_list.append(img_np)
        return np.stack(img_list, axis=0)

    def _save_to_history_buffer(self,
                                obs_batch: List[Dict],
                                actions: List[str]) -> None:
        """
        Store (previous obs, current action) in per-env buffers.
        The raw LLM response is saved directly; parse if you need finer data.
        """
        if self.buffers is None:
            return
        for i, action in enumerate(actions):
            self.buffers[i].append({
                "text_obs": obs_batch[i]["obs_str"],
                "action": action
            })


def make_envs(config):
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")

    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1

    if "android" in config.env.env_name.lower():
        from environments.env_package.android import build_android_envs

        _envs = build_android_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n,
                                   is_train=True)
        _val_envs = build_android_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1,
                                       is_train=False)

        projection_f = lambda x: x  # default projection if not specified
        envs = AndroidEnvironmentManager(_envs, projection_f, config.env.env_name)
        val_envs = AndroidEnvironmentManager(_val_envs, projection_f, config.env.env_name)
        return envs, val_envs

    raise RuntimeError("Environment not supported: " + config.env.env_name)


if __name__ == "__main__":
    from environments.env_package.android import build_android_envs
    import base64
    from io import BytesIO
    from PIL import Image
    import json


    def make_action(x, y, think_text):
        return f"<think>{think_text}</think>\n<answer>{json.dumps({'action_type': 'click', 'x': x, 'y': y})}</answer>"


    def decode_and_save_image(b64_img: str, save_path: str) -> None:
        img_data = base64.b64decode(b64_img)
        img = Image.open(BytesIO(img_data))
        img.save(f"{save_path}.png")
        print(f"[Saved] {save_path}.png")


    env_num = 2
    envs = build_android_envs(seed=42, env_num=env_num, group_n=1, is_train=True, env_kwargs=None)
    projection_f = lambda x: x
    env_manager = AndroidEnvironmentManager(envs, projection_f, "android")

    observations, infos = env_manager.reset()

    action_sequences = [
        [make_action(177, 404, "点击顶部的搜索按钮"), make_action(405, 2274, "点击 Shorts"),
         make_action(675, 2274, "点击 Subscriptions")],
        [make_action(135, 2274, "点击 Home"), make_action(405, 2274, "点击 Shorts"),
         make_action(945, 2274, "点击 Library")],
    ]

    for step_idx in range(3):
        print(f"\n=== [STEP {step_idx + 1}] ===")
        try:
            actions = [action_sequences[i][step_idx] for i in range(env_num)]
            for i, a in enumerate(actions):
                print(f"[Env-{i}] Action: {a}")
            next_obs, rewards, dones, infos = env_manager.step(actions)
            for i, (r, d, info) in enumerate(zip(rewards, dones, infos)):
                print(
                    f"[Env-{i}] Reward={float(r):.2f}, Done={bool(d)}, Valid={info.get('is_action_valid')}, Feedback={info.get('env_feedback')}")
                b64_img = info.get("screenshot_b64")
                if b64_img:
                    decode_and_save_image(b64_img, f"env{i}_step{step_idx + 1}")
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"[ERROR] Step {step_idx + 1} 执行失败: {e}")

    env_manager.close()
    print("测试完成")