"""
AndroidEnv
==========

A lightweight Android GUI RL environment that talks to a remote emulator
via REST (`/reset`, `/step`, `/close`) and evaluates rewards with an LLM.

Key features
------------
* Multimodal observation: <image> + task instruction + formatting guide.
* Single-step and episode-level rewards provided by `RewardEvaluator`
  (PRM = process reward, ORM = outcome reward).
* Self-logging of screenshots / metadata for offline inspection.

"""
from __future__ import annotations

# ───────────────────────────────────────── stdlib ─────────────────────────────────────────
import base64
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ──────────────────────────────────────── 3rd-party ───────────────────────────────────────
import jsonlines
import requests
from PIL import Image

# ───────────────────────────────────────── project ────────────────────────────────────────
from environments.env_package.android.base_env import BaseEnv
from environments.env_package.android.env_config import AndroidEnvConfig
from environments.env_package.android.reward_evaluator import RewardEvaluator
from environments.env_package.android.utils import (
    PARSE_FUNC_MAP,
    log_with_time,
    parse_llm_raw_response,
    call_model,
)
from environments.prompts.android import format_prompt, system_prompt

# ───────────────────────────────────────── logging ─────────────────────────────────────────
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ──────────────────────────────────────── constants ────────────────────────────────────────
TIMEOUT_SEC = int(os.getenv("ANDROID_HTTP_TIMEOUT", "300"))
RETRY_TIMES = int(os.getenv("ANDROID_HTTP_RETRY", "3"))
MAX_EPISODE_STEP = int(os.getenv("MAX_EPISODE_STEP", "20"))
FORMAT_REWARD = float(os.getenv("FORMAT_REWARD", "0.5"))


# --------------------------------------------------------------------------- #
# AndroidEnv                                                                  #
# --------------------------------------------------------------------------- #
class AndroidEnv(BaseEnv):
    def __init__(self, config: AndroidEnvConfig):
        """Android GUI environment with LLM-based reward evaluation."""
        super().__init__()
        self.config = config
        self.down_sample_ratio = getattr(config, "down_sample_ratio", 1.0)
        # I/O paths
        self.base_url = config.base_url
        self.data_path = Path(config.data_path)
        self.save_root = Path(config.save_root)
        self.cache_dir = Path(config.cache_dir)
        self.save_root.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Dataset
        self.task_dataset: List[Dict[str, Any]] = self._load_dataset()
        if not self.task_dataset:
            raise RuntimeError("Task dataset is empty!")

        # Reward evaluator (PRM + ORM)
        self.rew_eval = RewardEvaluator(call_modal)

        # Prompt helpers
        self.prompt_format = config.prompt_format
        self.format_prompt_func = format_prompt[self.prompt_format]
        self.parse_func = PARSE_FUNC_MAP[self.config.prompt_format]
        # ---------- runtime state ----------
        self.reset_episode_state()

        self._max_episode_steps = MAX_EPISODE_STEP

        self.action_sep = getattr(config, "action_sep", "")
        self.max_actions_per_step = getattr(config, "max_actions_per_step", "")
        self.special_token_list = None

        # ---------- others ----------
        self.rng = random.Random()
        self.task_name = ""
        self.standing = True
        self.image_placeholder = "<image>"
        self.format_reward = 0.5
        self.seed = 42
        self.number_of_episodes = len(self.task_dataset)

    def reset_episode_state(self):
        self.step_data_list = []
        self.screenshot_history = []
        self.rollout_id = ""
        self.device_id = ""
        self._current_step = 0
        self._rollout_start_time = time.time()
        self._episode_start_time = time.time()
        self.total_reward = 0.0
        self.done = False
        self.last_action = ""
        self.task_name = ""
        self.current_task = ""
        self.episode_language_instruction = ""
        self.valid_actions = []
        self.step_id = 0

    # ---------------- HTTP with retry ----------------
    def _request_with_retry(self, method: str, url: str, **kwargs):
        for attempt in range(1, RETRY_TIMES + 1):
            try:
                resp = requests.request(method, url, timeout=TIMEOUT_SEC, **kwargs)
                resp.raise_for_status()
                return resp
            except Exception as exc:
                logger.warning("[HTTP] %s attempt %d/%d failed: %s", url, attempt, RETRY_TIMES, exc)
                time.sleep(1)
        raise RuntimeError(f"HTTP request to {url} failed after {RETRY_TIMES} retries")

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict, Dict]:
        """Reset environment and return initial observation & info."""
        start_ts = time.time()
        log_with_time("[RESET] ---------------- start ----------------")

        # ------------------------------------------------------------------
        # 1. Select task
        # ------------------------------------------------------------------
        self.seed = seed
        if seed is not None:
            self.rng.seed(seed)
        idx = self.rng.randint(0, len(self.task_dataset) - 1)
        task = self.task_dataset[idx]

        # --- internal state ---
        self._current_step = 0
        self.step_data_list.clear()
        self._rollout_start_time = start_ts
        self.episode_data = task
        self.current_task = task.get("task_template", "Please begin the task.")
        self.task_name = task.get("task_name", f"task_{idx}")
        self.episode_language_instruction = self.current_task

        # ------------------------------------------------------------------
        # 2. Remote reset (with retry)
        # ------------------------------------------------------------------
        try:
            resp = self._request_with_retry("POST", f"{self.base_url}/reset")
            data = resp.json()
            self.rollout_id = data.get("rollout_id")
            self.device_id = data.get("device_id")
            screenshot = data.get("screenshot_b64")
        except Exception as e:
            raise RuntimeError(f"[RESET] Remote /reset failed: {e}") from e

        if not all([self.rollout_id, self.device_id, screenshot]):
            raise RuntimeError(f"[RESET] Missing fields in response: {data}")

        # ------------------------------------------------------------------
        # 3. Book‑keeping & disk I/O
        # ------------------------------------------------------------------
        self._save_image(screenshot, -1)
        self._save_step_data(-1, screenshot, None, 0.0, False, {"type": "init"})
        self.screenshot_history = [screenshot]

        if screenshot:
            obs = self._render(screenshot_b64=screenshot, init_obs=True)
        else:
            print(
                f"[WARN] {self.task_name}_{self.rollout_id}_{self.device_id}No screenshot from reset API, using fallback observation.")
            logger.debug(
                f"[WARN] {self.task_name}_{self.rollout_id}_{self.device_id}No screenshot from reset API, using fallback observation.")
            obs = {
                "obs_str": "Reset failed",
                "multi_modal_data": {},
            }

        reset_time = time.time() - start_ts
        info: Dict[str, Any] = {
            "rollout_id": self.rollout_id,
            "device_id": self.device_id,
            "instruction": self.episode_language_instruction,
            "task_index": idx,
            "task_id": task.get("task_id", f"task_{idx}"),
            "reset_success": True,
            "reset_time": reset_time,
            "screenshot_b64": screenshot
        }

        # ------------------------------------------------------------------
        # 4. Clear counters / logs
        # ------------------------------------------------------------------
        self.total_reward = 0.0
        self.reward = 0.0
        self.episode_log = []
        self.standing = True
        self._episode_start_time = time.time()

        logger.debug(
            f"[RESET] {self.task_name}_{self.rollout_id}_{self.device_id} finished in {reset_time:.3f}s"
        )
        self.save_rollout_to_disk()
        return obs, info

    def step(self, llm_raw_response: str) -> Tuple[Dict, float, bool, Dict]:
        """Take one environment step using the model response."""
        start_ts = time.time()
        log_with_time(f"[{self.rollout_id}] -------- step {self._current_step} start --------")

        info: Dict[str, Any] = {
            "metrics": {
                "turn_metrics": {
                    "action_is_valid": False,
                    "action_is_effective": False,
                    "action_executed": False,
                },
                "traj_metrics": {
                    "success": False,
                    "rollout_duration": None,
                },
            },
            "instruction": self.episode_language_instruction,
            "env_step": self._current_step,
            "episode_elapsed_seconds": time.time() - self._episode_start_time,
            "task_success": False,
            "env_feedback": "No action executed.",
            "fallback_used": False,
        }

        print(f"[DEBUG]{self.task_name}_{self.rollout_id}_{self.device_id} Raw LLM Response:", llm_raw_response)
        logger.debug(f"[DEBUG]{self.task_name}_{self.rollout_id}_{self.device_id} Raw LLM Response: {llm_raw_response}")
        img_before_b64 = self.screenshot_history[-1]

        # ------------------------------------------------------------------
        # 1. Parse LLM response → action dict
        # ------------------------------------------------------------------

        try:
            parsed_action = parse_llm_raw_response(llm_raw_response)
            action_dict = json.loads(parsed_action["action_content"])
            action_is_valid = True
        except Exception as exc:
            logger.warning("[%s] Action parse failed: %s — fallback click.", self.rollout_id, exc)
            action_dict = {"action_type": "click", "x": 400, "y": 800}
            action_is_valid = False
            info["fallback_used"] = True
        # record format parse result (e.g. format_correct) if available
        info.update(self.parse_func(
            response=llm_raw_response,
            special_token_list=self.special_token_list,
            action_sep=self.action_sep,
            max_actions=self.max_actions_per_step,
        ))

        # ------------------------------------------------------------------
        # 2. Send action to server
        # ------------------------------------------------------------------
        def _post_step():
            for n in range(RETRY_TIMES):
                try:
                    r = requests.post(
                        f"{self.base_url}/step",
                        json={"rollout_id": self.rollout_id, "action": action_dict},
                        timeout=TIMEOUT_SEC,
                    )
                    r.raise_for_status()
                    return r.json()
                except Exception as e:
                    logger.warning("/step retry %d/%d failed: %s", n + 1, RETRY_TIMES, e)
                    time.sleep(1)
            raise RuntimeError("/step failed after retries")

        try:
            resp_json = _post_step()
            action_executed = resp_json.get("ok", False)
            img_after_b64 = resp_json.get("after_b64") if action_executed else None
            ui_elements = resp_json.get("ui_elements") if action_executed else None

            os.makedirs("outputs", exist_ok=True)
            with open(f"outputs/ui_{self.rollout_id}.json", "w", encoding="utf-8") as f:
                json.dump(ui_elements, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.error("[%s] /step error: %s", self.rollout_id, exc)
            action_executed = False
            img_after_b64 = None

        # --------------------------------------------------------------
        # 3. reward & success eval
        # --------------------------------------------------------------
        step_success = task_success = 0.0
        step_score = task_score = 0.0

        if action_executed and img_after_b64:
            self.screenshot_history.append(img_after_b64)

            # ──────────────────────────── PRM ────────────────────────────
            step_success, step_score = self.rew_eval.evaluate(
                img_before_b64,  # <image1>
                img_after_b64,  # <image2>
                self.current_task,  # goal / instruction
                mode="process"  # PRM
            )

            # ──────────────────────────── ORM ────────────────────────────
            task_success, task_score = self.rew_eval.evaluate(
                img_before_b64,
                img_after_b64,
                self.current_task,
                mode="outcome"  # ORM
            )
        else:
            logger.debug("[%s] ineffective action or no screenshot", self.rollout_id)
            step_success = task_success = 0.0
            step_score = task_score = 0.0

        # --------------------------------------------------------------
        # 4. accumulate reward
        # --------------------------------------------------------------
        reward = step_score
        if task_success:
            reward += task_score
        if action_is_valid and info.get("format_correct", True):
            reward += self.format_reward
            info["is_format_rewarded"] = True
        else:
            info["is_format_rewarded"] = False

        self.total_reward += reward

        # --------------------------------------------------------------
        # 5. metrics bookkeeping
        # --------------------------------------------------------------
        turn_metrics = info["metrics"]["turn_metrics"]
        turn_metrics.update({
            "action_is_valid": action_is_valid,
            "action_executed": action_executed,
            "action_is_effective": bool(step_success),
        })

        done = bool(task_success) or (self._current_step + 1 >= self._max_episode_steps)
        if done:
            info["metrics"]["traj_metrics"]["success"] = bool(task_success)
            info["metrics"]["traj_metrics"]["rollout_duration"] = time.time() - self._rollout_start_time

        info.update({
            "task_success": bool(task_success),
            "env_feedback": "action ok" if action_executed else "action failed",
            "step_duration": time.time() - start_ts,

        })

        # --------------------------------------------------------------
        # 6. persist step & image
        # --------------------------------------------------------------
        if img_after_b64:
            self._save_step_data(self._current_step, img_after_b64, action_dict, reward, done, info)
            self._save_image(img_after_b64, self._current_step)
        info["screenshot_b64"] = img_after_b64

        # optional visual diff
        # if img_after_b64:
        #     same, diff = compare_image_similarity(img_before_b64, img_after_b64)
        #     logger.debug("[diff] ssim %.3f same=%s", diff, same)

        self._current_step += 1

        # --------------------------------------------------------------
        # 7. final obs
        # --------------------------------------------------------------
        next_obs = self._render(screenshot_b64=img_after_b64 or img_before_b64, init_obs=False)
        log_with_time(f"[{self.rollout_id}] <<< STEP {self._current_step} end (reward {reward:.2f})")
        return next_obs, reward, done, info

    def _save_image(self, b64_str, step_id: int):
        task_dir = os.path.join(self.save_root, f"{self.rollout_id}")
        os.makedirs(task_dir, exist_ok=True)
        path = os.path.join(task_dir, f"step_{step_id:03d}.png")
        with open(path, "wb") as f:
            f.write(base64.b64decode(b64_str))

    def _save_step_data(self, step_id, b64_img, action, reward, done, info):
        self.step_data_list.append({
            "rollout_id": self.rollout_id,
            "step_id": step_id,
            "image": b64_img,
            "action": json.dumps(action) if action else None,
            "reward": reward,
            "done": done,
            "info": info
        })

    def compute_reward(self) -> float:
        return 0.0

    def system_prompt(self) -> str:
        # Get format prompt with examples for system prompt
        format_prompt_text = self.format_prompt_func(
            max_actions_per_step=self.max_actions_per_step,
            action_sep=self.action_sep,
            add_example=True
        )

        return system_prompt(format=self.prompt_format) + '\n' + format_prompt_text

    def close(self):
        if self.device_id:
            try:
                requests.post(
                    f"{self.base_url}/close",
                    json={"rollout_id": self.rollout_id},
                    timeout=5000
                )
                print(f"[DEBUG] {self.task_name}_{self.rollout_id}_{self.device_id} Closed device {self.device_id}")
                logger.debug(
                    f"[DEBUG] {self.task_name}_{self.rollout_id}_{self.device_id} Closed device {self.device_id}")
            except Exception as e:
                print(f"{self.task_name}_{self.rollout_id}_{self.device_id}[WARNING] close failed: {e}")
                logger.debug(f"{self.task_name}_{self.rollout_id}_{self.device_id}[WARNING] close failed: {e}")

    def _render(self, screenshot_b64: Optional[str] = None, init_obs: bool = False) -> Dict:
        """
        Render an observation for the Android environment.
        """
        print(f"[DEBUG] {self.task_name}_{self.rollout_id}_{self.device_id}_render called. init_obs={type(init_obs)}")
        logger.debug(
            f"[DEBUG] {self.task_name}_{self.rollout_id}_{self.device_id}_render called. init_obs={type(init_obs)}")
        # 1. Decode image: prioritize using the provided screenshot
        if screenshot_b64 is not None:
            try:
                image_data = base64.b64decode(screenshot_b64)
                image = Image.open(BytesIO(image_data))
            except Exception as e:
                print(f"[WARNING] {self.task_name}_{self.rollout_id}_{self.device_id}Screenshot decode failed: {e}")
                logger.debug(
                    f"[WARNING] {self.task_name}_{self.rollout_id}_{self.device_id}Screenshot decode failed: {e}")
                image = Image.new("RGB", (720, 1280), color="gray")
        else:
            image = Image.new("RGB", (720, 1280), color="white")  # fallback

        # 2. Construct multimodal input: use placeholder token to represent image
        img_placeholder = self.image_placeholder  # e.g., "<image>"
        multi_modal_data = {
            img_placeholder: [image]
        }

        # 3. Construct text prompt (task instruction + format template)
        if init_obs:
            task_desc = self.current_task
            obs_str = f"{img_placeholder}\n{task_desc}"
        else:
            obs_str = f"{img_placeholder}\nPlease choose the next action."

        # 4. Add formatting guide (without examples)
        format_prompt = self.format_prompt_func(
            max_actions_per_step=self.max_actions_per_step,
            action_sep=self.action_sep,
            add_example=False
        )
        obs_str += "\n" + format_prompt

        # 5. Return the observation dictionary
        return {
            "obs_str": obs_str,  # Text prompt for the LLM
            "multi_modal_data": multi_modal_data  # Current screenshot for multimodal input
        }

    def _load_dataset(self):
        dataset = []
        with jsonlines.open(self.data_path) as reader:
            for obj in reader:
                dataset.append(obj)

        if 0 <= self.down_sample_ratio < 1:
            select_every = round(1 / self.down_sample_ratio)
            dataset = dataset[::select_every]

        return dataset

    # def save_rollout_to_disk(self):
    #     if not self.rollout_id or not self.step_data_list:
    #         print(f"[WARNING] {self.task_name}_{self.rollout_id}_{self.device_id}No rollout_id or step data to save.")
    #         return

    #     save_path = os.path.join(self.cache_dir, f"{self.task_name}_{self.rollout_id}.parquet")
    #     try:
    #         print(f"{self.task_name}_{self.rollout_id}_{self.device_id}[SAVE] Saving rollout to {save_path} with {len(self.step_data_list)} steps")
    #         save_base64_images_to_parquet(self.step_data_list, save_path)
    #     except Exception as e:
    #         print(f"[ERROR] {self.task_name}_{self.rollout_id}_{self.device_id}Failed to save rollout: {e}")
    def save_rollout_to_disk(self):
        if not self.rollout_id or not self.step_data_list:
            print(f"[WARNING] {self.task_name}_{self.rollout_id}_{self.device_id} No rollout_id or step data to save.")
            logger.debug(
                f"[WARNING] {self.task_name}_{self.rollout_id}_{self.device_id} No rollout_id or step data to save.")
            return
        save_path = os.path.join(self.cache_dir, f"{self.task_name}_{self.rollout_id}.json")
        try:
            print(
                f"{self.task_name}_{self.rollout_id}_{self.device_id}[SAVE] Saving rollout to {save_path} with {len(self.step_data_list)} steps")
            logger.debug(
                f"{self.task_name}_{self.rollout_id}_{self.device_id}[SAVE] Saving rollout to {save_path} with {len(self.step_data_list)} steps")
            with open(save_path, 'w') as f:
                json.dump(self.step_data_list, f, ensure_ascii=False, indent=4)

            print(f"[INFO] Successfully saved {len(self.step_data_list)} steps to {save_path}")
            logger.debug(f"[INFO] Successfully saved {len(self.step_data_list)} steps to {save_path}")

        except Exception as e:
            print(f"[ERROR] {self.task_name}_{self.rollout_id}_{self.device_id} Failed to save rollout: {e}")
            logger.debug(f"[ERROR] {self.task_name}_{self.rollout_id}_{self.device_id} Failed to save rollout: {e}")
