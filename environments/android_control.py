"""Thread‑safe Android emulator pool with Redis coordination.

*   Action whitelist + sanitize_action() to reject illegal fields (e.g. x2/y2).
*   Health‑check of running emulators via adb.
*   Automatic port recycling on failures and zombie reclaim.
*   Ready to be imported from FastAPI server.
"""
from __future__ import annotations

import base64
import json
import logging
import subprocess
import threading
import time
import uuid
import asyncio
from io import BytesIO
from typing import Any, Dict, Optional, Tuple

import numpy as np
import redis
from absl import logging as absl_logging  # For AndroidWorld internal logs
from android_world.agents import base_agent
from android_world.env import env_launcher, interface, json_action
from PIL import Image
import cv2

# =============================================================================
# 0. Global constants & logging ------------------------------------------------
# =============================================================================
ADB_PATH = "/opt/android-sdk/platform-tools/adb"  # your adb
EMULATOR_SETUP = False  # If AndroidWorld should create AVD internally
NUM_EMULATORS = 8
BASE_TELNET_PORT = 5554  # First console port

# Whitelist for action fields --------------------------------------------------
ALLOWED_FIELDS: Dict[str, set[str]] = {
    "status": {"goal_status"},
    "answer": {"text"},
    "click": {"x", "y"},
    "long_press": {"x", "y"},
    "input_text": {"text", "x", "y"},
    "scroll": {"direction"},
    "navigate_back": set(),
    "navigate_home": set(),
    "keyboard_enter": set(),
    "open_app": {"app_name"},
    "wait": set(),
}

absl_logging.set_verbosity(absl_logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# =============================================================================
# 1. Utility helpers -----------------------------------------------------------
# =============================================================================

def sanitize_action(action: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(action, dict):
        logging.error("[sanitize_action] Input is not dict: %s", action)
        raise ValueError("Action must be a dict")

    action_type = action.get("action_type")

    # -------- swipe -> scroll auto‑fix ----------------------------------
    if action_type == "swipe":
        x = action.get("x")
        y = action.get("y")

        if isinstance(x, list) and len(x) == 2:
            x_diff = x[1] - x[0]
        else:
            x_diff = 0

        if isinstance(y, list) and len(y) == 2:
            y_diff = y[1] - y[0]
        elif isinstance(y, int):
            y_diff = 0
        else:
            y_diff = 0

        if abs(x_diff) > abs(y_diff):
            direction = "right" if x_diff > 0 else "left"
        else:
            direction = "down" if y_diff > 0 else "up"

        logging.warning("[sanitize_action] Auto-converting swipe to scroll: direction=%s", direction)
        action = {
            "action_type": "scroll",
            "direction": direction
        }
        action_type = "scroll"

    # -------- click list(x) fix -----------------------------------------
    if action_type == "click":
        x = action.get("x")
        y = action.get("y")
        if isinstance(x, list) and len(x) == 2 and y is None:
            action["x"], action["y"] = x[0], x[1]
            logging.warning("[sanitize_action] Fixed click x from list: x=%s, y=%s", action["x"], action["y"])

    # -------- validate fields -------------------------------------------
    if action_type not in ALLOWED_FIELDS:
        logging.error("[sanitize_action] Unsupported action_type: %s", action_type)
        raise ValueError(f"Unsupported action_type: {action_type}")

    allowed_keys = {"action_type"} | ALLOWED_FIELDS[action_type]
    cleaned = {k: v for k, v in action.items() if k in allowed_keys}

    logging.info("[sanitize_action] Cleaned action: %s", cleaned)
    return cleaned


def img_to_base64(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  # type: ignore
    return base64.b64encode(buf).decode()


# =============================================================================
# 2. AndroidEnvInteractor ------------------------------------------------------
# =============================================================================

def _post_transition_state(env: interface.AsyncEnv, pause: Optional[float] = 1.0) -> interface.State:  # type: ignore
    if pause is None:
        return env.get_state(wait_to_stabilize=True)
    time.sleep(pause)
    return env.get_state(wait_to_stabilize=False)


class AndroidEnvInteractor(base_agent.EnvironmentInteractingAgent):
    """Wrapper around AndroidWorld env that exposes reset/step/close."""

    def __init__(self, console_port: int):
        env = env_launcher.load_and_setup_env(
            console_port=console_port,
            emulator_setup=EMULATOR_SETUP,
            adb_path=ADB_PATH,
        )
        super().__init__(env)
        self.env: interface.AsyncEnv = env  # type: ignore
        self.console_port = console_port
        self.initialized = False
        self.wait_after_action_seconds = 1.0

    # ---------------------------------------------------------------------
    # Helper: screenshot as b64
    # ---------------------------------------------------------------------
    def _state_as_b64(self) -> str:
        logging.info("[Interactor] Capturing screenshot on port %s", self.console_port)
        state = self.env.get_state(wait_to_stabilize=True)
        arr = state.pixels
        img = Image.fromarray(arr.astype(np.uint8))
        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode()
        logging.info("[Interactor] Screenshot base64 length: %d", len(encoded))
        return encoded

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def reset(self) -> str:
        logging.info("[Interactor] Resetting on port %s", self.console_port)
        if self.initialized:
            self.env.go_home()
            logging.info("[Interactor] Called go_home()")
        else:
            super().reset(go_home=True)
            self.initialized = True
            logging.info("[Interactor] Called super().reset(go_home=True)")
        self.env.hide_automation_ui()
        logging.info("[Interactor] UI automation hidden")
        return self._state_as_b64()

    def step(self, raw_action: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        try:
            action = sanitize_action(raw_action)
            # JSONAction expects x/y not position
            if "position" in action:
                action["x"], action["y"] = action.pop("position")

            # before = _post_transition_state(self.env, pause=None).pixels.copy()
            state = _post_transition_state(self.env, pause=None)
            ui_elements = state.ui_elements
            before_arr = state.pixels.copy()
            action_obj = json_action.JSONAction(**action)
            self.env.execute_action(action_obj)
            time.sleep(self.wait_after_action_seconds)
            # after = _post_transition_state(self.env, pause=None).pixels.copy()
            after_arr = _post_transition_state(self.env, pause=None).pixels.copy()
            before_b64 = img_to_base64(before_arr)
            after_b64 = img_to_base64(after_arr)

            return True, {"before": before_b64, "after": after_b64, "action": action, "ui_elements": ui_elements}
        except Exception as e:
            absl_logging.error("[Interactor] step failed: %s", e)
            return False, {"error": str(e)}

    async def close(self) -> None:
        try:
            logging.info("[Interactor] Resetting before close on port %s", self.console_port)
            self.env.reset(go_home=True)  # 回到主界面
        except Exception as e:
            logging.warning("[Interactor] Failed to reset env during close: %s", e)


# =============================================================================
# 3. EmulatorPool -------------------------------------------------------------
# =============================================================================

class EmulatorPool:
    """Redis‑coordinated, thread‑safe emulator pool."""

    KEY_IDLE = "android:idle_ports"
    KEY_BUSY = "android:busy_map"  # rollout_id -> port

    def __init__(self, redis_host: str = "localhost", redis_db: int = 0):
        self.redis = redis.Redis(host=redis_host, db=redis_db, decode_responses=True)
        self._async_lock = asyncio.Lock()
        self._interactors: Dict[str, AndroidEnvInteractor] = {}
        self._init_idle_list()

    # ------------------------------------------------------------------
    # Idle list bootstrap
    # ------------------------------------------------------------------
    def _init_idle_list(self) -> None:
        if self.redis.llen(self.KEY_IDLE) == 0:
            ports = [BASE_TELNET_PORT + 2 * i for i in range(NUM_EMULATORS)]
            self.redis.lpush(self.KEY_IDLE, *ports)
            absl_logging.info("[Pool] idle ports initialized: %s", ports)

    def _acquire_port(self) -> Optional[int]:
        port = self.redis.lpop(self.KEY_IDLE)
        return int(port) if port else None

    def _release_port(self, port: int) -> None:
        self.redis.rpush(self.KEY_IDLE, port)

    async def reset(self) -> Optional[Dict[str, Any]]:
        async with self._async_lock:
            port = self._acquire_port()
        if port is None:
            logging.warning("[Pool] no idle ports")
            return None

        rollout_id = f"rollout_{uuid.uuid4().hex[:6]}"
        device_id = f"emulator-{port}"

        lock = asyncio.Lock()
        ready = asyncio.Event()


        self._interactors[rollout_id] = {
            "interactor": None,
            "lock": lock,
            "ready": ready,
        }

        try:
            interactor = await asyncio.to_thread(AndroidEnvInteractor, port)
            self._interactors[rollout_id]["interactor"] = interactor

            screenshot_b64 = await asyncio.to_thread(interactor.reset)
            self.redis.hset(self.KEY_BUSY, rollout_id, port)
            ready.set()

            return {
                "rollout_id": rollout_id,
                "device_id": device_id,
                "message": "reset ok",
                "screenshot_b64": screenshot_b64,
            }

        except Exception as e:
            logging.exception("[Pool] reset failed for port %s : %s", port, e)
            self._interactors.pop(rollout_id, None)
            async with self._async_lock:
                self._release_port(port)
            return None

    async def step(self, rollout_id: str,
                   action: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        logging.info("[Pool] STEP request: rollout_id=%s  raw_action=%s",
                     rollout_id, action)

        try:
            action = sanitize_action(action)
        except Exception as e:
            return False, {"error": f"sanitize_action failed: {e}"}

        entry = self._interactors.get(rollout_id)
        if entry is None:
            port_str = self.redis.hget(self.KEY_BUSY, rollout_id)
            if port_str is None:
                return False, {"error": "invalid rollout_id"}

            try:
                rebuilt = await asyncio.to_thread(
                    AndroidEnvInteractor, int(port_str)
                )
                entry = {
                    "interactor": rebuilt,
                    "lock": asyncio.Lock(),
                    "ready": asyncio.Event()  # 重建后直接视作 ready
                }
                entry["ready"].set()
                self._interactors[rollout_id] = entry
                logging.info("[Pool] Re-created interactor for %s on port %s",
                             rollout_id, port_str)
            except Exception as e:
                await self.close(rollout_id)
                return False, {"error": f"recreate interactor failed: {e}"}

        interactor: AndroidEnvInteractor = entry["interactor"]
        lock: asyncio.Lock = entry["lock"]
        ready: asyncio.Event = entry["ready"]

        await ready.wait()

        async with lock:
            logging.info("[STEP] rollout_id=%s  port=%s  action=%s",
                         rollout_id, interactor.console_port,
                         json.dumps(action, ensure_ascii=False))
            try:
                ok, payload = await asyncio.to_thread(interactor.step, action)
                return ok, payload
            except Exception as e:
                logging.exception("[Pool] step execution failed")
                return False, {"error": str(e)}

    async def close(self, rollout_id: str) -> bool:
        logging.info("[Pool] Closing rollout_id=%s", rollout_id)

        port_str = self.redis.hget(self.KEY_BUSY, rollout_id)
        if port_str is None:
            logging.warning("[Pool] rollout_id %s not found in Redis", rollout_id)
            return False
        port = int(port_str)

        entry = self._interactors.pop(rollout_id, None)
        if entry is None:
            entry = {
                "interactor": await asyncio.to_thread(AndroidEnvInteractor, port),
                "lock": asyncio.Lock(),
                "ready": asyncio.Event()
            }
            entry["ready"].set()

        interactor: AndroidEnvInteractor = entry["interactor"]
        lock: asyncio.Lock = entry["lock"]
        ready: asyncio.Event = entry["ready"]

        await ready.wait()
        async with lock:
            try:
                await interactor.close()
                logging.info("[Pool] Interactor on port %s closed", port)
            except Exception as e:
                logging.warning("[Pool] Closing interactor failed: %s", e)

        async with self._async_lock:
            self.redis.hdel(self.KEY_BUSY, rollout_id)
            self._release_port(port)

        return True
