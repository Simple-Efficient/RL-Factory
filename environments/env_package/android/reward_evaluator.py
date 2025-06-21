import json
import logging

logger = logging.getLogger(__name__)

class RewardEvaluator:
    """
    RewardEvaluator evaluates user actions on Android GUI environments
    using large language models (LLMs). It supports two evaluation modes:

    - "process": Process Reward Model (PRM), checks whether the action made progress.
    - "outcome": Outcome Reward Model (ORM), checks whether the goal has been completed.

    The class works by comparing two screenshots (before and after an action),
    along with a user instruction (goal), and then querying an LLM API
    (via `call_modal`) to obtain a structured score and reasoning.

    This module can be used to build your own PRM/ORM system powered by LLMs.
    """

    def __init__(self, call_modal):
        """
        Args:
            call_modal (Callable): Function to call a multi-modal LLM API.
                It should accept ([img1_base64, img2_base64], prompt) and return a string response.
        """
        self.call_modal = call_modal

    def evaluate(self, img1_b64: str, img2_b64: str, instruction: str, mode: str = "process"):
        """
        Evaluate the action between two screenshots with respect to a user instruction.

        Args:
            img1_b64 (str): Base64 image before the action.
            img2_b64 (str): Base64 image after the action.
            instruction (str): User's task goal (e.g., "Open Gmail and compose a message").
            mode (str): Either "process" or "outcome".

        Returns:
            Tuple[float, float]: A tuple (success_flag, score), where:
                - success_flag = 1.0 if score == 5, else 0.0
                - score is the numeric result from the LLM (0 or 5)
        """
        prompt = self._build_prompt(instruction, mode)

        try:
            response = self.call_modal([img1_b64, img2_b64], prompt)
            if not response:
                logger.warning("[RewardEvaluator] Empty response.")
                return 0.0, 0.0

            # Remove markdown formatting if present
            response = response.strip().replace("```json", "").replace("```", "")
            logger.debug(f"[RewardEvaluator] Cleaned model response: {response}")

            parsed = json.loads(response)
            score = float(parsed.get("score", 0))
            success = 1.0 if score == 5 else 0.0
            return success, score

        except json.JSONDecodeError as e:
            logger.error(f"[RewardEvaluator] JSON decode error: {e}. Raw response: {response}")
            return 0.0, 0.0
        except Exception as e:
            logger.error(f"[RewardEvaluator] Exception in model call: {e}")
            return 0.0, 0.0

    def _build_prompt(self, instruction: str, mode: str) -> str:
        """
        Construct a textual prompt for the LLM based on the task mode.

        - For "outcome" mode, ask whether the goal is fully completed.
        - For "process" mode, ask whether the action makes meaningful progress.

        Args:
            instruction (str): The user goal.
            mode (str): Either "process" or "outcome".

        Returns:
            str: A formatted prompt to send to the LLM.
        """
        if mode == "outcome":
            return (
                "You are an intelligent assistant tasked with evaluating whether a user operation "
                "on a mobile device has successfully **completed the intended goal**.\n\n"
                "You will receive two screenshots:\n"
                "- <image1>: The screen before the action.\n"
                "- <image2>: The screen after the action.\n\n"
                f"**User goal**: {instruction}\n\n"
                "Please compare the screenshots and determine if the goal has been **fully achieved**.\n\n"
                "Scoring criteria:\n"
                "- If the goal is fully achieved: `score = 5`\n"
                "- If the goal is not achieved: `score = 0`\n\n"
                "Respond in the following JSON format:\n"
                "```json\n"
                '{\n'
                '  "score": 0 or 5,\n'
                '  "reason": "Brief explanation of why the task was or was not completed."\n'
                '}\n'
                "```"
            )
        else:  # mode == "process"
            return (
                "You are an intelligent assistant tasked with evaluating whether a user operation "
                "on a mobile device has **helped make progress** toward the intended goal.\n\n"
                "You will receive two screenshots:\n"
                "- <image1>: The screen before the action.\n"
                "- <image2>: The screen after the action.\n\n"
                f"**User goal**: {instruction}\n\n"
                "Please compare the screenshots and determine if the action has made meaningful progress "
                "towards completing the goal.\n\n"
                "Scoring criteria:\n"
                "- If the action made meaningful progress: `score = 5`\n"
                "- If the action did not help or was unrelated: `score = 0`\n\n"
                "Respond in the following JSON format:\n"
                "```json\n"
                '{\n'
                '  "score": 0 or 5,\n'
                '  "reason": "Brief explanation of why the action helped or not."\n'
                '}\n'
                "```"
            )
