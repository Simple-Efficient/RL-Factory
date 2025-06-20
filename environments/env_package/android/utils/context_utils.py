import re
from PIL import Image
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import logging
import re
import json
from typing import Dict


def clean_action_content(action_content: str) -> str:
    action_content = re.sub(r"[\u200b\u200e\u200f]", "", action_content)  # 零宽字符
    action_content = action_content.strip(" \n\t×#。、，。；;")
    return action_content


def parse_llm_raw_response(response: str,
                           special_token_list: List[str] = None,
                           action_sep: str = ",",
                           max_actions: int = 3) -> Dict[str, Any]:
    pattern = r"<think>(.*?)</think>\s*<answer>(.*?)</answer>"
    match = re.search(pattern, response, re.DOTALL)
    format_correct = match is not None

    think_content = ""
    action_content = ""
    actions: List[Any] = []

    if match:
        think_content, action_content = match.group(1).strip(), match.group(2).strip()

        if special_token_list:
            for token in special_token_list:
                think_content = think_content.replace(token, "")
                action_content = action_content.replace(token, "")

        action_content = clean_action_content(action_content)

        try:
            parsed_action = json.loads(action_content)
            if isinstance(parsed_action, dict) and "action_type" in parsed_action:
                actions = [parsed_action]
            else:
                raise ValueError("Parsed object is not a valid action dict")
        except Exception as e:
            logging.warning("[parse_llm_raw_response] JSON 解析失败: %s\n原始 action_content: %s", e, action_content)
            raw_actions = [a.strip() for a in action_content.split(action_sep) if a.strip()]
            actions = raw_actions[:max_actions]
            action_content = (" " + action_sep + " ").join(actions)

    else:
        logging.warning("[parse_llm_raw_response] 没有匹配到 <think>...<answer>...> 格式：%s", response.strip()[:200])

    llm_response = f"<think>{think_content}</think><answer>{action_content}</answer>"

    return {
        "llm_raw_response": response,
        "llm_response": llm_response,
        "think_content": think_content,
        "action_content": action_content,
        "actions": actions,
        "format_correct": format_correct,
    }


# def parse_llm_raw_response(response: str,special_token_list=None,action_sep=',',max_actions=3) -> Dict:
#     """
#     assume a good format is <think>...</think><answer>...</answer>
#     returns a dict with keys:
#     - llm_raw_response: the original response
#     - llm_response: the response with <think> and <answer> tags
#     - think_content: the content inside <think> tag
#     - action_content: the content inside <answer> tag
#     - actions: a list of actions extracted from action_content
#     """

#     pattern = r'<think>(.*?)</think>\s*<answer>(.*?)</answer>'
#     match = re.search(pattern, response, re.DOTALL)
#     format_correct = match is not None

#     if not match:
#         think_content, action_content, actions = "", "", []
#     else:
#         think_content, action_content = match.group(1), match.group(2)
#         if special_token_list is not None:
#             for special_token in special_token_list: # remove all special tokens in responses to forbid confusion in training
#                 action_content = action_content.replace(special_token, "").strip()
#                 think_content = think_content.replace(special_token, "").strip()
#         actions = [action.strip() for action in action_content.split(action_sep) if action.strip()]
#         if len(actions) > max_actions:
#             actions = actions[:max_actions] #Only the first MAX_ACTIONS actions are kept in the rollout.
#             action_content = (" " + action_sep + " ").join(actions)

#     llm_response = "<think>" + think_content.strip() + "</think>" + "<answer>" + action_content.strip() + "</answer>"
#     return {
#         "llm_raw_response": response,
#         "llm_response": llm_response,
#         "think_content": think_content,
#         "action_content": action_content,
#         "actions": actions,
#         "format_correct": format_correct,
#     }

def convert_numpy_to_PIL(numpy_array: np.ndarray) -> Image.Image:
    if numpy_array.shape[-1] == 3:
        # Convert numpy array to RGB PIL Image
        return Image.fromarray(numpy_array, mode='RGB')
    else:
        raise ValueError(f"Unsupported number of channels: {numpy_array.shape[-1]}. Expected 3 (RGB).")
