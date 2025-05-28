import re
import string
import random
from .base import Env


class RewardRolloutEnv(Env):
    def __init__(self, config):
        super().__init__(config)
        self.use_verify_tool = False

    def _get_single_prompt_str(self, data_source, solution_str, ground_truth, extra_info, reward_tokenizer):
        system_content = """
You are the assistant to an intelligent search helper, responsible for evaluating the quality of search results. These evaluations will be used to adjust search queries and to better respond to users using the relevant information obtained from searches.
Below are the outputs from a fine-tuned LLM and the corresponding ground truth labels. The model outputs are enclosed in <output></output> tags, while the ground truth labels are enclosed in <gt></gt> tags. Please assess the quality of the model's output, assign a score, and present the score within <score></score> tags.
"""
        user_content = """
Model outputs: <output>{}</output> 
Ground Truth: <gt>{}</gt>
"""
        # extract only the answer for the reward judge model
        solution_str = self._extract_answer(solution_str)
        # If you need use the reasoning model to generate the reward, you need to modify the following code
        result = reward_tokenizer.apply_chat_template([
                {'role': 'system', 'content': system_content},
                {'role': 'user', 'content': user_content.format(solution_str, ground_truth['target'])}
                ], add_generation_prompt=True, tokenize=False)
        
        return result

    def _compute_single_score_with_reward_rollout_wg(self, data_source, solution_str, ground_truth, extra_info):
        pattern = r'<score>([^<>]*)</score>'
        matches = list(re.finditer(pattern, solution_str, re.DOTALL))
        print('solution_str: ', solution_str)
        if matches:
            last_match = matches[-1]
            try:
                result = float(last_match.group(1))
            except:
                result = 0.0
            return result
        else:
            return 0.0
    
    def _extract_answer(self, input_str) -> str:
        start_tag = "<answer>" # for default answer tags
        end_tag = "</answer>"
        extracted_strings = ""
        start_index = input_str.find(start_tag)
        while start_index != -1:
            end_index = input_str.find(end_tag, start_index)
            if end_index != -1:
                extracted_str = input_str[start_index + len(start_tag):end_index]
                extracted_strings += extracted_str + "\n"
                start_index = input_str.find(start_tag, end_index)
            else:
                break
        return extracted_strings
