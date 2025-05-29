import re
import json
import string
import random
import torch
from .base import Env
from verl import DataProto

class SearchEnv(Env):
    def __init__(self, config):
        super().__init__(config)
        self.use_verify_tool = False

    def get_step_reward(self, responses, format_score=0.1):
        step_reward = []
    
        for response in responses:
            temp_action, temp_tool_list = self.tool_manager.parse_response(response_content=response)
            if temp_action == 'answer':
                step_reward.append(torch.nan)
            else:
                if temp_tool_list[0]['name'] == '<empty>':
                    step_reward.append(-0.5 * format_score)
                else:
                    fail_number = 0
                    for i in range(len(temp_tool_list )):
                        if temp_tool_list[i]['name'] == '<error>':
                            fail_number += 1
                    step_rew = ((len(temp_tool_list) - 2 *fail_number) / len(temp_tool_list)) * format_score
                    step_reward.append(step_rew)
       

        return step_reward

    # NOTE: Add your reward calculation rules here!
    def _compute_score_with_rules(self, data, tokenizer, if_val=False):
        def normalize_answer(s):
            def remove_articles(text):
                return re.sub(r"\b(a|an|the)\b", " ", text)

            def white_space_fix(text):
                return " ".join(text.split())

            def remove_punc(text):
                exclude = set(string.punctuation)
                return "".join(ch for ch in text if ch not in exclude)

            def lower(text):
                return text.lower()

            return white_space_fix(remove_articles(remove_punc(lower(s))))

        def em_check(prediction, golden_answers):
            if isinstance(golden_answers, str):
                golden_answers = [golden_answers]
            normalized_prediction = normalize_answer(prediction)
            score = 0
            for golden_answer in golden_answers:
                golden_answer = normalize_answer(golden_answer)
                if golden_answer == normalized_prediction:
                    score = 1
                    break
            return score

        def extract_solution(solution_str):
            """Extract the equation from the solution string."""
            # Remove everything before the first "Assistant:"
            # if "Assistant:" in solution_str:
            #     solution_str = solution_str.split("Assistant:", 1)[1]
            # elif "<|im_start|>assistant" in solution_str:
            #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
            # else:
            #     return None
            # solution_str = solution_str.split('\n')[-1]
            think_pattern = r'<think>.*?</think>'
            solution_str = re.sub(think_pattern, '', solution_str, flags=re.DOTALL)

            answer_pattern = r'<answer>(.*?)</answer>'
            match = re.finditer(answer_pattern, solution_str, re.DOTALL)
            matches = list(match)
            
            # If there are 0 or exactly 1 matches, return None
            # if len(matches) <= 1:
            if len(matches) <= 0:
                return None
            
            # If there are 2 or more matches, return the last one
            return matches[-1].group(1).strip()

        def compute_score_em(solution_str, ground_truth, method='strict', format_score=0.0, score=1.):
            """The scoring function for exact match (EM).

            Args:
                solution_str: the solution text
                ground_truth: the ground truth
                method: the method to extract the solution, choices are 'strict' and 'flexible'
                format_score: the score for the format
                score: the score for the correct answer
            """
            answer = extract_solution(solution_str=solution_str)
            do_print = random.randint(1, 64) == 1
            
            if do_print:
                print(f"--------------------------------")
                print(f"Golden answers: {ground_truth['target']}")
                print(f"Extracted answer: {answer}")
                print(f"Solution string: {solution_str}")
            
            answer_format_score = format_score if check_alternate_tags(solution_str, r"</?answer>") else (-1 * format_score)
            num_score=0
            if check_alternate_tags(solution_str, r"</?tool_call>"):
                tool_call_format_score = format_score
                pattern = r"<tool_call>(.*?)</tool_call>"
                matches = re.findall(pattern, solution_str, re.DOTALL)
                if len(matches) == 0:
                    tool_call_format_score = -1 * format_score
                else:
                    success_num, fail_num = 0, 0
                    for idx, content in enumerate(matches):
                        content_stripped = content.strip()
                        try:
                            parsed = json.loads(content_stripped)
                            success_num += 1
                        except json.JSONDecodeError:
                            fail_num += 1
                    
                    tool_call_format_score = 2 * format_score * success_num / (success_num + fail_num) - format_score
                    if success_num + fail_num > 2:
                        tool_call_format_score -= 0.5 * format_score
                        num_score = -format_score
            else:
                tool_call_format_score = -0.5 * format_score
                
            #total_format_score = tool_call_format_score + answer_format_score
            total_format_score = answer_format_score+num_score

            if answer is None:
                return -1 * format_score + 0.5 * total_format_score
            else:
                if em_check(answer, ground_truth['target']):
                    return score + 0.5 * total_format_score
                else:
                    return total_format_score
        
        def check_alternate_tags(text, tag_pattern):
            # 匹配所有<tool_call>和</tool_call>标签
            tags = re.findall(tag_pattern, text)
            
            stack = []
            for tag in tags:
                if tag == "<tool_call>":
                    if stack:
                        # 发现有嵌套，说明不是严格交替
                        return False
                    stack.append(tag)
                elif tag == "</tool_call>":
                    if not stack:
                        # 没有对应的<tool_call>，说明标签不配对
                        return False
                    stack.pop()
            
            # 最终栈必须为空，才是严格交替
            return len(stack) == 0

        format_score = 0.0 if if_val else 0.1
        scores = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            
            # process the data_item to the token and decode them
            processed_data = self._process_data(data_item=data_item, tokenizer=tokenizer)
            ground_truth, response_str = processed_data['ground_truth'], processed_data['response_str']
            
            # reserved for compatibility
            prompt_str, data_source, extra_info = processed_data['prompt_str'], processed_data['data_source'], processed_data['extra_info']

            score = compute_score_em(response_str, ground_truth, format_score=format_score)
            scores.append([score])

        return scores
