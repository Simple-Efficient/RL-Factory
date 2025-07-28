import re
import json
import string
import torch
import sys
from typing import List
from PIL import Image
from .mmbase import MMEnv

class VisionEnv(MMEnv):
    def __init__(self, config, centralized_actor=None):
        super().__init__(config, centralized_actor)
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
                    for i in range(len(temp_tool_list)):
                        if temp_tool_list[i]['name'] == '<error>':
                            fail_number += 1
                    step_rew = ((len(temp_tool_list) - 2 * fail_number) / len(temp_tool_list)) * format_score
                    step_reward.append(step_rew)

        return step_reward

    @staticmethod
    def _normalize_answer(s):
        """Normalize answer by removing articles, punctuation and extra spaces."""
        s = re.sub(r"\b(a|an|the)\b", " ", s.lower())
        s = "".join(ch for ch in s if ch not in string.punctuation)
        return " ".join(s.split())

    @staticmethod
    def _extract_answer(solution_str):
        """Extract answer from solution string, removing <think> tags."""
        solution_str = re.sub(r'<think>.*?</think>', '', solution_str, flags=re.DOTALL)
        matches = re.findall(r'<answer>(.*?)</answer>', solution_str, re.DOTALL)
        return matches[-1].strip() if matches else None

    @staticmethod
    def _check_tags_valid(text, tag_pattern):
        """Check if tags are properly formatted and alternating."""
        tags = re.findall(tag_pattern, text)
        stack = []
        for tag in tags:
            if tag.startswith('</'):
                if not stack or stack[-1] != tag[2:-1]:
                    return False
                stack.pop()
            else:
                stack.append(tag[1:-1])
        return len(stack) == 0

    @staticmethod
    def _is_valid_json(text):
        """Check if text is valid JSON."""
        try:
            json.loads(text)
            return True
        except json.JSONDecodeError:
            return False

    def _compute_format_score(self, solution_str, format_score):
        """Compute format score for answer and tool_call tags."""
        answer_score = format_score if self._check_tags_valid(solution_str, r"</?answer>") else -format_score
        
        if not self._check_tags_valid(solution_str, r"</?tool_call>"):
            return answer_score - 0.5 * format_score
            
        tool_calls = re.findall(r"<tool_call>(.*?)</tool_call>", solution_str, re.DOTALL)
        if not tool_calls:
            return answer_score - format_score
            
        valid_calls = sum(1 for call in tool_calls if self._is_valid_json(call.strip()))
        total_calls = len(tool_calls)
        
        tool_score = 2 * format_score * valid_calls / total_calls - format_score
        if total_calls > 2:
            tool_score -= 0.5 * format_score
            
        return answer_score + (tool_score if total_calls > 0 else 0)

    def _compute_score_with_rules(self, data, tokenizer, if_val=False, verbose=False):
        """Compute scores for given data using exact match and format rules."""
        format_score = 0.0 if if_val else 0.1
        scores = []
        
        if verbose:
            print(f"\n=== 开始评分 {len(data)} 个样本 ===", file=sys.stderr)
        
        for i, data_item in enumerate(data):
            processed_data = self._process_data(data_item=data_item, tokenizer=tokenizer)
            ground_truth = processed_data['ground_truth']
            response_str = processed_data['response_str']
            
            # 提取答案
            answer = self._extract_answer(response_str)
            format_score_val = self._compute_format_score(response_str, format_score)
            
            # 计算分数
            if answer is None:
                score = -format_score + 0.5 * format_score_val
                result = "无答案"
            else:
                normalized_answer = self._normalize_answer(answer)
                is_correct = any(self._normalize_answer(truth) == normalized_answer 
                               for truth in ground_truth)
                if is_correct:
                    score = 1.0 + 0.5 * format_score_val
                    result = "正确"
                else:
                    score = format_score_val
                    result = "错误"
            
            if verbose:
                print(f"\n#{i+1}: {result} | 分数: {score:.3f}", file=sys.stderr)
                print(f"问题: {processed_data.get('prompt_str', '')[:200]}...", file=sys.stderr)
                print(f"答案: {answer or '未提取到'}", file=sys.stderr)
                print(f"标准: {ground_truth}", file=sys.stderr)
            
            scores.append([score])
        
        if verbose:
            avg_score = sum(s[0] for s in scores) / len(scores)
            print(f"\n=== 平均分: {avg_score:.3f} ===", file=sys.stderr)

        return scores