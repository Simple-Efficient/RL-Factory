import re
import json
import string
import torch
import sys

from typing import List
from PIL import Image
from .mmbase import MMEnv

class VisionEnv(MMEnv): # used for mm tool use demo
    def __init__(self, config, centralized_actor=None):
        """Initialize vision environment with configuration"""
        super().__init__(config, centralized_actor)
        self.use_verify_tool = False
        


    def get_step_reward(self, responses, format_score=0.1):
        """Calculate step reward based on tool usage validity"""
        step_reward = []
    
        for response in responses:
            temp_action, temp_tool_list = self.tool_manager.parse_response(response_content=response)
            if temp_action == 'answer':
                step_reward.append(torch.nan)  # No reward for direct answer
            else:
                if temp_tool_list[0]['name'] == '<empty>':
                    step_reward.append(-0.5 * format_score)  # Penalty for empty tool
                else:
                    fail_number = 0
                    for i in range(len(temp_tool_list)):
                        if temp_tool_list[i]['name'] == '<error>':
                            fail_number += 1
                    # Calculate reward based on success/failure ratio
                    step_rew = ((len(temp_tool_list) - 2 * fail_number) / len(temp_tool_list)) * format_score
                    step_reward.append(step_rew)
        return step_reward

    @staticmethod
    def _normalize_answer(s):
        """Normalize answer string by removing articles/punctuation/extra spaces"""
        s = re.sub(r"\b(a|an|the)\b", " ", s.lower())
        s = "".join(ch for ch in s if ch not in string.punctuation)
        return " ".join(s.split())

    @staticmethod
    def _extract_answer(solution_str):
        """Extract answer content from solution string"""
        solution_str = re.sub(r'<think>.*?</think>', '', solution_str, flags=re.DOTALL)
        matches = re.findall(r'<answer>(.*?)</answer>', solution_str, re.DOTALL)
        return matches[-1].strip() if matches else None  # Get last answer

    @staticmethod
    def _check_tags_valid(text, tag_pattern):
        """Validate proper tag nesting and formatting"""
        tags = re.findall(tag_pattern, text)
        stack = []
        for tag in tags:
            if tag.startswith('</'):
                if not stack or stack[-1] != tag[2:-1]:
                    return False
                stack.pop()
            else:
                stack.append(tag[1:-1])
        return len(stack) == 0  # Check balanced tags

    @staticmethod
    def _is_valid_json(text):
        """Validate JSON format correctness"""
        try:
            json.loads(text)
            return True
        except json.JSONDecodeError:
            return False

    def _compute_format_score(self, solution_str, format_score):
        """Calculate formatting score for answer/tool_call tags"""
        # Check answer tags
        answer_score = format_score if self._check_tags_valid(solution_str, r"</?answer>") else -format_score
        
        # Check tool_call tags
        if not self._check_tags_valid(solution_str, r"</?tool_call>"):
            return answer_score - 0.5 * format_score
            
        tool_calls = re.findall(r"<tool_call>(.*?)</tool_call>", solution_str, re.DOTALL)
        if not tool_calls:
            return answer_score - format_score  # Penalty for no tool calls
            
        # Count valid JSON tool calls
        valid_calls = sum(1 for call in tool_calls if self._is_valid_json(call.strip()))
        total_calls = len(tool_calls)
        
        # Calculate tool score with penalty for excessive calls
        tool_score = 2 * format_score * valid_calls / total_calls - format_score
        if total_calls > 2:
            tool_score -= 0.5 * format_score
            
        return answer_score + (tool_score if total_calls > 0 else 0)

    def _compute_score_with_rules(self, data, tokenizer, if_val=False, verbose=False):
        """Compute final scores using exact match and format rules"""
        format_score = 0.0 if if_val else 0.1  # Disable format score in validation mode
        scores = []
        
        if verbose:
            print(f"\n=== start evaluating {len(data)} samples ===", file=sys.stderr)
        
        for i, data_item in enumerate(data):
            processed_data = self._process_data(data_item=data_item, tokenizer=tokenizer)
            ground_truth = processed_data['ground_truth']
            response_str = processed_data['response_str']
            
            # Extract answer and calculate format score
            answer = self._extract_answer(response_str)
            format_score_val = self._compute_format_score(response_str, format_score)
            
            # Determine final score
            if answer is None:
                score = -format_score + 0.5 * format_score_val
                result = "No Ans"
            else:
                normalized_answer = self._normalize_answer(answer)
                is_correct = any(self._normalize_answer(truth) == normalized_answer 
                               for truth in ground_truth)
                if is_correct:
                    score = 1.0 + 0.5 * format_score_val  # Base score for correct answer
                    result = "Correct"
                else:
                    score = format_score_val  # Only format score
                    result = "Wrong"
            
            if verbose:
                print(f"\n#{i+1}: {result} | score: {score:.3f}", file=sys.stderr)
                print(f"Q: {processed_data.get('prompt_str', '')[:200]}...", file=sys.stderr)
                print(f"A: {answer or 'extraction failure'}", file=sys.stderr)
                print(f"std: {ground_truth}", file=sys.stderr)
            
            scores.append([score])
        
        if verbose:
            avg_score = sum(s[0] for s in scores) / len(scores)
            print(f"\n=== Avg Score: {avg_score:.3f} ===", file=sys.stderr)

        return scores
