import re
import json
import string
import random
from ....envs.base import Env
import torch
from ..reward_score.evaluation.custom_eval import evaluate_plan
from typing import Optional, Dict

# travel planning agent support
class TravelplannerEnv(Env):
    """
    Travel planning environment for reinforcement learning-based travel itinerary generation.
    
    This environment extends the base Env class to provide specialized functionality
    for travel planning tasks. It handles the evaluation of travel plans generated
    by language models, including parsing structured outputs and calculating rewards
    based on plan quality and constraint satisfaction.
    
    Key features:
    - Processes travel planning responses with <answer> tags
    - Evaluates plan quality using custom evaluation logic
    - Calculates rewards based on format compliance and plan correctness
    - Supports both training and validation modes
    - Integrates with the VERL framework for RL training
    
    The environment expects responses in a specific format with JSON-structured
    travel plans enclosed in <answer> tags, and evaluates these plans against
    predefined constraints and requirements.
    """
    def __init__(self, config, centralized_actor=None):
        super().__init__(config, centralized_actor)
        self.use_verify_tool = False

    # NOTE: Add your reward calculation rules here!
    def _compute_score_with_rules(self, data, tokenizer, if_val=False):
        """
        
        Calculate reward scores for each batch
        
        Args:
            data: Input data
            tokenizer: Tokenizer
            if_val: Whether it is validation mode
        
        Returns:
            torch.Tensor: Reward score tensor with shape [batch_size, 1]
        """

        def calculate_final_answer_reward(
            decoded_text: str, 
            hard_constraint_data: Optional[Dict] = None,
            answer_format_score: float = 1.0
        ) -> float:
            """
            从解码后的文本中提取 <answer>...</answer> 部分，并计算奖励。
            
            参数：
                decoded_text: 解码后的文本
                hard_constraint_data: 预设定的查询数据（可选）
            
            返回：
                计算得到的奖励分数（float）
            """
            # total_reward = answer_format_reward + final_answer_reward
            total_reward = 0.0  
            answer_format_reward = 0.0
            final_answer_reward = 0.0
            pattern = r'<answer>(.*?)</answer>'
            final_answers = re.findall(pattern, decoded_text, re.DOTALL)
            
            # 检查是否至少有两个 <answer> 块
            if len(final_answers) < 2:
                return total_reward, answer_format_reward, final_answer_reward
            
            # 提取第二个 <answer> 内容并清理
            second_answer = final_answers[-1].strip()
            second_answer = re.sub(r'```(json)?|```', '', second_answer)  # 移除所有代码块标记
            
            try:
                test_data = json.loads(second_answer)  # 解析 JSON
            except json.JSONDecodeError as e:
                print(f"DEBUG: JSON 解析失败 - {e}")
                return total_reward, answer_format_reward, final_answer_reward
            except Exception as e:
                print(f"DEBUG: 未知解析错误 - {e}")
                return total_reward, answer_format_reward, final_answer_reward

            try:
                final_answer_reward = evaluate_plan(hard_constraint_data, test_data)
                answer_format_reward = 1.0
                total_reward = total_reward + answer_format_reward + final_answer_reward
            except Exception as e:
                print(f"DEBUG: evaluate_plan 执行失败 - {e}")
                return total_reward, answer_format_reward, final_answer_reward
            
            return total_reward, answer_format_reward, final_answer_reward

        batch_size = len(data)
        meta_info = data.meta_info
        query_datas = data.non_tensor_batch['reward_model']
        
        answer_format_rewards = []
        final_answer_rewards = []
        all_scores = []
        
        for i in range(batch_size):
            data_item = data[i]
            prompts = data_item.batch['prompts'].unsqueeze(0)
            responses = data_item.batch['responses'].unsqueeze(0)
            attention_mask = data_item.batch['attention_mask'].unsqueeze(0)
            
            # 计算有效长度
            prompt_length = prompts.size(1)
            valid_prompt_length = int(attention_mask[0, :prompt_length].sum().item())
            valid_response_length = int(attention_mask[0, prompt_length:].sum().item())
            
            # 获取有效token
            valid_prompt_ids = prompts[:, -valid_prompt_length:]
            valid_response_ids = responses[:, :valid_response_length]
            sequences = torch.cat((valid_prompt_ids, valid_response_ids), dim=1)
            sequences_str = tokenizer.decode(sequences[0].tolist())
            
            # 计算奖励
            query_data = query_datas[i]['ground_truth']
            final_answer_reward, answer_format_score, final_answer_reward = calculate_final_answer_reward(sequences_str, query_data)
            
            answer_format_rewards.append(answer_format_score)
            final_answer_rewards.append(final_answer_reward)
            all_scores.append(final_answer_reward)  # 直接使用 final_answer_reward 作为总分
        
        # 更新meta信息
        meta_info.update({
            'reward/answer_format_rewards': answer_format_rewards,
            'reward/final_answer_rewards': final_answer_rewards,
            'reward/all_scores': all_scores
        })
        data.meta_info.update(meta_info)
        
        # 返回 [batch_size, 1] 的张量
        return torch.tensor(all_scores, dtype=torch.float32).view(batch_size, 1)