import torch
import re
import numpy as np
import json
import ast
# 从hard_constraints模块导入硬约束评估函数
from verl.utils.reward_score.evaluation.custom_eval import evaluate_plan
from typing import Dict, Optional
from verl import DataProto


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

def compute_score_fn(data: DataProto, 
                     tool_reward: float, 
                     memory_reward: float, 
                     answer_format_reward: float, 
                     tokenizer=None):
    """
    计算批次数据的奖励分数，包含工具使用、记忆使用和答案质量的综合评估。

    参数:
        data (DataProto): 包含批次数据的对象，需包含以下内容:
            - batch: 包含 'prompts'、'responses'、'attention_mask' 的字典
            - meta_info: 包含 'tool_action_stats' 和 'memory_action_stats' 的统计信息
            - non_tensor_batch: 包含 'reward_model' 的查询数据
        tool_reward (float): 使用工具的奖励分数
        memory_reward (float): 使用记忆的奖励分数
        answer_format_reward (float): 答案格式的奖励分数
        final_answer_reward (float): 最终答案的奖励分数
        tokenizer: 用于解码token序列的分词器对象

    返回:
        torch.Tensor: 形状为 (batch_size, response_length) 的奖励张量，
                     仅在每个序列的最后一个有效token位置设置奖励值

    注意:
        1. 奖励计算逻辑:
           - 如果使用了工具: reward = tool_reward + final_answer_reward
           - 如果使用了记忆: reward = memory_reward + final_answer_reward
           - 其他情况: reward = final_answer_reward
        
        2. 函数会更新 meta_info 中的以下统计信息:
           - reward/tool_rewards
           - reward/memory_rewards
           - reward/answer_format_rewards
           - reward/final_answer_rewards
    """
    batch_size = len(data)
    all_scores = []
    # breakpoint()
    meta_info = data.meta_info

    # 创建形状为 [batch_size, response_length] 的零张量
    response_length = data.batch['responses'].size(1)   
    reward_tensor = torch.zeros((batch_size, response_length), dtype=torch.float32)
    
    print(f"批次大小 batch_size: {batch_size}")
    print(f"回复长度 response_length: {response_length}")
    
    # 从 meta_info 中获取统计数据
    tool_stats = data.meta_info.get("tool_action_stats", [0] * batch_size)
    memory_stats = data.meta_info.get("memory_action_stats", [0] * batch_size)
    
    # breakpoint()
    # TODO: 从 meta_info 中获取预设定的查询数据

    query_datas = data.non_tensor_batch['reward_model']
    
    tool_rewards = []
    memory_rewards = []
    answer_format_rewards = []
    final_answer_rewards = []
    
    print(f"tool_stats: {tool_stats}")
    print(f"memory_stats: {memory_stats}")
    
    # if 1 in tool_stats:
    for i in range(batch_size):
        data_item = data[i]
        # 获取每个批次的张量
        prompts = data_item.batch['prompts'].unsqueeze(0)  # 添加批次维度
        responses = data_item.batch['responses'].unsqueeze(0)  # 添加批次维度
        attention_mask = data_item.batch['attention_mask'].unsqueeze(0)  # 添加批次维度
        
        # print(f"\n批次 {i} 的数据形状:")
        # print(f"prompts shape: {prompts.shape}")
        # print(f"responses shape: {responses.shape}")
        # print(f"attention_mask shape: {attention_mask.shape}")
        
        prompt_length = prompts.size(1)
        valid_prompt_length = int(attention_mask[0, :prompt_length].sum().item())
        
        # print(f"prompt_length: {prompt_length}")
        # print(f"valid_prompt_length: {valid_prompt_length}")
        
        # 取出 prompt 中最后 valid_prompt_length 个 token（假设右侧对齐）
        valid_prompt_ids = prompts[:, -valid_prompt_length:]
        
        # 对 response 部分，从 prompt_length 开始的 attention_mask部分
        valid_response_length = int(attention_mask[0, prompt_length:].sum().item())
        # 取 response 中前 valid_response_length 个 token
        valid_response_ids = responses[:, :valid_response_length]
        
        # 拼接有效的 prompt 与 response token 序列
        sequences = torch.cat((valid_prompt_ids, valid_response_ids), dim=1)  # shape (1, L_effective)
        # 解码成字符串
        sequences_str = tokenizer.decode(sequences[0].tolist())
        
        # 计算 final_answer_reward
        # query_data = ast.literal_eval(query_datas[i]['ground_truth'])
        query_data = query_datas[i]['ground_truth']
        final_answer_reward, answer_format_score, final_answer_reward = calculate_final_answer_reward(sequences_str, query_data)
        answer_format_rewards.append(answer_format_score)
        final_answer_rewards.append(final_answer_reward)
        
        all_reward = 0.0
        
        if tool_stats[i] > 0:
            tool_rewards.append(tool_reward)
            all_reward = tool_reward + final_answer_reward
            # tool_rewards.append(0.0)
            # reward_value = final_answer_reward
        else:
            tool_rewards.append(0.0)
            all_reward = final_answer_reward
            
        if memory_stats[i] > 0:
            memory_rewards.append(memory_reward)
            all_reward = memory_reward + final_answer_reward
        else:
            memory_rewards.append(0.0)
            all_reward = final_answer_reward
        
        all_scores.append(all_reward)
        # 仅在最后一个有效token处设置奖励（稀疏奖励）
        if valid_response_length > 0:
            reward_tensor[i, valid_response_length - 1] = all_reward
    
    meta_info['reward/tool_rewards'] = tool_rewards
    meta_info['reward/memory_rewards'] = memory_rewards
    meta_info['reward/answer_format_rewards'] = answer_format_rewards
    meta_info['reward/final_answer_rewards'] = final_answer_rewards
    meta_info['reward/all_scores'] = all_scores
    data.meta_info.update(meta_info)

    return reward_tensor