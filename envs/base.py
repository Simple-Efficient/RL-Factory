import re
import torch
import string
import random
from abc import ABC
from verl import DataProto
from envs.tool_manager import TOOL_MANAGER_REGISTRY
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import tokenize_and_postprocess_data


class Env(ABC):
    def __init__(self, config):
        tool_manager_name = config.get('tool_manager', 'qwen3')
        self.tool_manager = TOOL_MANAGER_REGISTRY[tool_manager_name](verl_config=config)
        self.max_prompt_length = config.get('max_prompt_length', 2048)
        self.use_verify_tool = False
        self.use_simulated_user_feedback = config.get('use_simulated_user_feedback', True)

        
    def verify_tool(self, data_source, solution_str, ground_truth, extra_info):
        # If you need a tool to evaluate the generated response, you need to modify the following code
        # the data would be stored in data[i].non_tensor_batch['reward_model']['ground_truth']['verified_results']
        # print('verify tool start')
        raise NotImplementedError

    def _process_data(self, data_item, tokenizer):
        # process the data_item to the token and decode them
        prompt_ids = data_item.batch['prompts']

        prompt_length = prompt_ids.shape[-1]

        valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch['responses']
        valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        # decode
        prompt_str = tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
        data_source = data_item.non_tensor_batch['data_source']
        extra_info = data_item.non_tensor_batch.get('extra_info', None)
        
        return {
            'prompt_str': prompt_str,
            'response_str': response_str,
            'ground_truth': ground_truth,
            'data_source': data_source,
            'extra_info': extra_info
        }

    def step(self, responses, tokenizer):
        cur_actions, tool_results = self.tool_manager.execute_actions(responses=responses)
        next_obs, dones, valid_action, is_tool = [], [], [], []
        user_feedback_flag = []

        for action, tool_result in zip(cur_actions, tool_results):
            if action == 'answer':
                if self.use_simulated_user_feedback:
                    if random.random() < self.tool_manager.user_feedback_prob:
                        user_feedback_flag.append(1)
                        temp_next_obs, temp_done, temp_valid_action, temp_is_tool = '', False, 1, 0
                    else:
                        user_feedback_flag.append(0)
                        temp_next_obs, temp_done, temp_valid_action, temp_is_tool = '', True, 1, 0
                    temp_next_obs, temp_done, temp_valid_action, temp_is_tool = '', True, 1, 0
                else:
                    temp_next_obs, temp_done, temp_valid_action, temp_is_tool = '', True, 1, 0
            elif action == 'error':
                temp_next_obs = self.tool_manager.get_prompt(
                    input_data=tool_result, 
                    tokenizer=tokenizer,
                    mode='tool_call', 
                    add_generation_prompt=True
                )
                temp_done, temp_valid_action, temp_is_tool = False, 0, 0
            elif action == 'actions':
                temp_next_obs = self.tool_manager.get_prompt(
                    input_data=tool_result, 
                    tokenizer=tokenizer,
                    mode='tool_call',
                    add_generation_prompt=True
                )
                temp_done, temp_valid_action, temp_is_tool = False, 1, 1
            else:
                raise ValueError('Unexpected action: {}'.format(action))
            next_obs.append(temp_next_obs)
            dones.append(temp_done)
            valid_action.append(temp_valid_action)
            is_tool.append(temp_is_tool)

        if self.use_simulated_user_feedback:
            next_obs, dones, valid_action, is_tool = self.tool_manager.simulated_user_feedback(responses, tokenizer, user_feedback_flag, next_obs, dones, valid_action, is_tool)
        return next_obs, dones, valid_action, is_tool



    def compute_score(self, reward_rollout_wg, reward_tokenizer, tokenizer, data: DataProto, if_val=False):
        if reward_rollout_wg is not None:
            scores = self._compute_score_with_reward_rollout_wg(reward_rollout_wg, reward_tokenizer, data)
        else:
            scores = self._compute_score_with_rules(data, tokenizer, if_val=if_val)
        
        return scores
    
    def _compute_score_with_rules(self, data, tokenizer, if_val=False):
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            # process the data_item to the token and decode them
            processed_data = self._process_data(data_item=data_item, tokenizer=tokenizer)
            ground_truth, response_str = processed_data['ground_truth'], processed_data['response_str']

            # reserved for compatibility
            prompt_str, data_source, extra_info = processed_data['prompt_str'], processed_data['data_source'], processed_data['extra_info']

        scores = [[0.0]] * len(data)
        
        return scores

    def get_prompt_for_reward(self, reward_tokenizer, data: DataProto):
        reward_prompt_strs = []
        
        for i in range(len(data)):
            data_item = data[i]
            step_mask = data.batch['step_mask'][i]
            
            # 获取有效的prompt和response
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            
            # 获取非tensor数据
            non_tensor_data = {
                'data_source': data_item.non_tensor_batch['data_source'],
                'ground_truth': data_item.non_tensor_batch['reward_model']['ground_truth'],
                'extra_info': data_item.non_tensor_batch.get('extra_info', None)
            }
            
            # 找到所有step的位置
            mask_indices = torch.where(step_mask == 1)[0]
            assert len(mask_indices) > 0, "no step mask"
            
            # 处理所有responses
            reward_prompt_str_list = []
            start_idx = 0
            
            for end_idx in mask_indices:
                # 截取当前response
                current_response = valid_response_ids[start_idx:end_idx]
                current_response_str = reward_tokenizer.decode(current_response, skip_special_tokens=True)
                
                # 生成prompt
                reward_prompt_str = self._get_single_prompt_str(
                    data_source=non_tensor_data['data_source'],
                    solution_str=current_response_str,
                    ground_truth=non_tensor_data['ground_truth'],
                    extra_info=non_tensor_data['extra_info'],
                    reward_tokenizer=reward_tokenizer
                )
                reward_prompt_str_list.append(reward_prompt_str)
                
                start_idx = end_idx

            assert start_idx == len(valid_response_ids) - 1, "start_idx is not the last index"

            reward_prompt_strs.append(reward_prompt_str_list)
        
        return reward_prompt_strs
    
    def _get_single_prompt_str(self, data_source, solution_str, ground_truth, extra_info, reward_tokenizer):
        # If you need use the reasoning model to generate the reward, you need to modify the following code
        # result = reward_tokenizer.apply_chat_template([
        #     {'role': 'system', 'content': 'You are a assistant. '},
        #     {'role': 'user', 'content': '你是Qwen吗？你只需要回答是或者不是即可。'}
        # ], add_generation_prompt=True, tokenize=False)
        # return result
        raise NotImplementedError

    def _compute_score_with_reward_rollout_wg(self, reward_rollout_wg, reward_tokenizer, data: DataProto):
        # 基于actor rollout的回答和真实答案构造judge model的prompts
        reward_prompt_strs = self.get_prompt_for_reward(reward_tokenizer, data)
        
        # 展平reward_prompt_strs为一个batch
        flat_prompts = []
        original_shapes = []  # 记录每个样本的prompt数量
        for prompts in reward_prompt_strs:
            original_shapes.append(len(prompts))
            flat_prompts.extend(prompts)
        
        # 将flat_prompts转换为DataProto格式
        input_ids = []
        attention_mask = []
        for prompt in flat_prompts:
            # 使用 tokenize_and_postprocess_data 处理每个 prompt
            ids, mask = tokenize_and_postprocess_data(
                prompt=prompt,
                tokenizer=reward_tokenizer,
                max_length=self.max_prompt_length,
                pad_token_id=reward_tokenizer.pad_token_id,
                left_pad=True,
                truncation='error'
            )
            input_ids.append(ids)
            attention_mask.append(mask)

        # 创建DataProto
        tensors = {
            "input_ids": torch.cat(input_ids, dim=0),
            "attention_mask": torch.cat(attention_mask, dim=0)
        }
        
        # 计算position_ids
        tensors["position_ids"] = compute_position_id_with_mask(tensors["attention_mask"])

        data_proto = DataProto.from_dict(tensors=tensors)
        
        # 生成responses
        responses_data = reward_rollout_wg.generate_sequences(data_proto)
        
        # 计算每个response的分数
        flat_scores = []
        for i, temp_response_data in enumerate(responses_data):
            # 找到对应的data_item
            data_idx = 0
            prompt_count = 0
            while data_idx < len(original_shapes) and prompt_count + original_shapes[data_idx] <= i:
                prompt_count += original_shapes[data_idx]
                data_idx += 1
            
            data_item = data[data_idx]

            temp_prompt_ids = temp_response_data.batch['prompts']
            temp_prompt_length = temp_prompt_ids.shape[-1]
            temp_response_ids = temp_response_data.batch['responses']
            temp_valid_response_length = temp_response_data.batch['attention_mask'][temp_prompt_length:].sum()
            temp_valid_response_ids = temp_response_ids[:temp_valid_response_length]
            response_str = reward_tokenizer.decode(temp_valid_response_ids, skip_special_tokens=True)

            score = self._compute_single_score_with_reward_rollout_wg(
                data_item.non_tensor_batch['data_source'], 
                response_str, 
                data_item.non_tensor_batch['reward_model']['ground_truth'], 
                data_item.non_tensor_batch.get('extra_info', None)
            )
            flat_scores.append(score)
        
        assert data_idx == len(data) - 1, "data_idx is {}".format(data_idx)
        
        # 将scores重新组织为原来的形状
        scores = []
        start_idx = 0
        for shape in original_shapes:
            end_idx = start_idx + shape
            scores.append(flat_scores[start_idx:end_idx])
            start_idx = end_idx
        
        return scores
    
    def _compute_single_score_with_reward_rollout_wg(self, data_source, solution_str, ground_truth, extra_info):
        # print("solution_str: ", solution_str)
        # return 1.0
        raise NotImplementedError
