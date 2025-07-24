import re
import torch
import string
import random
from abc import ABC
from verl import DataProto
from envs.tool_manager import TOOL_MANAGER_REGISTRY
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import tokenize_and_postprocess_data
from typing import List, Tuple
from PIL import Image
import sys



class MMEnv(ABC):
    def __init__(self, config, centralized_actor=None):
        tool_manager_name = config.get('tool_manager', 'qwen3')
        # 检查是否指定工具管理器，如果没有采用自适应模式Add commentMore actions
        if not tool_manager_name:
            tool_manager_name = "adaptive"
        if tool_manager_name.startswith('centralized_'):
            if centralized_actor is None:
                raise ValueError(f"使用集中式工具管理器 '{tool_manager_name}' 需要提供 centralized_actor 参数")
            self.tool_manager = TOOL_MANAGER_REGISTRY[tool_manager_name](
                verl_config=config, 
                centralized_actor_handle=centralized_actor
            )
        else:
            # 分布式模式，保持原有逻辑
            if tool_manager_name == 'adaptive':
                model_type = config.get('model_type')
                if 'qwen3' in model_type:
                    tool_manager_name = 'qwen3'
                elif 'qwen2' in model_type:
                    tool_manager_name = 'qwen2_5'
                elif 'qwen2_5_vl' in model_type:
                    tool_manager_name = 'qwen2_5_vl'
                elif 'llama' in model_type:
                    tool_manager_name = 'llama3'  
                else:
                    tool_manager_name = model_type
                    raise ValueError(f"'{tool_manager_name}' 需要进行适配，请添加一个对应的tool_manager")                   

            self.tool_manager = TOOL_MANAGER_REGISTRY[tool_manager_name](verl_config=config)
        
        self.max_prompt_length = config.get('max_prompt_length', 2048)
        self.use_verify_tool = False
        self.use_process_reward = config.get('use_process_reward', False)
        self.chat_template = config.get('chat_template', None)
        
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
        # prompt_str = tokenizer.decode(valid_prompt_ids)
        # response_str = tokenizer.decode(valid_response_ids)
        # response_str = response_str.replace("<|image_pad|>", "") 
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
    def _replace_all_image_pads(original_str, count):
        """
        将一个字符串中所有的 '<image_pad>' 替换为指定数量的 '<image_pad>'。

        参数:
        original_str (str): 原始输入字符串。
        count (int): 希望替换成的 '<image pad>' 的数量。

        返回:
        str: 替换后的新字符串。
        """
        # 1. 根据数量生成替换用的字符串
        # 例如，当 count=3 时，replacement 为 "<image pad> <image pad> <image pad>"
        replacement = ' '.join(['<image_pad>'] * count)
        # 2. 使用 replace() 方法进行全局替换
        new_str = original_str.replace('<image_pad>', replacement)
        return new_str
    
    def get_step_reward(self, responses, format_score=0.1):
        step_reward = [1] * len(responses)
        return step_reward 
    
    def _mm_process(self, mm_output, image_result, temp_next_obs, processor):
        if not mm_output:
            return None, None, temp_next_obs
        mm_data = processor.image_processor(image_result, return_tensors='pt')
        temp_multi_modal_data = {"pixel_values": mm_data["pixel_values"], "image_grid_thw": mm_data["image_grid_thw"]}
        temp_image_data = mm_output
        temp_next_obs = self._replace_all_image_pads(temp_next_obs, (mm_data["image_grid_thw"][0].prod() // (processor.image_processor.merge_size**2)))
        return temp_multi_modal_data, temp_image_data, temp_next_obs
    
    def step(self, responses, tokenizer, image_data: List[List[Image.Image]]):
        print("start to the env step", file=sys.stderr, flush=True)
        cur_actions, tool_results = self.tool_manager.execute_actions(responses=responses, image_data=image_data)
        next_obs, dones, valid_action, is_tool, new_image = [], [], [], [], []
        raw_prompt = []
        multi_modal_data = []
        valid_tool = []
        
        for action, tool_result in zip(cur_actions, tool_results):
            temp_valid_tool = 0
            temp_multi_modal_data = None
            temp_raw_prompt = None
            if action == 'answer':
                temp_next_obs, temp_done, temp_valid_action, temp_is_tool, temp_image_data, temp_valid_tool, temp_raw_prompt = '', True, 0, 1, None, 0, None
            elif action == 'error':
                temp_next_obs = self.tool_manager.get_prompt(
                    input_data=tool_result, 
                    tokenizer=tokenizer,
                    mode='tool_call', 
                    add_generation_prompt=True
                )
                temp_done, temp_valid_action, temp_is_tool, temp_image_data, temp_raw_prompt = False, 0, 0, None, temp_next_obs
            elif action == 'actions':
                mm_output, image_result = (True, tool_result[1]) if isinstance(tool_result, Tuple) else (False, None) 
                temp_next_obs = self.tool_manager.get_prompt(
                    input_data=tool_result if not mm_output else tool_result[0],
                    tokenizer=tokenizer,
                    mode='tool_call',
                    add_generation_prompt=True
                )
                temp_raw_prompt = temp_next_obs.deepcopy()
                temp_multi_modal_data, temp_image_data, temp_next_obs = self._mm_process(mm_output, image_result, temp_next_obs, self.processor)
                temp_done, temp_valid_action, temp_is_tool, temp_valid_tool = False, 0, 1, 1 if mm_output else 0

            else:
                raise ValueError('Unexpected action: {}'.format(action))
            
            next_obs.append(temp_next_obs)
            dones.append(temp_done)
            valid_action.append(temp_valid_action)
            is_tool.append(temp_is_tool)
            new_image.append(temp_image_data)
            multi_modal_data.append(temp_multi_modal_data)
            valid_tool.append(temp_valid_tool)
            raw_prompt.append(temp_raw_prompt)

        print(f"image valid tool execute is {sum(valid_tool)} and overall batch size is  {len(dones)}",file=sys.stderr, flush=True)
        assert sum(valid_tool) == sum(1 for item in new_image if isinstance(item, Image.Image))
        breakpoint()
        return next_obs, dones, valid_action, is_tool, new_image, raw_prompt, multi_modal_data, valid_tool
    

    def compute_score(self, reward_rollout_wg, reward_tokenizer, tokenizer, data: DataProto, if_val=False, use_process_reward=False):
        if reward_rollout_wg is not None:
            scores = self._compute_score_with_reward_rollout_wg(reward_rollout_wg, reward_tokenizer, data)
        else:
            score = self._compute_score_with_rules(data, tokenizer, if_val=if_val)
            if use_process_reward and not if_val:
                scores = []
                for i in range(len(data)):
                    data_item = data[i]
                    tool_use_score = data_item.batch['tool_use_scores']
                    validate_score = tool_use_score[ ~ torch.isnan(tool_use_score)].tolist()
                    scores.append(validate_score + score[i])
            else:
                scores = score
        
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
