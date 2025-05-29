# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import asyncio
from verl import DataProto
from envs.base import Env


class AsyncRewardManager:
    """The reward manager with async processing.
    """
    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.env_object = None
        self.reward_rollout_wg = None
        self.reward_tokenizer = None
        self.reward_fn_key = reward_fn_key
        self.if_val = False
        self.use_process_reward = False
        self.stop_token = None
    
    def set_env_object(self, env_object):
        self.env_object: Env = env_object
    
    def set_reward_rollout_wg(self, reward_rollout_wg):
        self.reward_rollout_wg = reward_rollout_wg
    
    def set_reward_tokenizer(self, reward_tokenizer):
        self.reward_tokenizer = reward_tokenizer
    
    def verify(self, data):
        device = data[0].batch['prompts']
        
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        if self.env_object.use_verify_tool:
            data = self._get_verified_results(data)
        
        # 初始化step_mask
        step_mask = torch.zeros_like(data.batch['responses'], dtype=torch.long)
        for i in range(len(data)):
            data_item = data[i]
            # 获取prompts的长度
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            
            # 获取response部分的attention_mask
            response_attention_mask = data_item.batch['attention_mask'][prompt_length:]
            # 找到最后一个为1的位置
            last_one_idx = torch.where(response_attention_mask == 1)[0][-1]
            # 在最后一个为1的位置设置为1
            step_mask[i, last_one_idx] = 1

        # 将step_mask添加到data中
        data.batch['step_mask'] = step_mask

        scores = self.env_object.compute_score(self.reward_rollout_wg, self.reward_tokenizer, self.tokenizer, data, if_val=self.if_val)
        
        data.batch['acc'] = torch.tensor(scores, dtype=torch.float32, device=device)

        return scores
    

    def find_tool_end_positions(self, data: DataProto, tokenizer):
        """Find positions of all stop_tokens in responses, marking the end of tool calls.
        
        Args:
            data (DataProto): The data containing responses
            tokenizer: The tokenizer to decode token IDs
            
        Returns:
            list: A list of dictionaries, each containing positions of stop_token for one sample
        """
        tool_end_positions = []
        
        # Get the token ID for stop_token
        im_end_token_id = tokenizer.encode(self.stop_token, add_special_tokens=False)
        if isinstance(im_end_token_id, list) and len(im_end_token_id) > 0:
            im_end_token_id = im_end_token_id[0]
        
        for i in range(len(data)):
            data_item = data[i]
            
            # Get the entire response tokens
            response_ids = data_item.batch['responses']
            
            # Find all occurrences of im_end_token_id in response_ids
            positions = []
            
            # Convert to numpy for easier processing if it's a tensor
            if isinstance(response_ids, torch.Tensor):
                response_ids_np = response_ids.cpu().numpy()
            else:
                response_ids_np = response_ids
                
            # Find all positions where the token ID matches im_end_token_id
            for idx, token_id in enumerate(response_ids_np):
                if token_id == im_end_token_id:
                    positions.append(idx)
            
            tool_end_positions.append({
                'response_length': len(response_ids_np),
                'im_end_positions': positions,  # Positions in the response_ids array
                'im_end_count': len(positions)
            })
        
        return tool_end_positions
    
    def get_step_mask(self, data: DataProto):

        # 初始化step_mask
        step_mask = torch.zeros_like(data.batch['responses'], dtype=torch.long)

        #找到所有stop token的位置
        tool_end_positions = self.find_tool_end_positions(data,self.tokenizer)

        for i in range(len(data)):
            data_item = data[i]
            # 获取prompts的长度
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            response_attention_mask = data_item.batch['attention_mask'][prompt_length:]
            last_one_idx = torch.where(response_attention_mask == 1)[0][-1]
            
            step_index=[]
            #记录每个<tool_call></tool_call>以及<answer></answer>导致的停止位置
            if tool_end_positions[i]['im_end_count']%2 == 1:
                for j in range((tool_end_positions[i]['im_end_count'] // 2) + 1):
                    step_index.append(tool_end_positions[i]['im_end_positions'][2 * j])
            else:
                for j in range((tool_end_positions[i]['im_end_count'] // 2)):
                    step_index.append(tool_end_positions[i]['im_end_positions'][2 * j])
                step_index.append(last_one_idx)

            #根据是否使用过程奖励来定义step_mask
            if self.use_process_reward:
                for idx in step_index:
                    step_mask[i, idx] = 1
            else:
                step_mask[i, last_one_idx] = 1

        return step_mask

    
    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""
        
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

         #if use process reward
        self.use_process_reward = self.env_object.use_process_reward

        #获取step_mask
        step_mask = self.get_step_mask(data)

        # 将step_mask添加到data中
        data.batch['step_mask'] = step_mask

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        scores = self.env_object.compute_score(self.reward_rollout_wg, self.reward_tokenizer, self.tokenizer, data, if_val=self.if_val,use_process_reward=self.use_process_reward)

        reward_tensor = self._set_reward_tensor(scores, data)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": {},
            }
        else:
            return reward_tensor
    
    def _set_reward_tensor(self, scores, data: DataProto):
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        step_mask = data.batch['step_mask']
        for i in range(len(data)):
            cur_step_mask, cur_scores = step_mask[i], scores[i]
            mask_indices = torch.where(cur_step_mask == 1)[0]
            assert cur_step_mask.sum() == len(cur_scores)

            for j, idx in enumerate(mask_indices):
                reward_tensor[i, idx] = cur_scores[j]
            '''
            if self.use_process_reward:
                for j, idx in enumerate(mask_indices):
                    reward_tensor[i, idx] = cur_scores[j]
            else:
                 for j, idx in enumerate(mask_indices):
                    if j == len(mask_indices)-1:
                        reward_tensor[i, idx] = cur_scores[-1]
                        #reward_tensor[i,idx]=cur_scores.sum()
            '''

        return reward_tensor
    
    def _get_verified_results(self, data: DataProto):
        async def _get_single_result(data_source, solution_str, ground_truth, extra_info):
            # 使用asyncio.to_thread将同步函数转换为异步操作
            return await asyncio.to_thread(
                self.env_object.verify_tool, 
                data_source, 
                solution_str, 
                ground_truth, 
                extra_info
            )
        
        async def _process_all():
            tasks = []
            for i in range(len(data)):
                data_item = data[i]  # DataProtoItem

                prompt_ids = data_item.batch['prompts']
                prompt_length = prompt_ids.shape[-1]

                valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                response_ids = data_item.batch['responses']
                valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]

                # decode
                prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

                ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
                data_source = data_item.non_tensor_batch['data_source']
                extra_info = data_item.non_tensor_batch.get('extra_info', None)

                # 创建异步任务
                task = _get_single_result(data_source, response_str, ground_truth, extra_info)
                tasks.append(task)

            # 并行执行所有任务
            return await asyncio.gather(*tasks)

        # 在同步函数中运行异步代码
        results = asyncio.run(_process_all())

        for i in range(len(data)):
            data[i].non_tensor_batch['reward_model']['ground_truth']['verified_results'] = results[i]

        return data