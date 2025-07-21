import os
import logging
import torch
import itertools
import numpy as np
from verl import DataProto
from tensordict import TensorDict

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

class ToolUtils:
    def __init__(self, tokenizer, meta_info, config, env_object):
        self.tokenizer = tokenizer  
        self.final_str = config.stop[-1] if config.stop else ''
        self.config_prompt_length = config.prompt_length
        self.config_response_length = config.response_length
        self.stop_id = self.tokenizer.encode(config.stop[0], add_special_tokens=False)[0]
        self.max_turns = config.max_turns
        self.max_prompt_length = config.prompt_length
        self.max_tool_response_length = config.tool_response_length
        
        pad_token_id = meta_info.get('pad_token_id')
        if pad_token_id is not None:
            self.pad_token_id = pad_token_id
        else:
            eos_token_id = meta_info.get('eos_token_id')
            if isinstance(eos_token_id, (list, tuple)):
                self.pad_token_id = eos_token_id[-1]
            else:
                self.pad_token_id = eos_token_id
                
        eos_token_id = meta_info.get('eos_token_id')
        if isinstance(eos_token_id, (list, tuple)):
            self.eos_token_id = eos_token_id[0]
        else:
            self.eos_token_id = eos_token_id
        
        self.meta_info = meta_info
        self.loop_cnt = 0

        self.env_object = env_object
        
    def postprocess_output(self, output: DataProto, step: int):
        '''output: cpu'''
        # init loop responses token
        if self.loop_cnt == 0:
            self.batch_size = output.batch.batch_size[0]
            self.loop_responses_token = [[] for _ in range(self.batch_size)]
            self.init_prompt_token = output.batch.get('prompts')
            self.tool_use = [[] for _ in range(self.batch_size)]
            prompt_length = self.init_prompt_token.shape[-1]
            self.init_attention_mask = output.batch.get('attention_mask')[:,:prompt_length]  

            batch_idxs = list(range(self.batch_size))
            for idx in range(self.batch_size):
                prompt_token = self.init_prompt_token[idx]
                prompt_token_list = prompt_token[prompt_token != self.pad_token_id].tolist()
                self.loop_responses_token[idx].append(prompt_token_list)
        else:
            batch_idxs = output.meta_info['index']

        responses = output.batch.get('responses')

        process_response = []
        for idx, batch_idx in enumerate(batch_idxs):
            response_token = responses[idx]
            response_token_list = response_token[response_token != self.pad_token_id].tolist()
            if self.env_object.use_process_reward:
            # assure last token is stop token （add or change）
                if response_token_list[-1] != self.stop_id:
                    if len(response_token_list) != self.config_response_length:
                        response_token_list.append(self.stop_id)
                    else:
                        response_token_list[-1] = self.stop_id
            self.loop_responses_token[batch_idx].append(response_token_list)
            process_response.append(response_token_list)

        # decode responses for env step (detect tool call)
        responses_str = self.tokenizer.batch_decode(
            process_response,
            skip_special_tokens=False,
        )
        infos_str, dones, _, _ = self.env_object.step(
            responses=responses_str, tokenizer=self.tokenizer
        )
        #if not use_process_reward will be 0
        if self.env_object.use_process_reward:
            step_scores = self.env_object.get_step_reward(responses=responses_str)
        else:
            step_scores = [0] * len(responses_str)

        # encode infos for next prompt
        info_tokens = self.tokenizer(infos_str, truncation=True, max_length=self.max_tool_response_length).input_ids
        next_prompt_token = []
        next_prompt_length = []
        next_sample_idx = []
        for idx, batch_idx in enumerate(batch_idxs):
            if not dones[idx]:
                info_token_list = info_tokens[idx]
                self.loop_responses_token[batch_idx].append(info_token_list)
                next_sample_idx.append(batch_idx)
                promt_token = list(itertools.chain.from_iterable(self.loop_responses_token[batch_idx]))
                next_prompt_token.append(promt_token)
                next_prompt_length.append(len(promt_token))
                # get process reward 
                self.tool_use[batch_idx].append(step_scores[idx])
        
        if len(next_prompt_token) == 0:
            return 
        
        # left pad
        max_len = max(max(next_prompt_length), self.config_prompt_length)
        next_prompt_token_pad = []
        for prompt_token in next_prompt_token:
            token = [self.pad_token_id] * (max_len - len(prompt_token)) + prompt_token
            next_prompt_token_pad.append(token)

        next_input_ids = torch.tensor(next_prompt_token_pad, dtype=torch.int64)
        next_attention_mask = next_input_ids != self.pad_token_id
        # position_ids = (torch.cumsum(next_attention_mask, dim=1) - 1) * next_attention_mask
        position_ids = torch.clip(torch.cumsum(next_attention_mask, dim=-1) - 1, min=0, max=None) * next_attention_mask
        
        max_len = self.config_prompt_length
        next_batch = TensorDict(
            {
                'input_ids': next_input_ids[:, -max_len:],
                'position_ids': position_ids[:, -max_len:],
                'attention_mask': next_attention_mask[:, -max_len:]
            },
            batch_size=next_input_ids.shape[0]
        )
        raw_prompt_ids = np.empty(len(next_prompt_token), dtype=object)
        # raw_prompt_ids[:] = [np.array(x[-max_len:]) for x in next_prompt_token]
        raw_prompt_ids[:] = [x[-max_len:] for x in next_prompt_token]

        next_data = DataProto(batch=next_batch, non_tensor_batch={'raw_prompt_ids': raw_prompt_ids})
        next_data.meta_info.update(self.meta_info)
        next_data.meta_info['index'] = next_sample_idx
        next_data.meta_info['do_sample'] = False # step > 0 does not do sample
        self.loop_cnt += 1

        return next_data

    def compose_final_output(self, step, group) -> DataProto:
        """Compose final generation output."""
        input_ids_list = []
        loss_mask_list = []
        length_list = []
        
        for idx, responses in enumerate(self.loop_responses_token):
            # loss_mask = [0]*len(responses[0]) # init_prompt loss
            loss_mask = []
            prompts_list = list(itertools.chain.from_iterable(responses[1:]))
            # responses_token: [prompt_token, reponse_token_1, info_token_1, response_token_2....]
            for turn_idx in range(len(responses[1:])): 
                length = len(responses[turn_idx + 1])
                loss_mask.extend([(turn_idx + 1) % 2] * length)
            input_ids_list.append(prompts_list)
            loss_mask_list.append(loss_mask)
            length_list.append(len(prompts_list))
        
        # max_len = max(max(length_list), self.config_response_length)
        max_response_length = torch.tensor([max(length_list)], device=torch.cuda.current_device())
        # because only tp=0 will exec postprocess_output and compose_final_output
        # so we exec all_reduce in specified group(dp group)
        torch.distributed.all_reduce(max_response_length, op=torch.distributed.ReduceOp.MAX, group=group)
        max_len = int(max_response_length)
        
        # right pad
        input_ids = []
        loss_mask = []
        for idx, input_ids in enumerate(input_ids_list):
            input_ids = input_ids + [self.pad_token_id] * (max_len - len(input_ids))
            loss_mask = loss_mask_list[idx] + [0] * (max_len - len(loss_mask_list[idx]))
            input_ids_list[idx] = input_ids
            loss_mask_list[idx] = loss_mask[0:max_len]

        response_token = torch.tensor(input_ids_list, dtype=torch.int64)[:,:max_len]
        response_loss_mask = torch.tensor(loss_mask_list, dtype=torch.float32)
        response_attention_mask = (response_token != self.pad_token_id).long()

        # get the max length of the process rewards
        max_tool_use_len = self.max_turns
        for tool_use_item in self.tool_use:
            max_tool_use_len = max(max_tool_use_len, len(tool_use_item))
        tool_use_tensor = []

        # Pad tool_use to have consistent dimensions
        for idx in range(len(self.tool_use)):
            if not self.tool_use[idx]:
                padded_tool_use = [torch.nan] * max_tool_use_len
            else:
                padded_tool_use = self.tool_use[idx] + [torch.nan] * (max_tool_use_len - len(self.tool_use[idx]))
            tool_use_tensor.append(padded_tool_use)

        tool_use_score = torch.tensor(tool_use_tensor)
        
        input_ids = torch.cat([self.init_prompt_token, response_token], dim=-1)
        attention_mask = torch.cat([self.init_attention_mask, response_attention_mask], dim=-1)
        # position_ids = torch.cumsum(attention_mask, dim=1, dtype=torch.long) - 1
        position_ids = torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0, max=None) * attention_mask
        loss_mask = torch.cat([torch.zeros_like(self.init_attention_mask, dtype=torch.float32), response_loss_mask], dim=-1)
        final_batch = TensorDict(
            {
                'prompts': self.init_prompt_token,
                'responses': response_token,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'position_ids': position_ids,
                'loss_mask': loss_mask,
                'tool_use_scores': tool_use_score
            },
            batch_size=self.batch_size,
        )  

        final_output = DataProto(batch=final_batch)
        return final_output