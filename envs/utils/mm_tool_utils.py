import torch
import os
import logging
import itertools
import numpy as np
from verl import DataProto
import torch.distributed as dist
from tensordict import TensorDict
from typing import List
from PIL import Image
import sys
from .tool_utils import ToolUtils
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

class MMToolUtils(ToolUtils):
    def __init__(self, tokenizer, processor, meta_info, config, env_object):
        super().__init__(tokenizer, meta_info, config, env_object)
        self.processor = processor

        
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
        
        # Qwen2.5VL mrope specific parameters
        self.vision_start_token_id = getattr(tokenizer, 'vision_start_token_id', 151652)
        self.image_token_id = getattr(tokenizer, 'image_token_id', 151655) 
        self.video_token_id = getattr(tokenizer, 'video_token_id', 151656)
        self.spatial_merge_size = 2
        self.tokens_per_second = 2

    def postprocess_output(self, output: DataProto, step: int):
        print(f"[ToolUtils] start the {step}th postprocess",file=sys.stderr,flush=True)
        '''output: cpu'''
        if self.loop_cnt == 0:
            self.batch_size = output.batch.batch_size[0]
            self.loop_responses_token = [[] for _ in range(self.batch_size)]
            
            self.loop_raw_responses_token = [[] for _ in range(self.batch_size)]
            self.init_prompt_token = output.batch.get('prompts')
            self.raw_prompt_id = output.non_tensor_batch.get('raw_prompt_ids')

            self.multi_modal_inputs = [[] for _ in range(self.batch_size)]
            self.image_list = [i["image"] for i in output.non_tensor_batch["multi_modal_data"]]
            self.tool_use = [[] for _ in range(self.batch_size)]
            prompt_length = self.init_prompt_token.shape[-1]
            self.init_attention_mask = output.batch.get('attention_mask')[:,:prompt_length]

            batch_idxs = list(range(self.batch_size))
            
            for idx in range(self.batch_size):

                self.multi_modal_inputs[idx].append(output.non_tensor_batch["multi_modal_inputs"][idx])

                prompt_token = self.init_prompt_token[idx]
                assert isinstance(prompt_token, torch.Tensor)
                prompt_token_list = prompt_token[prompt_token != self.pad_token_id].tolist()

                self.loop_responses_token[idx].append(prompt_token_list)
                self.loop_raw_responses_token[idx].append(self.raw_prompt_id[idx])

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
            self.loop_raw_responses_token[batch_idx].append(response_token_list)
            process_response.append(response_token_list)

        responses_str = self.tokenizer.batch_decode(
            process_response,
            skip_special_tokens=False,
        )
        
        # breakpoint()
        infos_str, dones, _, _, new_image_data, raw_prompt, multi_modal_data, valid_tool = self.env_object.step(
            responses=responses_str, processor=self.processor, image_data=[self.image_list[idx] for idx in batch_idxs]
        )
        # breakpoint()
        for idx, batch_idx in enumerate(batch_idxs):
            if multi_modal_data[idx] is not None:
                self.multi_modal_inputs[batch_idx].append(multi_modal_data[idx])

        #if not use_process_reward will be 0
        if self.env_object.use_process_reward:
            step_scores = self.env_object.get_step_reward(responses=responses_str)
        else:
            step_scores = [0] * len(responses_str)


        def tokenize_infos(infos_str):
            if infos_str == "":
                return []
            if infos_str is None:
                return []
            try:
                info_tokens = self.tokenizer(text=infos_str).input_ids
            except:
                raise ValueError(f"tokenize_infos error, the infos_str is {infos_str}")
            return info_tokens

        next_prompt_token = []
        next_prompt_length = []
        next_sample_idx = []
        next_image_data = []
        next_raw_prompt_token = []
        next_raw_prompt_length = []

        for idx, batch_idx in enumerate(batch_idxs):
            if not dones[idx]:
                info_token_list = tokenize_infos(infos_str[idx])
                raw_prompt_list = tokenize_infos(raw_prompt[idx])
                # breakpoint()
                self.loop_responses_token[batch_idx].append(info_token_list)
                self.loop_raw_responses_token[batch_idx].append(raw_prompt_list)
                next_sample_idx.append(batch_idx)
                promt_token = list(itertools.chain.from_iterable(self.loop_responses_token[batch_idx]))
                raw_prompt_token = list(itertools.chain.from_iterable(self.loop_raw_responses_token[batch_idx]))
                next_prompt_token.append(promt_token)
                next_raw_prompt_token.append(raw_prompt_token)
                next_prompt_length.append(len(promt_token))
                next_raw_prompt_length.append(len(raw_prompt_token))
                self.tool_use[batch_idx].append(step_scores[idx])
                if valid_tool[idx]:
                    self.image_list[batch_idx].append(new_image_data[idx])
                next_image_data.append(self.image_list[batch_idx])
        
        if len(next_prompt_token) == 0:
            return 
        
        # left pad
        max_len = (max(next_prompt_length) + self.config_prompt_length)
        
        next_prompt_token_pad = []
        for prompt_token in next_prompt_token:
            token = [self.pad_token_id] * (max_len - len(prompt_token)) + prompt_token
            next_prompt_token_pad.append(token)

        next_input_ids = torch.tensor(next_prompt_token_pad, dtype=torch.int64)
        next_attention_mask = next_input_ids != self.pad_token_id
        position_ids = torch.clip(torch.cumsum(next_attention_mask, dim=-1) - 1, min=0, max=None) * next_attention_mask
        next_batch = TensorDict(
            {
                'input_ids': next_input_ids[:, -max_len:].cpu().share_memory_(),
                'position_ids': position_ids[:, -max_len:].cpu().share_memory_(),
                'attention_mask': next_attention_mask[:, -max_len:].to(dtype=torch.int64).cpu().share_memory_()
            },
            batch_size=next_input_ids.shape[0]
        ).share_memory_()
        
        raw_prompt_ids = np.empty(len(next_raw_prompt_token), dtype=object)
        raw_prompt_ids[:] = [x[-max_len:] for x in next_raw_prompt_token]

        next_image_data = np.array([{"image": img} for img in next_image_data], dtype=object)
        next_data = DataProto(batch=next_batch, non_tensor_batch={'raw_prompt_ids': raw_prompt_ids, 'multi_modal_data': next_image_data, })
        next_data.meta_info.update(self.meta_info)
        next_data.meta_info['index'] = next_sample_idx
        next_data.meta_info['do_sample'] = False # step > 0 does not do sample
        self.loop_cnt += 1
        return next_data

    def compose_final_output(self, step, group) -> DataProto:
        print(f"[compose_final_output] start compose the final output, step is {step}", flush=True, file=sys.stderr)
        """Compose final generation output."""
        input_ids_list = []
        loss_mask_list = []
        length_list = []
        
        for idx, responses in enumerate(self.loop_responses_token):
            loss_mask = []
            prompts_list = list(itertools.chain.from_iterable(responses[1:]))
            # responses_token: [prompt_token, reponse_token_1, info_token_1, response_token_2....]
            for turn_idx in range(len(responses[1:])): 
                length = len(responses[turn_idx + 1])
                loss_mask.extend([(turn_idx + 1) % 2] * length)
            input_ids_list.append(prompts_list)
            loss_mask_list.append(loss_mask)
            length_list.append(len(prompts_list))

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
        multi_modal_inputs = np.array(self.merge_multi_modal_inputs(self.multi_modal_inputs))
        input_ids = torch.cat([self.init_prompt_token, response_token], dim=-1)
        attention_mask = torch.cat([self.init_attention_mask, response_attention_mask], dim=-1)
        if self.processor is not None and self.processor.image_processor._processor_class== "Qwen2_5_VLProcessor":
            from verl.models.transformers.qwen2_vl import get_rope_index
            position_ids = [] 
            # breakpoint()
            for idx, input_id in enumerate(input_ids):
                try:
                    position_id = get_rope_index(
                            self.processor,
                            input_ids=input_id,
                            image_grid_thw=multi_modal_inputs[idx][0].get("image_grid_thw"),
                            video_grid_thw=multi_modal_inputs[idx][0].get("video_grid_thw"),
                            attention_mask=attention_mask[idx],
                        )
                    
                    position_ids.append(position_id)
                except:
                    # 报错raise
                    print(f"get_rope_index error, the input_ids is {input_id}")
                    breakpoint()
            
            # Stack the 2D position_ids into a 3D tensor
            position_ids = torch.stack(position_ids, dim=0)
            
        else:
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
        image_list = np.array([{"image": img} for img in self.image_list], dtype=object)
        modal_inputs = np.array([inputs[0] for inputs in multi_modal_inputs])
        
        final_output = DataProto(
            batch=final_batch,
            non_tensor_batch={
                'multi_modal_data': image_list, 
                "multi_modal_inputs": modal_inputs
            }
        )
        print("[Final Compose] finish final compose", flush=True, file=sys.stderr)
        return final_output



    
    def merge_tensor_dicts(self, list_of_dicts):

        if not list_of_dicts:
            return {}

        keys = list_of_dicts[0].keys()
        merged_dict = {}

        for key in keys:
            tensors_to_concat = [d[key] for d in list_of_dicts]
            merged_dict[key] = torch.cat(tensors_to_concat, dim=0)

        return merged_dict

    def merge_multi_modal_inputs(self,multi_modal_inputs):
        """
        Merges dictionaries within each sublist of multi_modal_inputs.
        """
        merged_inputs = []
        for sublist in multi_modal_inputs:
            merged_dict = self.merge_tensor_dicts(sublist)
            merged_inputs.append([merged_dict])
        return merged_inputs
