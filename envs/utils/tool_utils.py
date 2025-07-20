import torch
import itertools
import numpy as np
from verl import DataProto
import torch.distributed as dist
from tensordict import TensorDict
from typing import List
from PIL import Image
import sys
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

class ToolUtils:
    def __init__(self, tokenizer, processor, meta_info, config, env_object):
        self.tokenizer = tokenizer  
        self.processor = processor
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
        
        # Qwen2.5VL mrope specific parameters
        self.vision_start_token_id = getattr(tokenizer, 'vision_start_token_id', 151652)
        self.image_token_id = getattr(tokenizer, 'image_token_id', 151655) 
        self.video_token_id = getattr(tokenizer, 'video_token_id', 151656)
        self.spatial_merge_size = 2
        self.tokens_per_second = 2





    def postprocess_output_tp(self, output: DataProto, image_data: List[List[Image.Image]], step: int=2):
        '''output: cpu'''
        # init loop responses token
        if self.loop_cnt == 0:
            self.batch_size = output.batch.batch_size[0]
            self.tool_use = [[] for _ in range(self.batch_size)]
            self.loop_responses_token = [[] for _ in range(self.batch_size)]
            self.end_flags = [False for _ in range(self.batch_size)]
            self.init_prompt_token = output.batch.get('prompts')
            prompt_length = self.init_prompt_token.shape[-1]
            self.init_attention_mask = output.batch.get('attention_mask')[:,:prompt_length]  

            batch_idxs = list(range(self.batch_size))
            for idx in range(self.batch_size):
                prompt_token = self.init_prompt_token[idx]
                # prompt_token_list = torch.tensor(prompt_token)[torch.tensor(prompt_token) != self.pad_token_id].tolist()
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
            responses=responses_str, tokenizer=self.tokenizer, image_data=image_data
        )
        
        #if not use_process_reward will be 0
        if self.env_object.use_process_reward:
            step_scores = self.env_object.get_step_reward(responses=responses_str)
        else:
            step_scores = [0] * len(responses_str)
        
        # encode infos for next prompt
        info_tokens = self.tokenizer(infos_str).input_ids
        next_prompt_token = []
        next_prompt_length = []
        next_sample_idx = []
        for idx, batch_idx in enumerate(batch_idxs):
            # 只在第一次未结束时添加response_token_list
            if not self.end_flags[batch_idx]:
                response_token = responses[idx]
                response_token_list = response_token[response_token != self.pad_token_id].tolist()
                self.loop_responses_token[batch_idx].append(response_token_list)
                # get process reward
                self.tool_use[batch_idx].append(step_scores[idx])

            # 如果done了，设置end_flag
            if dones[idx]:
                self.end_flags[batch_idx] = True

            # info_token_list只在未done时添加
            if not dones[idx] and not self.end_flags[batch_idx]:
                info_token_list = info_tokens[idx]
                self.loop_responses_token[batch_idx].append(info_token_list)

            next_sample_idx.append(batch_idx)
            promt_token = list(itertools.chain.from_iterable(self.loop_responses_token[batch_idx]))
            next_prompt_token.append(promt_token)
            next_prompt_length.append(len(promt_token))
        
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
                'input_ids': next_input_ids[:, -max_len:].cpu().share_memory_(),
                'position_ids': position_ids[:, -max_len:].cpu().share_memory_(),
                'attention_mask': next_attention_mask[:, -max_len:].to(dtype=torch.int64).cpu().share_memory_()
            },
            batch_size=next_input_ids.shape[0]
        ).share_memory_()
        raw_prompt_ids = np.empty(len(next_prompt_token), dtype=object)
        # raw_prompt_ids[:] = [np.array(x[-max_len:]) for x in next_prompt_token]
        raw_prompt_ids[:] = [x[-max_len:] for x in next_prompt_token]

        next_data = DataProto(batch=next_batch, non_tensor_batch={'raw_prompt_ids': raw_prompt_ids})
        next_data.meta_info.update(self.meta_info)
        next_data.meta_info['index'] = next_sample_idx
        next_data.meta_info['do_sample'] = False # step > 0 does not do sample
        self.loop_cnt += 1

        return next_data
    
    def postprocess_output(self, output: DataProto, step: int = 2):
        """output: cpu"""
        print(f"[ToolUtils] start the {step}th postprocess", file=sys.stderr, flush=True)

        # 初始化或获取批次索引
        if self.loop_cnt == 0:
            self._initialize_first_loop(output)
            batch_idxs = list(range(self.batch_size))
        else:
            batch_idxs = output.meta_info['index']

        # 处理响应
        responses = output.batch.get('responses')
        process_response = [r[r != self.pad_token_id].tolist() for r in responses]
        
        # 更新循环响应token
        for idx, batch_idx in enumerate(batch_idxs):
            self.loop_responses_token[batch_idx].append(process_response[idx])
            self.loop_raw_responses_token[batch_idx].append(process_response[idx])

        # 执行环境步骤
        responses_str = self.tokenizer.batch_decode(process_response, skip_special_tokens=False)
        infos_str, dones, _, _, new_image_data, raw_prompt, multi_modal_data, valid_tool = self.env_object.step(
            responses=responses_str, tokenizer=self.tokenizer, image_data=self.image_list, processor=self.processor
        )
        
        # 更新多模态数据
        for idx, batch_idx in enumerate(batch_idxs):
            if multi_modal_data[idx] is not None:
                self.multi_modal_inputs[batch_idx].append(multi_modal_data[idx])

        # 处理未完成的样本
        next_data_info = self._process_unfinished_samples(batch_idxs, dones, infos_str, raw_prompt, new_image_data, valid_tool)
        
        if not next_data_info['sample_idx']:
            return None
            
        # 创建下一批数据
        return self._create_next_batch(next_data_info)

    def _initialize_first_loop(self, output):
        """初始化第一次循环的变量"""
        self.batch_size = output.batch.batch_size[0]
        self.init_prompt_token = output.batch.get('prompts')
        self.raw_prompt_id = output.non_tensor_batch.get('raw_prompt_ids')
        
        prompts_unpadded = [p[p != self.pad_token_id].tolist() for p in self.init_prompt_token]
        self.loop_responses_token = [[p] for p in prompts_unpadded]
        self.loop_raw_responses_token = [[r] for r in self.raw_prompt_id]
        self.multi_modal_inputs = [[d] for d in output.non_tensor_batch["multi_modal_inputs"]]
        self.image_list = [i["image"] for i in output.non_tensor_batch["multi_modal_data"]]
        self.tool_use = [[] for _ in range(self.batch_size)]
        
        prompt_length = self.init_prompt_token.shape[-1]
        self.init_attention_mask = output.batch.get('attention_mask')[:, :prompt_length]

    def _tokenize_infos(self, infos_str):
        """tokenize信息字符串"""
        if not infos_str:
            return []
        try:
            return self.tokenizer(text=infos_str).input_ids
        except:
            raise ValueError("tokenization error")

    def _process_unfinished_samples(self, batch_idxs, dones, infos_str, raw_prompt, new_image_data, valid_tool):
        """处理未完成的样本"""
        next_info = {
            'sample_idx': [],
            'prompt_token': [],
            'raw_prompt_token': [],
            'image_data': []
        }
        
        for idx, batch_idx in enumerate(batch_idxs):
            if not dones[idx]:
                # tokenize信息和原始提示
                info_tokens = self._tokenize_infos(infos_str[idx])
                raw_prompt_tokens = self._tokenize_infos(raw_prompt[idx])
                
                # 更新token列表
                self.loop_responses_token[batch_idx].append(info_tokens)
                self.loop_raw_responses_token[batch_idx].append(raw_prompt_tokens)
                
                # 准备下一个样本数据
                next_info['sample_idx'].append(batch_idx)
                next_info['prompt_token'].append(list(itertools.chain.from_iterable(self.loop_responses_token[batch_idx])))
                next_info['raw_prompt_token'].append(list(itertools.chain.from_iterable(self.loop_raw_responses_token[batch_idx])))
                
                self.tool_use[batch_idx].append(0)  # step_scores固定为0
                
                # 更新图像数据
                if valid_tool[idx]:
                    self.image_list[batch_idx].append(new_image_data[idx])
                next_info['image_data'].append(self.image_list[batch_idx])
        
        return next_info

    def _create_next_batch(self, next_info):
        """创建下一批数据"""
        prompt_lengths = [len(tokens) for tokens in next_info['prompt_token']]
        max_len = max(prompt_lengths) + self.config_prompt_length
        
        # 左填充
        next_prompt_pad = [
            [self.pad_token_id] * (max_len - len(tokens)) + tokens
            for tokens in next_info['prompt_token']
        ]
        
        # 创建tensor
        input_ids = torch.tensor(next_prompt_pad, dtype=torch.int64)
        attention_mask = input_ids != self.pad_token_id
        position_ids = torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0, max=None) * attention_mask
        
        # 创建批次
        next_batch = TensorDict({
            'input_ids': input_ids[:, -max_len:].cpu().share_memory_(),
            'position_ids': position_ids[:, -max_len:].cpu().share_memory_(),
            'attention_mask': attention_mask[:, -max_len:].to(dtype=torch.int64).cpu().share_memory_()
        }, batch_size=input_ids.shape[0]).share_memory_()
        
        # 准备原始提示ID和图像数据
        raw_prompt_ids = np.empty(len(next_info['raw_prompt_token']), dtype=object)
        raw_prompt_ids[:] = [x[-max_len:] for x in next_info['raw_prompt_token']]
        
        image_data = np.array([{"image": img} for img in next_info['image_data']], dtype=object)
        
        # 创建数据协议
        next_data = DataProto(
            batch=next_batch, 
            non_tensor_batch={'raw_prompt_ids': raw_prompt_ids, 'multi_modal_data': image_data}
        )
        next_data.meta_info.update(self.meta_info)
        next_data.meta_info['index'] = next_info['sample_idx']
        next_data.meta_info['do_sample'] = False
        
        self.loop_cnt += 1
        
        # 验证图像token一致性
        self._validate_image_consistency(next_info['sample_idx'], raw_prompt_ids, image_data)
        
        return next_data

    def _validate_image_consistency(self, sample_idxs, raw_prompt_ids, image_data):
        """验证图像token的一致性"""
        try:
            for i, (sample_idx, images) in enumerate(zip(sample_idxs, image_data)):
                full_prompt_tokens = raw_prompt_ids[i]
                num_image_tokens = full_prompt_tokens.count(self.image_token_id)
                num_images = len(images["image"])
                if num_image_tokens != num_images:
                    print(
                        f"Final consistency check failed for sample {sample_idx}. "
                        f"Found {num_image_tokens} image tokens but {num_images} images.",
                        flush=True, file=sys.stderr
                    )
                    raise ValueError("Image token consistency error")
        except Exception:
            breakpoint()

    def compose_final_output(self, step) -> DataProto:
        """Compose final generation output."""
        print(f"[compose_final_output] start compose the final output, step is {step}", flush=True, file=sys.stderr)
        
        # 构建响应数据
        response_data = self._build_response_data()
        max_len = self._get_distributed_max_length(response_data['lengths'])
        
        # 填充和创建tensor
        response_token, response_loss_mask = self._pad_and_create_tensors(response_data, max_len)
        response_attention_mask = (response_token != self.pad_token_id).long()
        
        # 处理工具使用分数
        tool_use_score = self._create_tool_use_tensor()
        
        # 创建完整的输入数据
        input_ids, attention_mask, position_ids = self._create_full_input_data(
            response_token, response_attention_mask
        )
        
        # 创建损失掩码
        loss_mask = torch.cat([
            torch.zeros_like(self.init_attention_mask, dtype=torch.float32), 
            response_loss_mask
        ], dim=-1)
        
        # 创建最终输出
        return self._create_final_data_proto(
            input_ids, attention_mask, position_ids, loss_mask, 
            response_token, tool_use_score
        )

    def _build_response_data(self):
        """构建响应数据和损失掩码"""
        input_ids_list = []
        loss_mask_list = []
        length_list = []
        
        for responses in self.loop_responses_token:
            # 跳过初始prompt，只处理响应
            response_tokens = list(itertools.chain.from_iterable(responses[1:]))
            
            # 构建损失掩码：奇数轮次(用户响应)为1，偶数轮次(工具响应)为0
            loss_mask = []
            for turn_idx, response_part in enumerate(responses[1:]):
                mask_value = (turn_idx + 1) % 2  # 1 for odd turns, 0 for even turns
                loss_mask.extend([mask_value] * len(response_part))
            
            input_ids_list.append(response_tokens)
            loss_mask_list.append(loss_mask)
            length_list.append(len(response_tokens))
        
        return {
            'input_ids': input_ids_list,
            'loss_masks': loss_mask_list,
            'lengths': length_list
        }

    def _get_distributed_max_length(self, lengths):
        """获取分布式环境下的最大长度"""
        max_response_length = torch.tensor([max(lengths)])
        dist.all_reduce(max_response_length, op=dist.ReduceOp.MAX)
        return int(max_response_length)

    def _pad_and_create_tensors(self, response_data, max_len):
        """填充序列并创建tensor"""
        # 右填充
        padded_input_ids = []
        padded_loss_masks = []
        
        for input_ids, loss_mask in zip(response_data['input_ids'], response_data['loss_masks']):
            # 填充input_ids
            padded_ids = input_ids + [self.pad_token_id] * (max_len - len(input_ids))
            padded_input_ids.append(padded_ids[:max_len])
            
            # 填充loss_mask
            padded_mask = loss_mask + [0] * (max_len - len(loss_mask))
            padded_loss_masks.append(padded_mask[:max_len])
        
        response_token = torch.tensor(padded_input_ids, dtype=torch.int64)
        response_loss_mask = torch.tensor(padded_loss_masks, dtype=torch.float32)
        
        return response_token, response_loss_mask

    def _create_tool_use_tensor(self):
        """创建工具使用分数tensor"""
        max_tool_use_len = max(self.max_turns, max(len(tool_use) for tool_use in self.tool_use))
        
        tool_use_tensor = []
        for tool_use_item in self.tool_use:
            if not tool_use_item:
                padded_tool_use = [torch.nan] * max_tool_use_len
            else:
                padding_len = max_tool_use_len - len(tool_use_item)
                padded_tool_use = tool_use_item + [torch.nan] * padding_len
            tool_use_tensor.append(padded_tool_use)
        
        return torch.tensor(tool_use_tensor)

    def _create_full_input_data(self, response_token, response_attention_mask):
        """创建完整的输入数据"""
        input_ids = torch.cat([self.init_prompt_token, response_token], dim=-1)
        attention_mask = torch.cat([self.init_attention_mask, response_attention_mask], dim=-1)
        
        # 处理position_ids
        if self._is_qwen_processor():
            position_ids = self._create_qwen_position_ids(input_ids, attention_mask)
        else:
            position_ids = torch.clip(
                torch.cumsum(attention_mask, dim=-1) - 1, min=0, max=None
            ) * attention_mask
        
        return input_ids, attention_mask, position_ids

    def _is_qwen_processor(self):
        """检查是否为Qwen处理器"""
        return (self.processor is not None and 
                hasattr(self.processor, 'image_processor') and
                getattr(self.processor.image_processor, '_processor_class', None) == "Qwen2_5_VLProcessor")

    def _create_qwen_position_ids(self, input_ids, attention_mask):
        """为Qwen模型创建position_ids"""
        from verl.models.transformers.qwen2_vl import get_rope_index
        
        multi_modal_inputs = self.merge_multi_modal_inputs(self.multi_modal_inputs)
        position_ids = []
        
        for idx, input_id in enumerate(input_ids):
            position_id = get_rope_index(
                self.processor,
                input_ids=input_id,
                image_grid_thw=multi_modal_inputs[idx][0].get("image_grid_thw"),
                video_grid_thw=multi_modal_inputs[idx][0].get("video_grid_thw"),
                attention_mask=attention_mask[idx],
            )
            position_ids.append(position_id)
        
        return torch.stack(position_ids, dim=0)

    def _create_final_data_proto(self, input_ids, attention_mask, position_ids, 
                               loss_mask, response_token, tool_use_score):
        """创建最终的DataProto对象"""
        final_batch = TensorDict({
            'prompts': self.init_prompt_token,
            'responses': response_token,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'loss_mask': loss_mask,
            'tool_use_scores': tool_use_score
        }, batch_size=self.batch_size)
        
        # 准备图像和多模态数据
        image_list = np.array([{"image": img} for img in self.image_list], dtype=object)
        multi_modal_inputs = self.merge_multi_modal_inputs(self.multi_modal_inputs)
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