import torch
import itertools
import numpy as np
from verl import DataProto
import torch.distributed as dist
from tensordict import TensorDict


class ToolUtils:
    def __init__(self, tokenizer, meta_info, config, env_object):
        self.tokenizer = tokenizer  
        self.final_str = config.stop[-1] if config.stop else ''
        self.config_prompt_length = config.prompt_length
        self.config_response_length = config.response_length

        self.max_prompt_length = config.prompt_length

        self.pad_token_id = meta_info.get('pad_token_id')
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
        for idx, batch_idx in enumerate(batch_idxs):
            response_token = responses[idx]
            response_token_list = response_token[response_token != self.pad_token_id].tolist()
            self.loop_responses_token[batch_idx].append(response_token_list)

        # decode responses for env step (detect tool call)
        responses_str = self.tokenizer.batch_decode(
            output.batch.get('responses'),
            skip_special_tokens=False,
        )
        responses_str = [response.replace(self.tokenizer.pad_token, '') for response in responses_str]
        infos_str, dones, _, _ = self.env_object.step(
            responses=responses_str, tokenizer=self.tokenizer
        )

        # encode infos for next prompt TODO: can tokenize be faster?
        info_tokens = self.tokenizer(infos_str).input_ids
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

    def compose_final_output(self, step) -> DataProto:
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
        dist.all_reduce(max_response_length, op=dist.ReduceOp.MAX)
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
                'loss_mask': loss_mask
            },
            batch_size=self.batch_size,
        )  

        final_output = DataProto(batch=final_batch)
        return final_output
