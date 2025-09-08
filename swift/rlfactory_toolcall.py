import asyncio
import json
import os, sys
import swift
from swift.plugin import multi_turns
from swift.plugin.multi_turn import MultiTurnScheduler
from swift.utils import get_logger
from envs import TOOL_ENV_REGISTRY

import nest_asyncio
nest_asyncio.apply()
class ENV(object):
    def __init__(self, config):
        self.cfg = config
        for k,v in config.items():
           setattr(self, k, v)
    def get(self, key, default_value=None):
        if key not in self.cfg:
            return default_value
        return self.cfg[key]

env_dict = {
    "name": "search",
    "mcp_mode": "stdio",
    "tool_manager": "qwen3",
    "enable_thinking": True,
    "config_path": "./envs/configs/mcp_tools.pydata",
    "use_process_reward": False,
    "model_type": "qwen3",
    "tool_name_selected": [],      # selected tools for using when call a sse MCP server; default [] will call all tools
    "max_prompt_length": 4096,
    "enable_limiter": False,
    "tool_timeout": 10,            # this is the timeout for each tool call
    "max_concurrency": 10,         # this is the max concurrency for tool manager
    "use_storage_manager": False,
    "max_size": 1000,
    "cache": "single",
    "persist": "false",
    "eviction": "lru",
    "persist_dir": "./cache",
    "local_cache": "null",
    "asyncio_run": False,
    "parallel_sse_tool_call": ENV({
      "is_enabled": True,
      "num_instances": 3
    })
}
config_path = env_dict.get("config_path")
if not os.path.exists(config_path):
    print(f"Config file for toolcall not found: {config_path}")
    sys.exit(1)

# read from /workdir/toolcall_env.json or /tmp/toolcall_env.json
env_dict_2 = {}
locations = ["./swift/toolcall_env.json"]
for location in locations:
    if os.path.exists(location):
        with open(location, 'r') as f:
            env_dict_2 = json.load(f)
            break

# update env_dict with env_dict_2
for k, v in env_dict_2.items():
    env_dict[k] = v

print("toolcall env: ", env_dict)

from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union
if TYPE_CHECKING:
    from swift.llm.infer.protocol import RolloutResponseChoice
    from swift.llm.template import RolloutInferRequest

class RLFactoryToolcall(MultiTurnScheduler):
    def __init__(self, infer_engine=None, max_turns=None, *args, **kwargs):
        print('--start tool init--')
        super().__init__(infer_engine, max_turns, *args, **kwargs)

        self.args = args
        self.kwargs = kwargs
        env = ENV(env_dict)
        self.env_object = TOOL_ENV_REGISTRY[env.name](
            config=env, 
            centralized_actor=False
        )
        print('--end tool init--')

    def check_finished(self, infer_request: 'RolloutInferRequest', result: 'RolloutResponseChoice',
                       current_turn: int) -> bool:

        action, tool = self.env_object.tool_manager.parse_response(infer_request.messages[-1]["content"])
        if action == 'answer':
            # If action is 'answer', end the conversation
            return True

        #length too long or reaches max turn
        return super().check_finished(infer_request, result, current_turn)

    def step(
        self,
        infer_request: 'RolloutInferRequest',
        result: 'RolloutResponseChoice',
        current_turn: int,
    ):
        extra_dict = {}

        # Last request
        msg = infer_request.messages[-1]

        step_score = self.env_object.get_step_reward([infer_request.messages[-1]['content']])
        #if '<tool_call>' in msg['content']:
        #print("message: ", msg)
        if msg['role'] == 'assistant':
            res = self.env_object.step([msg['content']])
            #print("res: ", res)
            #print("")
            if asyncio.iscoroutine(res):
                res = asyncio.get_event_loop().run_until_complete(res)

            next_obs, dones, valid_action, is_tool = res
            infer_request.messages.append(next_obs[-1][-1])

            step_scores_full = self.env_object.get_step_reward(responses=[d['content'] for d in infer_request.messages])
        else:
            print("the last message is not assistant")

        extra_dict["rollout_reward"] = step_score
        if swift.version.__version__ <= '3.8.0':
           return infer_request, extra_dict
        else:
            return {'infer_request': infer_request, 'rollout_infos': extra_dict}
        # for swift 3.7
        # return infer_request, extra_dict

multi_turns['rlfactory_toolcall'] = RLFactoryToolcall


if __name__ == "__main__":
    rlf = RLFactoryToolcall()
    rlf.env_object.step(['message <tool_call>{"name": "query_rag", "arguments": {"query": "who is the 14th president of nigeria"}}</tool_call>'])