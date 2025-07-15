# Minimal Implementation Tutorial

+ This document uses the reproduction of [Search-R1](https://github.com/PeterGriffinJin/Search-R1) as an example to illustrate how to use our RLFactory to implement your own Multi-turn tool-use RL post-training. In general, you only need to complete "**Tool Definition**" and "**Reward Function Definition**" to start training!

## Step 1 - Tool Definition
+ Based on [qwen_agent](https://github.com/QwenLM/Qwen-Agent), custom tools (inheriting from `BaseTool`), MCP toolsets, and built-in tools from [`qwen_agent`](https://github.com/QwenLM/Qwen-Agent) (such as `code_interpreter`) are supported. It is recommended to define your tools under the `tools` folder to keep the project organized. We provide a search example, which refers to [Search-R1](https://github.com/PeterGriffinJin/Search-R1):

+ **Prepare the RAG Server** - Referencing [Search-R1](https://github.com/PeterGriffinJin/Search-R1), it builds an offline corpus, sets up a wiki with RAG, and exposes a search interface.
    + **Download the corpus**
        ```bash
        save_path=/the/path/to/save
        python rag_server/download.py --save_path $save_path
        cat $save_path/part_* > $save_path/e5_Flat.index
        gzip -d $save_path/wiki-18.jsonl.gz
        ```
    + **Process the dataset**
        ```bash
        python rag_server/data_process/nq_search.py
        # or
        tar -zxvf rag_server/nq_search.tar.gz
        mv nq_search/ data/
        ```
    + **Run the RAG Server:** Before running the bash script, modify the parameters in `launch.sh` (`file_path` is the storage location of the corpus, `retriever` is the local path of the model intfloat/e5-base-v2)
        ```bash
        bash rag_server/launch.sh
        ```
+ **Prepare MCP Startup File:** Implement MCP-format registration for the search tool in [`envs/tools/search.py`](../../../envs/tools/search.py)
    ```python
    from mcp.server.fastmcp import FastMCP  # Assuming you have this base library
    
    mcp = FastMCP("LocalServer")
    
    @mcp.tool()
    def query_rag(query: str, topk: int = 3) -> str:
        ...
    ```
+ **Prepare MCP Configuration File:** Create a new MCP configuration file [`mcp_tools.pydata`](../../../envs/configs/mcp_tools.pydata) in `envs/configs`
    ```json
    [
        {'mcpServers': {
            'search': {
                'command': 'python',
                'args': ['envs/tools/search.py']
            }
        }}
    ]
    ```
    In [`main_grpo.sh`](../../../main_grpo.sh), set the parameter `actor_rollout_ref.env.config_path` to the path of this config file.

## Step 2 - Reward Function Definition

+ The main logic of the reward function refers to [search-r1](https://github.com/PeterGriffinJin/Search-R1) - `verl/utils/reward_score/qa_em.py`
  + **Create** a new environment file [`envs/search.py`](../../../envs/search.py)
    ```python
    import re
    import string
    import random
    from .base import Env

    class SearchEnv(Env):
        def __init__(self, config):
            super().__init__(config)
            self.use_verify_tool = False

        # NOTE: Add your reward calculation rules here!
        def _compute_score_with_rules(self, data, tokenizer):
            ...
    ```
  + **Register** this environment file as `search` in [`envs/__init__.py`](../../../envs/__init__.py)
    ```python
    from .base import Env as BaseEnv
    from .search import SearchEnv

    __all__ = ['BaseEnv', 'SearchEnv']

    TOOL_ENV_REGISTRY = {
        'base': BaseEnv,
        'search': SearchEnv,
    }
    ```
+ **Define the reward function based on rules**
    ```python
    def _compute_score_with_rules(self, data, tokenizer):
        def normalize_answer(s):
            ...

        def em_check(prediction, golden_answers):
            ...

        def extract_solution(solution_str):
            """Extract the equation from the solution string."""
            ...

        def compute_score_em(solution_str, ground_truth, method='strict', format_score=0.0, score=1.):
            """The scoring function for exact match (EM)."""
            ...
        
        scores = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            
            # process the data_item to the token and decode them
            processed_data = self._process_data(data_item=data_item, tokenizer=tokenizer)
            ground_truth, response_str = processed_data['ground_truth'], processed_data['response_str']

            # reserved for compatibility
            prompt_str, data_source, extra_info = processed_data['prompt_str'], processed_data['data_source'], processed_data['extra_info']

            score = compute_score_em(response_str, ground_truth)
            scores.append([score])

        return scores
    ```
    The key part of the above program is the definition of `compute_score_em(...)`, other processing procedures can remain unchanged.

## Step 3 - Start Training!

+ Modify the script file [`main_grpo.sh`](../../../main_grpo.sh)
    ```bash
    set -e -x

    export MODEL_PATH=/the/path/to/model
    export REWARD_MODEL_PATH=/the/path/to/reward_rollout_model

    python3 -m verl.trainer.main_ppo\
    ...
    actor_rollout_ref.env.name=search\
    actor_rollout_ref.env.tool_manager=qwen3\
    actor_rollout_ref.env.enable_thinking=True\
    actor_rollout_ref.env.config_path=/the/path/to/mcp_tools.pydata\
    ...
    ```
+ Run the training program
    ```bash
    bash main_grpo.sh
    ```

## After Training - View Experiment Results

+ Use Tensorboard or other verl-supported methods to view experiment curves
    ```bash
    tensorboard --logdir=./
    ```

## After Training - Start Evaluate！ 

+ Simply adjust main_eval.sh based on main_grpo.sh
    ```bash
    set -e -x

    export MODEL_PATH=/the/path/to/model
    export REWARD_MODEL_PATH=/the/path/to/reward_rollout_model

    python3 -m verl.trainer.main_evaluate\
    ...
    data.val_files=path/to/evaluation.parquet\
    data.val_batch_size=1024\
    actor_rollout_ref.rollout.val_kwargs.temperature=0\
    actor_rollout_ref.rollout.val_kwargs.top_k=-1\
    actor_rollout_ref.rollout.val_kwargs.top_p=1\
    trainer.val_only=True\
    trainer.default_local_dir=path_for_results\
    ...
    ```
