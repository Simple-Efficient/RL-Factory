set -e -x
FILE="$(pwd)/verl/utils/reward_score/search.py"
FUNCTION_NAME="compute_score"

export MODEL_PATH='your/path/to/Qwen/Qwen3-8B'
export REWARD_MODEL_PATH=/your/path/to/huggingface.co/Qwen/QwQ-32B
export TEST_DATA='your/path/to/data/hotpot/test.parquet'
# export VLLM_ATTENTION_BACKEND=XFORMERS

python3 -m verl.trainer.main_evaluate\
    data.val_files=$TEST_DATA\
    data.val_batch_size=2048\
    data.max_prompt_length=4096\
    data.max_response_length=512\
    actor_rollout_ref.model.path=$MODEL_PATH\
    actor_rollout_ref.model.use_remove_padding=True\
    actor_rollout_ref.model.enable_gradient_checkpointing=True\
    actor_rollout_ref.actor.ppo_mini_batch_size=256\
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32\
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32\
    actor_rollout_ref.rollout.tensor_model_parallel_size=1\
    actor_rollout_ref.rollout.name=vllm\
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9\
    actor_rollout_ref.rollout.max_turns=2\
    actor_rollout_ref.rollout.val_kwargs.temperature=0\
    actor_rollout_ref.rollout.val_kwargs.top_k=-1\
    actor_rollout_ref.rollout.val_kwargs.top_p=1\
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32\
    actor_rollout_ref.env.name=search\
    actor_rollout_ref.env.mcp_mode=stdio\
    actor_rollout_ref.env.tool_manager=null\
    actor_rollout_ref.env.enable_thinking=False\
    actor_rollout_ref.env.config_path=envs/configs/mcp_tools.pydata\
    reward_rollout.if_use_reward_rollout=False\
    reward_rollout.rollout.tensor_model_parallel_size=4\
    reward_rollout.rollout.gpu_memory_utilization=0.75\
    reward_rollout.rollout.model_name=$REWARD_MODEL_PATH\
    reward_rollout.rollout.free_cache_engine=False\
    reward_rollout.rollout.response_length=2048\
    reward_model.reward_manager=parallel\
    trainer.logger=['tensorboard']\
    trainer.project_name='GRPO_search'\
    trainer.experiment_name='search_with_thinking'\
    trainer.n_gpus_per_node=8\
    trainer.nnodes=1\
    trainer.val_only=True\
    trainer.default_local_dir=ckpt\
    trainer.default_hdfs_dir=null $@ 2>&1 | tee grpo.log
