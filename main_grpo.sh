#!/bin/bash
# GRPO Training Script for RL-Factory
# This script runs Group Relative Policy Optimization (GRPO) training 
# for reinforcement learning agents with tool-calling capabilities.

set -e -x

export MODEL_PATH=/your/path/to/huggingface.co/Qwen/Qwen3-4B
export REWARD_MODEL_PATH=/your/path/to/huggingface.co/Qwen/QwQ-32B
export RESULT_DIR=/your/path/to/results/rl_factory/your_result_dir

python3 -m verl.trainer.main_ppo --config-name=rl_factory_ppo_trainer \
    algorithm.adv_estimator=grpo\
    data.train_files=data/nq_search/train.parquet\
    data.val_files=data/nq_search/test.parquet\
    data.train_batch_size=128\
    data.max_prompt_length=4096\
    data.max_response_length=512\
    actor_rollout_ref.model.path=$MODEL_PATH\
    actor_rollout_ref.model.use_remove_padding=True\
    actor_rollout_ref.model.enable_gradient_checkpointing=True\
    actor_rollout_ref.actor.optim.lr=1e-6\
    actor_rollout_ref.actor.ppo_mini_batch_size=32\
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16\
    actor_rollout_ref.actor.use_kl_loss=True\
    actor_rollout_ref.actor.kl_loss_coef=0.001\
    actor_rollout_ref.actor.kl_loss_type=low_var_kl\
    actor_rollout_ref.actor.fsdp_config.param_offload=True\
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True\
    actor_rollout_ref.actor.state_masking=True\
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16\
    actor_rollout_ref.rollout.tensor_model_parallel_size=1\
    actor_rollout_ref.rollout.name=vllm\
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75\
    actor_rollout_ref.rollout.n=4\
    actor_rollout_ref.rollout.max_turns=2\
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16\
    actor_rollout_ref.ref.fsdp_config.param_offload=False\
    actor_rollout_ref.rollout.enforce_eager=False\
    actor_rollout_ref.rollout.free_cache_engine=True\
    actor_rollout_ref.env.name=search\
    actor_rollout_ref.env.mcp_mode=stdio\
    actor_rollout_ref.env.tool_manager=qwen3\
    actor_rollout_ref.env.enable_thinking=False\
    actor_rollout_ref.env.config_path=envs/configs/mcp_tools.pydata\
    actor_rollout_ref.env.use_process_reward=False\
    reward_rollout.if_use_reward_rollout=False\
    reward_rollout.rollout.tensor_model_parallel_size=4\
    reward_rollout.rollout.gpu_memory_utilization=0.65\
    reward_rollout.rollout.model_name=$REWARD_MODEL_PATH\
    reward_rollout.rollout.free_cache_engine=True\
    reward_rollout.rollout.response_length=2048\
    reward_model.reward_manager=parallel\
    algorithm.kl_ctrl.kl_coef=0.001\
    trainer.critic_warmup=0\
    trainer.logger=['tensorboard']\
    trainer.project_name='GRPO_search'\
    trainer.experiment_name='search_with_thinking'\
    trainer.n_gpus_per_node=8\
    trainer.nnodes=1\
    trainer.val_before_train=False\
    trainer.default_local_dir=$RESULT_DIR\
    trainer.default_hdfs_dir=null\
    trainer.save_freq=20\
    trainer.test_freq=10\
    trainer.total_epochs=5 $@ 2>&1 | tee grpo.log