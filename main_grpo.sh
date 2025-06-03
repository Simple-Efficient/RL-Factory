#!/bin/bash
source /root/autodl-tmp/RL-Factory/tmp/env_autodl.sh
ray stop --force
sleep 5
export MODEL_PATH=/root/autodl-tmp/models/Qwen/Qwen3-0.6B
export REWARD_MODEL_PATH=/root/autodl-tmp/models/Qwen/Qwen3-0.6B
export WANDB_API_KEY=76ecf2334073036f76da7b9e4eb5bbe934767728
export HYDRA_FULL_ERROR=1
export RAY_DEBUG=1
# export RAY_DEBUG="legacy"
# export VLLM_ATTENTION_BACKEND=XFORMERS
DATA=/root/autodl-tmp/data/nq_hotpotqa_train



python3 -m verl.trainer.main_ppo\
    algorithm.adv_estimator=grpo\
    data.train_files=$DATA/train.parquet\
    data.val_files=$DATA/test.parquet\
    data.train_batch_size=32\
    data.max_prompt_length=1024\
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
    actor_rollout_ref.actor.fsdp_config.param_offload=False\
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False\
    actor_rollout_ref.actor.state_masking=True\
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16\
    actor_rollout_ref.rollout.tensor_model_parallel_size=1\
    actor_rollout_ref.rollout.name=vllm\
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4\
    actor_rollout_ref.rollout.n=4\
    actor_rollout_ref.rollout.max_turns=2\
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16\
    actor_rollout_ref.ref.fsdp_config.param_offload=False\
    actor_rollout_ref.rollout.enforce_eager=False\
    actor_rollout_ref.rollout.free_cache_engine=False\
    actor_rollout_ref.env.name=search\
    actor_rollout_ref.env.mcp_mode=stdio\
    actor_rollout_ref.env.tool_manager=qwen3\
    actor_rollout_ref.env.enable_thinking=False\
    actor_rollout_ref.env.config_path=/your/path/to/envs/configs/mcp_tools.pydata\
    reward_rollout.if_use_reward_rollout=False\
    reward_rollout.rollout.tensor_model_parallel_size=1\
    reward_rollout.rollout.gpu_memory_utilization=0.4\
    reward_rollout.rollout.model_name=$REWARD_MODEL_PATH\
    reward_rollout.rollout.free_cache_engine=False\
    reward_rollout.rollout.response_length=512\
    reward_model.reward_manager=parallel\
    algorithm.kl_ctrl.kl_coef=0.001\
    trainer.critic_warmup=0\
    trainer.logger=['tensorboard']\
    trainer.project_name='GRPO_search'\
    trainer.experiment_name='search_with_thinking'\
    trainer.n_gpus_per_node=1\
    trainer.nnodes=1\
    trainer.val_before_train=False\
    trainer.default_local_dir=ckpt\
    trainer.default_hdfs_dir=null\
    trainer.save_freq=20\
    trainer.test_freq=10\
    trainer.total_epochs=5 $@ 2>&1 | tee grpo.log
