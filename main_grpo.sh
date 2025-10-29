#!/bin/bash
# ============================================================================
# GRPO Training Script for RL-Factory
# ============================================================================
# This script runs Group Relative Policy Optimization (GRPO) training for
# reinforcement learning agents. It configures the model, data, and training
# parameters for agentic learning with tool-calling capabilities.
#
# Key Features:
# - GRPO algorithm for efficient RL training
# - Support for multi-turn tool calling
# - Integration with MCP (Model Context Protocol) tools
# - FSDP-based distributed training with parameter/optimizer offloading
# ============================================================================

set -e -x

# ============================================================================
# Environment Variables - MODIFY THESE PATHS
# ============================================================================
# Path to the base model (e.g., Qwen3-4B, Qwen3-8B)
export MODEL_PATH=/your/path/to/huggingface.co/Qwen/Qwen3-4B

# Path to the reward model for judging (e.g., QwQ-32B for reasoning)
export REWARD_MODEL_PATH=/your/path/to/huggingface.co/Qwen/QwQ-32B

# Directory to save training results, checkpoints, and logs
export RESULT_DIR=/your/path/to/results/rl_factory/your_result_dir

# ============================================================================
# Training Command - GRPO Configuration
# ============================================================================
python3 -m verl.trainer.main_ppo --config-name=rl_factory_ppo_trainer \
    # Algorithm Configuration
    algorithm.adv_estimator=grpo\
    # Data Configuration
    data.train_files=data/nq_search/train.parquet\
    data.val_files=data/nq_search/test.parquet\
    data.train_batch_size=128\
    data.max_prompt_length=4096\          # Maximum length for input prompts
    data.max_response_length=512\         # Maximum length for model responses
    # Model Configuration
    actor_rollout_ref.model.path=$MODEL_PATH\
    actor_rollout_ref.model.use_remove_padding=True\           # Remove padding for efficiency
    actor_rollout_ref.model.enable_gradient_checkpointing=True\ # Save memory during training
    # Actor Training Configuration
    actor_rollout_ref.actor.optim.lr=1e-6\                     # Learning rate for actor optimization
    actor_rollout_ref.actor.ppo_mini_batch_size=32\            # Mini-batch size for PPO updates
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16\   # Micro-batch size per GPU
    actor_rollout_ref.actor.use_kl_loss=True\                  # Enable KL divergence loss
    actor_rollout_ref.actor.kl_loss_coef=0.001\                # Coefficient for KL loss
    actor_rollout_ref.actor.kl_loss_type=low_var_kl\           # Type of KL loss computation
    # FSDP Configuration for Memory Efficiency
    actor_rollout_ref.actor.fsdp_config.param_offload=True\    # Offload parameters to CPU
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True\ # Offload optimizer states to CPU
    actor_rollout_ref.actor.state_masking=True\                # Enable state masking
    # Rollout Configuration (vLLM-based generation)
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16\
    actor_rollout_ref.rollout.tensor_model_parallel_size=1\     # Tensor parallelism size
    actor_rollout_ref.rollout.name=vllm\                        # Use vLLM for efficient inference
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75\      # GPU memory utilization ratio
    actor_rollout_ref.rollout.n=4\                              # Number of samples per prompt
    actor_rollout_ref.rollout.max_turns=2\                      # Maximum turns for multi-turn dialogue
    # Reference Model Configuration
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16\
    actor_rollout_ref.ref.fsdp_config.param_offload=False\      # Keep reference model in GPU
    # vLLM Engine Configuration
    actor_rollout_ref.rollout.enforce_eager=False\              # Use CUDA graph for efficiency
    actor_rollout_ref.rollout.free_cache_engine=True\           # Free cache between iterations
    # Environment Configuration
    actor_rollout_ref.env.name=search\                          # Environment name (search task)
    actor_rollout_ref.env.mcp_mode=stdio\                       # MCP communication mode
    actor_rollout_ref.env.tool_manager=qwen3\                   # Tool manager for Qwen3
    actor_rollout_ref.env.enable_thinking=False\                # Disable thinking tokens
    actor_rollout_ref.env.config_path=envs/configs/mcp_tools.pydata\ # Path to tool configuration
    actor_rollout_ref.env.use_process_reward=False\             # Disable process-based rewards
    # Reward Model Configuration (Optional: for model-based judging)
    reward_rollout.if_use_reward_rollout=False\                 # Disable reward model rollout (use rule-based)
    reward_rollout.rollout.tensor_model_parallel_size=4\        # Tensor parallelism for reward model
    reward_rollout.rollout.gpu_memory_utilization=0.65\         # GPU memory for reward model
    reward_rollout.rollout.model_name=$REWARD_MODEL_PATH\       # Path to reward/judge model
    reward_rollout.rollout.free_cache_engine=True\              # Free cache after generation
    reward_rollout.rollout.response_length=2048\                # Max response length for judge model
    reward_model.reward_manager=parallel\                       # Parallel reward computation
    # Algorithm Configuration
    algorithm.kl_ctrl.kl_coef=0.001\                            # KL coefficient for policy constraint
    # Trainer Configuration
    trainer.critic_warmup=0\                                    # No critic warmup (GRPO doesn't use critic)
    trainer.logger=['tensorboard']\                             # Use TensorBoard for logging
    trainer.project_name='GRPO_search'\                         # Project name for tracking
    trainer.experiment_name='search_with_thinking'\             # Experiment name
    trainer.n_gpus_per_node=8\                                  # Number of GPUs per node
    trainer.nnodes=1\                                           # Number of nodes
    trainer.val_before_train=False\                             # Skip validation before training
    trainer.default_local_dir=$RESULT_DIR\                      # Directory for saving results
    trainer.default_hdfs_dir=null\                              # No HDFS directory
    trainer.save_freq=20\                                       # Save checkpoint every 20 steps
    trainer.test_freq=10\                                       # Test every 10 steps
    trainer.total_epochs=5 $@ 2>&1 | tee grpo.log              # Train for 5 epochs, log to grpo.log