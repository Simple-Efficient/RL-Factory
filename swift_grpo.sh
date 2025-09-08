#!/bin/bash
# usage:
# run bash swift_grpo.sh rollout, then run bash swift_grpo.sh train
# 下方为用Qwen3跑R1-search的案例


VLLM_IP=localhots
MODEL_PATH=Qwen/Qwen3-4B
#通过plugin和下方的--use_hf会从hf自动拉数据
DATASET=RUC-NLPIR/FlashRAG_datasets
if [ -z $2 ]; then
    DATETIME=$(date "+%Y%m%d%H%M%S")
else
    DATETIME=$(date "+%Y%m%d%H%M%S")/$2
fi
LOGDIR=./log/$DATETIME/logdir
OUTDIR=./log/$DATETIME/outdir

export PYTHONPATH=$(pwd):$PYTHONPATH
export NCCL_DEBUG=WARN


# 什么时候用分离模式: 
# 多机下有些时候发现colocote速度不如分离, 即便分离使用了更多资源, 主要为多机且tool耗时相比rollout比例大的时候;
if [ $1 == "colocate" ]; then
RANK=0 \
MASTER_ADDR=localhost \
MASTER_PORT=29555 \
NPROC_PER_NODE=8 \
WORLD_SIZE=1 \
swift rlhf \
    --use_hf \
    --vllm_mode colocate \
    --dataloader_num_workers 8 \
    --use_vllm True \
    --vllm_tensor_parallel_size 4 \
    --vllm_pipeline_parallel_size 2 \
    --vllm_gpu_memory_utilization 0.5 \
    --offload_optimizer True \
    --offload_model     True \
    --sleep_level       1 \
    --rlhf_type grpo \
    --dataset $DATASET \
    --model   $MODEL_PATH \
    --use_async_engine true \
    --external_plugins ./swift/rlfactory_reward.py \
                       ./swift/rlfactory_toolcall.py \
                       ./swift/r1_reward.py \
                       ./swift/r1_dataset.py \
    --reward_funcs r1_reward \
    --multi_turn_scheduler rlfactory_toolcall \
    --system 'Answer the question, and when you are unsure, use query_rag as the search tool.  \
        Enclose each tool call within <tool_call>...</tool_call> tags, \
        using this format: <tool_call>{"name": "query_rag", "arguments": {"query": "put query here"}}</tool_call>, \
        the query argument should be a json string with query argument stating your question. \
        if the tool returns error or you are still unsure, you may fix the tool_call and call the tool again. \
        The answer should be enclosed within <answer>...</answer> tags. \
        The Question is: ' \
    --add_version True \
    --max_length 4096 \
    --max_completion_length 2048 \
    --max_turns 3 \
    --num_generations 4 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size  2 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 4 \
    --save_total_limit 3 \
    --save_steps 100 \
    --train_type full \
    --temperature 1 \
    --top_p 0.9 \
    --top_k 50 \
    --epsilon 0.2 \
    --delta 1.2 \
    --deepspeed zero3 \
    --sequence_parallel_size 1 \
    --logging_steps 1 \
    --report_to tensorboard \
    --logging_dir $LOGDIR \
    --output_dir  $OUTDIR
#分离模式, 分别启动server和rollout进程
elif [ $1 == "rollout" ]; then
CUDA_VISIBLE_DEVICES=0,1,2,3 \
RANK=0 \
MASTER_ADDR=localhost \
MASTER_PORT=29500 \
NPROC_PER_NODE=4 \
WORLD_SIZE=1 \
swift rollout \
    --use_async_engine true \
    --vllm_tensor_parallel_size 4 \
    --vllm_data_parallel_size 1 \
    --vllm_enable_prefix_caching True \
    --vllm_gpu_memory_utilization 0.4 \
    --external_plugins ./swift/rlfactory_toolcall.py \
    --multi_turn_scheduler rlfactory_toolcall \
    --max_length 4096 \
    --max_turns 3 \
    --port 8421 \
    --model $MODEL_PATH
elif [ $1 == "train" ]; then
#将下列的vllm_mode改为colocate可以在同一进程中同时部署vllm, 并且能在train过程中offload vllm节省显存, 但是无法使用vllm的AsyncEngine: https://raw.githubusercontent.com/modelscope/ms-swift/main/docs/resources/asyncengine.png
#好处是colocate可以在训练中offload vllm parameter
#rlfactory_reward 为rlfactory自带的格式分, 如果不需要则去掉这个reward function即可 --reward_funcs rlfactory
CUDA_VISIBLE_DEVICES=4,5,6,7 \
RANK=0 \
MASTER_ADDR=localhost \
MASTER_PORT=29501 \
NPROC_PER_NODE=4 \
WORLD_SIZE=1 \
swift rlhf \
    --use_hf \
    --vllm_mode server \
    --dataloader_num_workers 8 \
    --use_vllm True \
    --vllm_server_host $VLLM_IP \
    --vllm_server_port 8421 \
    --vllm_server_timeout 180 \
    --rlhf_type grpo \
    --dataset $DATASET \
    --model   $MODEL_PATH \
    --external_plugins ./swift/rlfactory_reward.py \
                       ./swift/rlfactory_toolcall.py \
                       ./swift/r1_reward.py \
                       ./swift/r1_dataset.py \
    --reward_funcs r1_reward \
    --system 'Answer the question, and when you are unsure, use query_rag as the search tool.  \
        Enclose each tool call within <tool_call>...</tool_call> tags, \
        using this format: <tool_call>{"name": "query_rag", "arguments": {"query": "put query here"}}</tool_call>, \
        the query argument should be a json string with query argument stating your question. \
        if the tool returns error or you are still unsure, you may fix the tool_call and call the tool again. \
        The answer should be enclosed within <answer>...</answer> tags. \
        The Question is: ' \
    --add_version True \
    --max_length 4096 \
    --max_completion_length 2048 \
    --max_turns 3 \
    --num_generations 4 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size  2 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 4 \
    --save_total_limit 3 \
    --save_steps 100 \
    --train_type full \
    --temperature 1 \
    --top_p 0.9 \
    --top_k 50 \
    --epsilon 0.2 \
    --delta 1.2 \
    --deepspeed zero3 \
    --sequence_parallel_size 1 \
    --logging_steps 1 \
    --report_to tensorboard \
    --logging_dir $LOGDIR \
    --output_dir  $OUTDIR
else
    echo command $1 not recognized, use train or rollout
fi
