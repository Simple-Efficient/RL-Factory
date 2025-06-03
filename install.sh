pip3 install accelerate bitsandbytes datasets deepspeed einops isort jsonlines loralib optimum packaging peft pynvml>=12.0.0 ray[default]==2.42.0 tensorboard torchmetrics tqdm transformers==4.48.3 transformers_stream_generator wandb wheel
pip3 install vllm==0.8.5
pip3 install "qwen-agent[code_interpreter]"
pip3 install llama_index bs4 pymilvus infinity_client codetiming tensordict==0.6 omegaconf torchdata==0.10.0 hydra-core easydict dill python-multipart
pip3 install -e . --no-deps
pip3 install faiss-gpu-cu12