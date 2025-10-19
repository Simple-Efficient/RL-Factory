<div align="center">
  <img src="https://github.com/user-attachments/assets/9793f779-c80e-48e6-813a-1c8f377cf5d1" alt="Description" style="width:300px; height:auto;"/>
</div>

<div align="center">

[📘Tutorial](docs/rl_factory/en/main_tutorial.md) &#124; [🛠️Installation](docs/rl_factory/en/install.md) &#124; [🎨Framework](docs/rl_factory/en/framework_design.md) &#124; [🏆Model](https://huggingface.co/Simple-Efficient/RLFactory-Qwen3-8B-GRPO)

</div>

--- 

RLFactory is an **easy** and **efficient** RL post-training framework for **Agentic Learning**. 

RL-Factory decouples the environment from RL post-training, enabling training with just a tool config and reward function while supporting async tool-calling to make RL post-training **2x faster**.

Current version natively supports one-click **DeepSearch** training and features multi-turn tool-calling, model judge reward, and training of multiple models including **Qwen3**. More easy and efficient agentic learning modules will be added in upcoming features.

<div align="center">
  <b>Now, everyone can easily and quickly train an Agent with Qwen3 (as base models) and MCP tools!</b>
</div>

## Release Log
We’ll keep a fast release cycle to quickly deliver and polish the upcoming features.
+ **Version 0.1**
  + **Environment decouple**: define your tool-use envinroment easily (tools setup and reward function definition)
  + **Qwen3 Model support**: quickly train your agent using Qwen3 (much better than Qwen2.5 in tool-call)
  + **Efficient training**: 2x faster than existing frameworks for rapid model iteration (mainly through async tool-use)
+ **Version 0.2** We are looking forward to more people participating in the development and construction together, to create a great Agenetic Training community. Please feel free to contact us.
  + **WebUI**: build a WebUI for data processing, tool & environment definition, training configuration, and project management [#2](https://github.com/Simple-Efficient/RL-Factory/issues/2)
  + **More efficient training**: support the AsyncLLMEngine for more efficient rollout [#4](https://github.com/Simple-Efficient/RL-Factory/issues/4)
  + **More models**: test more models (such as Deepseek, Llama, etc.) and add corresponding support configurations [#5](https://github.com/Simple-Efficient/RL-Factory/issues/5)
  + **Process Reward**: use process reward to better guide the tool-call behavior of your model [#6](https://github.com/Simple-Efficient/RL-Factory/issues/6)
  + **More applications**: help create more demos (such as [TravelPlanner](https://github.com/OSU-NLP-Group/TravelPlanner)) to adapt to more benchmarks
  + **Multimodal agentic learning**: support multimodal (image) agent training in terms of functionality [#66](https://github.com/Simple-Efficient/RL-Factory/issues/66)
  + **Android Environment**: added Android OS environment support [#38](https://github.com/Simple-Efficient/RL-Factory/pull/38)
  + **Tools cache**: cached tool invocation results to enhance post-processing efficiency [#57](https://github.com/Simple-Efficient/RL-Factory/pull/57)
  + **Handy evaluation**: added main_eval.sh for evaluation utility [#36](https://github.com/Simple-Efficient/RL-Factory/pull/36)
  + **Upgrade to VeRL-0.5**: Upgraded to VeRL-0.5 with maximal component decoupling [update/verl_0_5](https://github.com/Simple-Efficient/RL-Factory/tree/update/verl_0_5)
  + **Add MS-SWIFT-3.7 to RL-Factory**: Added the support of MS-SWIFT-3.7 to make it more convenient for individual developers [#95](https://github.com/Simple-Efficient/RL-Factory/pull/95)

## Our Framework Design
Our goal is to enable users to focus on reward logic and tool setup for fast agentic learning with minimal code, while hardcore developers could focus on improving training efficiency and model performance. 

For **easy-to-use**, we decouple the environment from RL-based post-training with several advantages. 
+ **Easy-to-design reward function**: Calculate rewards through **rules**, **model-judge**, and even **tools** to meet all your requirements for reward function.
+ **Seamless tool setup**: Simply provide the configuration file for your **MCP tools** and custom tools to integrate them into RL learning.
+ **Multi-Agent extention**: Convert your agent to the MCP format for easy Multi-Agent Interaction. LLM chat simulation will be also added in the future to improve multi-turn dialogue capabilities. 

For **efficient learning**, we develope several essential modules within the RL post-training framework, making training **2x faster**.
+ **Efficient tool-call**: Improve online RL training efficiency through batch processing and asynchronous parallel tool calls.
+ **Efficient reward calculation**: Deploy LRM (like QwQ-32B) in a distributed manner for efficient **model judging**, and use asynchronous parallelism to speed up reward calculation. 

For **future progression**, we will continue to prioritize **"easy"** and **"efficient"**.
+ **Easier**: Use WebUI to process data, define tool & environment, adjust training configuration, and manage project. (The WebUI is under rapid development.)
+ **More efficient**: Continuously iterating and improving the training framework (such as AsyncLLMEngine) and RL training algorithms.

<div align="center">
  <img src="https://github.com/user-attachments/assets/883fd8c0-afa9-4ed2-95be-333a79ce7e36" alt="Description" style="width:750px; height:auto;"/>
</div>

## User Instructions
- **Dependencies (Key)**
  ```yaml
  Cuda: >=12.0 (Recommended: 12.4)
  Python: >=3.10  (Recommended: 3.10)
  # For Qwen3 model support
  vllm: >=0.8.3 (Recommended: 0.8.5)
  ```
- **Install Requirements**
  ```bash
  pip3 install accelerate bitsandbytes datasets deepspeed==0.16.4 einops flash-attn==2.7.0.post2 isort jsonlines loralib optimum packaging peft pynvml>=12.0.0 ray[default]==2.46.0 tensorboard torch==2.6.0 torchmetrics tqdm transformers==4.51.3 transformers_stream_generator wandb wheel
  pip3 install vllm==0.8.5      # Mainly for Qwen3 model support
  pip3 install "qwen-agent[code_interpreter]"
  pip3 install llama_index bs4 pymilvus infinity_client codetiming tensordict==0.6 omegaconf torchdata==0.10.0 hydra-core easydict dill python-multipart mcp==1.9.3
  pip3 install -e . --no-deps
  pip3 install faiss-gpu-cu12   # Optional, needed for end-to-end search model training with rag_server
  pip3 install nvidia-cublas-cu12==12.4.5.8  # Optional, needed while encountering ray worker died issue during training
  ```
  <div style="padding:10px; background-color:#fff3cd; color:#856404; border:1px solid #ffeeba; border-radius:4px;">
  <strong>Note:</strong> Currently, only Qwen models are tested.
  </div>
- **What do you need to provide?**
  + An **environment** is enough! See the minimal tutorial in [`docs/rl_factory/main_tutorial.md`](docs/rl_factory/main_tutorial.md)
- **Training Command**
  ```bash
  # Before running, modify MODEL_PATH, REWARD_MODEL_PATH, and several actor_rollout_ref.env parameters as needed
  bash main_grpo.sh
  ```
- **Evaluate or Infer Command**
  ```bash
  # Before running, modify MODEL_PATH, REWARD_MODEL_PATH, and several data and trainer parameters as needed
  bash main_eval.sh
  ```


## Demo in DeepSearch Training
+ In [`docs/rl_factory/main_tutorial.md`](docs/rl_factory/main_tutorial.md), we provide an RLFactory reproduction example of [Search-R1](https://github.com/PeterGriffinJin/Search-R1). We use `Qwen3-4B` and `Qwen3-8B` as the base model for RL training. 
+ **Easy**: Start with Qwen3 and MCP tools to quickly train your own DeepSearch Agent.
  + Provide only one tool configuration and one reward function to start training! 
  + Qwen3 demonstrates significant advantages in Agent Learning. It can accurately call tools even without SFT, and it also supports the MCP protocol.

+ **Efficient**: Enjoy the efficient training enabled by asynchronous parallel tool-call.
  + Compared to Search-R1 based on the original verl, the required training time is reduced by **1.5 to 2 times**, and the efficiency gain is even greater if a **model judge** is involved.
  + After 100 steps of training (about 5 hours in 8*A100), `Qwen3-4B` achieves a score of 0.458 and `Qwen3-8B` achieves a score of 0.463. 
+ The table below presents our training results under identical computational resources, software, and verl versions
  + RLFactory trains in about half the time of Search-R1, demonstrating high efficiency.
  + Qwen3 as the base model outperforms Qwen2.5, enabling domain-specific tool-calling via RL post-training without SFT.

<p align="center">

| Model Name | Test Score (NQ) | Total Training Time (100 step) | Seconds per step | Training Resources |
| --- | :---: | :---: | :---: | :---: |
| Search-R1-Qwen2.5-3B-Instruct-GRPO | 0.356 | 7.39 h | 266 s | A100 × 8 |
| Search-R1-Qwen2.5-7B-Instruct-GRPO | 0.451 | 9.25 h | 333 s | A100 × 8 |
| Search-R1-Qwen3-4B-GRPO | 0.420 | 7.95 h | 286 s | A100 × 8 |
| **RLFactory-Qwen3-4B-GRPO** | **0.458** | **5.30 h** | **190 s** | A100 × 8 | 
| **RLFactory-Qwen3-8B-GRPO** | **0.463** | **5.76 h** | **207 s** | A100 × 8 | 

</p>



## How to contribute?
We welcome all users and developers to contribute code to RLFactory. If you have any questions, encounter bugs, or would like to collaborate on development, please feel free to contact us!

1. Submit an issue directly on GitHub.  
2. Contact us via email at chaijiajun@meituan.com or gjyin@outlook.com.
3. Join our WeChat group(preferred) and become a pioneer in Agent training!
<div align="center">
  <img src="https://github.com/user-attachments/assets/bdb76d94-eb44-4fa1-a8f0-2f904be1e6c4" alt="Description" style="width:200px; height:auto;"/>
  <img src="https://github.com/user-attachments/assets/e59f1a14-faee-4ec8-97cd-5e90ff34a43d" alt="Description" style="width:200px; height:auto;"/>
</div>

## Acknowledgement
This repo benefits from [verl](https://github.com/volcengine/veRL), [Search-R1](https://github.com/PeterGriffinJin/Search-R1), [Qwen-Agent](https://github.com/QwenLM/Qwen-Agent). Thanks for their wonderful works. We will also introduce [TRL](https://github.com/huggingface/trl) in the future to further expand the applicability of our framework.

## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=Simple-Efficient/RL-Factory&type=Date)](https://www.star-history.com/#Simple-Efficient/RL-Factory&Date)  
