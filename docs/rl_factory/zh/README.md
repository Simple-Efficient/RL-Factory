# RLFactory: Plug-and-Play RL Post-Training Framework for Empowering LLM Tool-Use

<div align="center">
 <b>Train your Tool-use Agent Now! </b>
</div>

<div align="center">
<table>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/9793f779-c80e-48e6-813a-1c8f377cf5d1" alt="Description" style="width:120px; height:auto;"/>
    </td>
    <td style="width:600px; word-break:break-all;">
      <p>
        <b>RLFactory</b>是一个即插即用的RL post-training框架，主要的infra是基于<a href="https://github.com/volcengine/veRL">verl</a>的，之所以将其与verl库分离而新建一个项目是因为我们希望专注于LLM tool use（尤其是multi-turn tool use），其包含两个核心的特征：<b>工具调用能力训练</b> & <b>多样的reward计算方式</b>。
      </p>
    </td>
  </tr>
</table>
</div>

- **工具调用能力训练**
  + **MCP工具调用**：得益于Qwen3模型的工具调用能力以及[qwen_agent](https://github.com/QwenLM/Qwen-Agent) 代码库对mcp工具的支持，我们将其集成在了RL训练过程中，用户**仅需提供mcp的config设置文件**即可无痛地一键训练！
  + **高效的并行工具调用**：基于asyncio实现的异步并行工具调用，工具调用将不再成为RL训练的关键瓶颈！同时提供了**loss mask**实现只对模型生成的内容进行训练，收敛速度大大提升！
  + **低成本工具环境搭建**：**完全解耦**的代码库设计，以极低的成本构建自己的工具调用环境！我们参考[Search-R1](https://github.com/PeterGriffinJin/Search-R1)提供了针对深度搜索的例程，后续也会实现基于WebUI的环境搭建方式。
- **多样的reward计算方式**
  + **基于规则的reward**：大体沿用verl的reward计算方式，但引入asyncio实现异步并行的奖赏计算（主要为了支持**使用工具计算reward**的需求，例如NL2SQL任务）。
  + **基于Model Judge的reward**：QwQ-32B这个模型具备较强的推理能力，能够很好地通过reasoning的方式评判每一个response，本框架能够支持这种需求下的分布式部署和调用，有极高的reward rollout效率。

**为什么要做针对Multi-turn tool use的RL训练？**

+ **业界的应用需求**：深度搜索、Computer Use、旅行规划等任务实际上都是多轮工具调用问题，急需一个针对性的、即插即用的RL post-training框架增强模型在垂类任务上能力
+ **Agent的发展趋势**：LLM Agent将从调用工具，逐步进化为调用LLM作为工具，以及调用其他Agent构成Agentic Graph，因此前瞻性地构建这一框架来实现Agentic post-training是很有必要的
+ **Qwen3模型带来的优势**：尽管Qwen3的整体性能尚弱于Deepseek-R1等SOTA模型，但其工具调用能力和指令遵循能力已经可以垂域训练忽略SFT阶段而直接进行RL训练，带来了更多的机会

## 概念辨析
+ 为了帮助使用者理解RLFactory能做什么事，我们在此进行一下概念辨析，回答以下两个问题
+ **问题：什么是工具？**
+ **回答**：广义上讲，RLFactory支持的工具指的是“不是由当前正在训练的模型生成的”，可以是一些程序，也可以是其他模型，甚至可以是其他的智能体
    + **程序形式**：各类搜索接口（输出查询内容-输出查询结果）、代码解释器（输入代码文本-输出程序运行结果）、计算器（输入计算公式-输出计算结果）
    + **模型形式**：其他的开源或闭源模型（例如使用GPT-4o做文档总结，输入Prompt-输出Response）
    + **智能体形式**：一系列程序+模型的集合（例如文献调研智能体，输入调研主题-输出调研结果）
+ **问题：为什么需要多轮工具调用？**
    + **回答**：对于给定的工具调用任务，人工编排的形式需要引入大量的人力设计，而使用RLFactory则可以训练一个端到端的Agent模型，由模型通过推理自行决定调用哪些工具、如何调用工具、何时停止交互过程、如何整理输出内容等等，是未来LLM应用的重要趋势，其交互流程如下图所示：
        <div align="center">
            <img src="https://github.com/user-attachments/assets/36bf88f7-d586-4cd4-a06e-aefbd807d3c0" alt="Description" style="width:750px; height:auto;"/>
        </div>
    + **端到端Agent模型运行流程**
        + **步骤1**：输入原始的`Prompt`，调用Training Model输出对应的`Response`
        + **步骤2**：对模型的输出进行后处理，尝试从中解析工具名字及相关参数（若未解析到工具则视为循环终止，输出模型回复）
        + **步骤3**：根据解析得到的工具名称及参数运行该工具（可能需要并行调用多个工具），并对工具返回的结果进行后处理
        + **步骤4**：将后处理过的工具结果添加到`Prompt`中，再次调用模型直至循环终止

## 关键特性列表
- RLFactory框架允许用户以极低的代码量实现针对Multi-Turn Tool Use的RL训练，在下图中，我们将框架的核心模块做了大致的分类
  + **应用层**：使用者需要构造自己的Env。使用者需要提供工具的配置文件`mcp_tools.pydata`以及如何计算奖赏函数（`compute_score`）
  + **组件层**：使用者可以不修改，但如果有一些特殊的需求可以修改或增加一些组件及相关功能，例如构造新的`ToolManager`
  + **基础层**：使用者完全无需修改，包括Multi-Turn Tool Use的交互逻辑、Model Judge的框架基建以及verl中原生的强化学习训练机制
    <div align="center">
      <img src="https://github.com/user-attachments/assets/475ff62c-83b7-4b74-a1e7-e5bea151b92b" alt="Description" style="width:750px; height:auto;"/>
    </div>
- 下图以GRPO算法为例，介绍了RLFactory的各个关键模块在强化学习的过程中起到何种作用，其与原始verl中的RL训练相比主要在于**工具调用环境的低代码性**和**奖赏函数设计的高效性**上
    <div align="center">
        <img src="https://github.com/user-attachments/assets/77232f2d-0636-4723-a733-1fcd67ac70fe" alt="Description" style="width:600px; height:auto;"/>
    </div>
- **工具调用环境**：核心文件在`envs`文件夹下，详细介绍请见[`docs/rl_factory/en/tools.md`](docs/rl_factory/en/tools.md)
  + **工具定义**：以`qwen_agent`为基础，支持自定义工具（继承自`BaseTool`）、MCP工具集以及`qwen_agent`自带工具（例如`code_interpreter`）
  + **工具解析**：在`tool_manager`文件夹下注册你自己的`manager`，用于实现你自己的工具解析逻辑，当然你也可以直接用我们的`qwen3_manager.py`，利用`qwen_agent`的相关方法实现了以上三类工具的解析、启动和灵活的调用
  + **工具调用**：在`envs`文件夹下注册你自己的`Env`，用于实现你整体的**工具调用逻辑**以及**奖赏计算逻辑**。其中的step函数提供了默认的工具调用逻辑，输入一个batch的responses，输出工具调用的结果，真正地把**工具调用**当成了与LLM**解耦**的环境！
  + **多轮工具调用及训练**：在`envs/utils/tool_utils.py`中实现多轮工具调用的prompt拼接、`loss_mask`定义及计算，**准确无误**地实现基于LLM的多轮工具调用！
- **奖赏计算**：核心内容在`envs`文件夹下，详细介绍请见[`docs/rl_factory/en/rewards.md`](docs/rl_factory/en/rewards.md)
  + **基于规则**：和在verl的基础上基于`asyncio`实现了异步的**并行reward计算**，主要是为了与后续并行评估保持兼容
  + **Model Judge**：额外引入了一个`reward_rollout_wg`并为其新建了**额外的资源池**，可以使用vllm实现对QwQ等LRM进行分布式的部署和调用，并在`Env`中使用它生成**模型的Judge结果**，以此实现对原始response的评估
  + **工具验证**：在`Env`中额外引入了`verify_tool`的方法，其能够在计算奖赏的时候被调用来评估原始response。以NL2SQL任务为例，模型最终输出的是一段代码，RLFactory可以执行这一SQL语句来**判断程序的准确性**，我们同样采用异步并行的方式**提高调用效率**（当然也可以在这里调用API来实现Model Judge）

## 安装方法
- **依赖环境（关键）**：
  ```yaml
  Cuda: >=12.0 （推荐12.4）
  Python: >=3.10  （推荐3.10）
  # 为了支持Qwen3模型
  vllm: >=0.8.3 （推荐0.8.5）
  ```
- **安装依赖项**：
  ```bash
  pip3 install accelerate bitsandbytes datasets deepspeed==0.16.4 einops flash-attn==2.7.0.post2 isort jsonlines loralib optimum packaging peft pynvml>=12.0.0 ray[default]==2.42.0 tensorboard torch torchmetrics tqdm transformers==4.48.3 transformers_stream_generator wandb wheel
  pip3 install vllm==0.8.5      # 主要是为了支持Qwen3模型
  pip3 install "qwen-agent[code_interpreter]"
  pip3 install llama_index bs4 pymilvus infinity_client codetiming tensordict==0.6 omegaconf torchdata==0.10.0 hydra-core easydict dill python-multipart
  pip3 install -e . --no-deps
  pip3 install faiss-gpu-cu12   # 可选，使用rag_server训练端到端搜索模型需要使用
  ```
<div style="padding:10px; background-color:#fff3cd; color:#856404; border:1px solid #ffeeba; border-radius:4px;">
<strong>Note:</strong> 目前仅支持Qwen3模型，并未针对其他模型进行适配调整
</div>

## 即将支持的特性
- **图形化界面**：通过图形化界面设置工具池，一键开始属于你的RL post-training！
- **过程奖励**：在每一次工具调用结束后都计算一次奖赏，使用PPO算法更有效地优化模型！（目前流程上已经支持，但并未进行稳定性测试）
- **兼容性优化**：适配旧一些的Qwen2.5系列模型等，在不同计算平台上的运行测试
- **效率优化**：优化多卡多机的训练效率，异步的模型Rollout & 工具调用，高并发的并行MCP Server部署

<div align="center">
  <img src="https://github.com/user-attachments/assets/883fd8c0-afa9-4ed2-95be-333a79ce7e36" alt="Description" style="width:600px; height:auto;"/>
</div>

## 使用说明
- **你需要提供什么？一个工具调用环境！**
  + **最小实现教程详见[`docs/rl_factory/main_tutorial.md`](docs/rl_factory/main_tutorial.md)**
  + **步骤1**：复制`envs/base.py`，新建一个你自己的交互环境文件`xxx.py`
  + **步骤2**：构建你的工具集（推荐放在`envs/tools`文件夹下），并将配置文件放在`envs/configs`文件夹下
  + **步骤3**：在`xxx.py`文件内修改`_compute_score_with_rules`或`_compute_score_with_reward_rollout_wg`，实现奖赏信号的计算逻辑（详细请参考[`docs/rl_factory/en/rewards.md`](docs/rl_factory/en/rewards.md)）
  + **步骤4（可选项）**：如果你需要调用工具来计算reward，请定义你自己的`verify_tool`。需要注意的是，你在这个方法中可以使用`self.tool_manager`中**已被注册的工具**
  + **步骤5**：调整一下你的超参数，开始训练！
- **训练指令**：
  ```bash
  # 在运行程序前，请先按照实际情况修改MODEL_PATH、REWARD_MODEL_PATH以及actor_rollout_ref.env的几个参数
  bash main_grpo.sh
  ```

## 使用示例
+ 我们在[`docs/rl_factory/main_tutorial.md`](docs/rl_factory/main_tutorial.md)中提供了RLFactory复现[Search-R1](https://github.com/PeterGriffinJin/Search-R1)的示例，我们以`Qwen3-4B`为基础模型进行RL训练，最终模型的测试分数（0.486）甚至超过使用Search-R1训练得到的7B模型`Qwen2.5-7B-Instruct-GRPO`（0.429），并且在训练资源更差的情况下，训练效率大幅提升了6.8倍！（34h -> 5h）
<div align="center">
  <img src="https://github.com/user-attachments/assets/3a37dd61-0da5-4d69-a3e5-b18fb8197233" alt="Description" style="width:500px; height:auto;"/>
</div>

<p align="center">

| Model Name | Test Score（NQ） | 收敛时间 | 训练资源 |
| --- | :---: | :---: | :---: |
| Search-R1-Qwen2.5-3B-Instruct-GRPO | 0.397 | 34h | H100 * 8 |
| Search-R1-Qwen2.5-7B-Instruct-GRPO | 0.429 | - | H100 * 8 |
| **RLFactory-Qwen3-4B-GRPO** | **0.486** | **5h** | A100 * 8 | 

</p>
