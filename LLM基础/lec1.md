# Basics


# Systems
## Kernels
介绍了A100的计算单元长啥样
DRAM 动态存储单元 和SRAM
数据一般存在DRAM里面 如果每次运输从DRAM也就是显存中取数，速度就会很慢。
解决方案就是在SRAM上 存一些数据 然后重复运算。
但是具体咋运输的 不清楚。重复运算的步骤是啥
GPT给了一个 矩阵乘法的例子
算 C[i,j] 
每次计算 `C[i,j]`，都要从 DRAM 把 **整行 A** 和 **整列 B** 拿过来，算一次就要搬一次，非常低效。
**Optimized，用 SRAM）**：  
先把 A 和 B 的一个小块搬到 SRAM（绿色框），然后在工厂里**反复利用这个小块**，一次性算出 C 里的一大块。等这块算完，再去搬下一批数据。
## Parallesim
并行 当用多块GPU的时候咋办
GPU之间搬运数据 很慢，甚至比GPU从显存中取数更慢。在多 GPU 并行训练时，跨 GPU 的数据传输是性能瓶颈，所以要尽量减少 GPU 之间的数据交流。

但是多GPU训练的时候 不可避免的要互相沟通。有的时候需要把不同的GPU的结果汇总
有的时候要把同一份数据复制到所有GPU 这种交流操作统称为 **collective operations**，常见的有：**all-reduce**：大家各算一部分，再把结果加起来，最后每张卡都有完整的结果。  **gather / reduce / broadcast**：不同方向的数据汇聚、分发。


有的时候模型太大，一块GPU放不下，就得把不同的东西“切开”分到多卡上
比如 
parameters 模型参数
activations 反向传播的中间结果
gradients 反向传播的梯度
optimizer States 优化器的额外状态 比如 Adam的动量项


并行的方法：
Data parallelism 数据并行

Tensor Parallelism 张量并行

pipline parallelism 流水线并行

Sequence Parallelism 序列并行

## Inference
推理目标：通过 prompt 来生成token
Inference  同样需要 强化学习 reinforcement learning
test-time compute  和 evaluation

Inference 分为两个部分 
prefill 预填充和 decode 解码
如图
![[Pasted image 20250915174024.png]]
Prefill(和训练时类似) Token 给了，一次性把token全部处理 (compute-bound)
Decode： 需要一次生成一个 token (memory-bound)

加速 Decoding的方法：
	用cheaper model ，通过 模型pruning quentization  和 蒸馏 distillation
	Speculative decoding  用更便宜的 draft 模型，小模型 去生成 多个 multiple tokens 然后用full model 来 score 打分 并行地打分。 大模型不需要慢慢生成这这些token ，如果生成的前面的toekn 符合大模型的分布 就接受 不符合 大模型从不符合的地方开始接管 然后继续生成。

system optimization : KV caching ,Batching


# Scaling laws
Scaling laws 的目标：小规模做实验 大规模 预测 超参数或者损失

问题来了：给定一个 FLOPs budget (C) 用 更大的模型 N 或者train on more tokens(D)?

![[Pasted image 20250915180213.png]]


# Data
Evaluation 评估
	困惑度 perplexity textbook evaluation 对于 language models 衡量模型在预测下一个词时的平均不确定性。数值越低，说明模型越能“准确预测”文本分布。 **局限**：只反映对“训练分布的拟合程度”，和真实下游任务的表现不一定一致。
	Standard testing 标准测试  例如MMLU HellaSwag  GSM8K - 指一些**标准化基准测试**，用来系统比较不同模型的能力。
- 举例：
    - **MMLU**：多学科知识测试（数学、历史、医学…）。
    - **HellaSwag**：常识推理。
    - **GSM8K**：小学数学题（算术推理）。
- **作用**：像学生考试一样，统一打分，便于横向比较
	Instruction following 例如 AlpacaEval  IFEval  WildBench - 测试模型能不能正确理解并遵循自然语言指令
- 举例：
    - **AlpacaEval**：基于 Alpaca 指令数据的评估。
    - **IFEval**：Instruction Following Eval。
    - **WildBench**：更贴近真实环境的任务评估。
- **作用**：检验“能不能听懂人话”，比如问答、写文章、总结。
	Scaling test-time compute 例如 CoT chain of thought， ensembling - 在**推理阶段（test-time）投入更多计算**，是否能让模型表现更好。
- 方法：
    - **Chain-of-thought (CoT)**：让模型一步步推理，写出中间步骤，而不是直接给答案。
    - **Ensembling**：让模型生成多个答案，再投票/融合。
- **作用**：研究“加推理算力，性能能否继续提升”。
	LM as a judge  评估generative tasks - 用另一个大模型（比如 GPT-4）来**自动评价**生成任务的质量。
	- 举例：比较两段生成文本，哪个更流畅/更符合要求，由模型来打分。
	- **作用**：解决人工评测成本高的问题。
	Full system RAG，agent - 不只是评测**模型本身**，而是评测**整个系统**。
	- 举例：
    - **RAG**（检索增强生成）：模型+检索库。
    - **Agent**：模型+环境交互（工具调用、多轮对话）。
- **作用**：看模型在“真实应用场景”里的综合表现，而不仅是纯文本预测。


Data curation 数据管理
这里有一个 Assignment4 是关于如何管理与处理数据的

Supervised finetuning (SFT)
这个其实就是 给出 Promt 和回答，逐渐让模型 学会我们要的回答模式和数据



# Assignment 总结
## A1
实施 BPE tokenizer 
Implement **Transformer   Cross entropy Loss AdamW optimizer, training loop ,Train on TinyStories and OpenWebText ** 
减少 OpenWebText perplexity 在 H100 上

## A2
实现 **RMSNorm kernel** 
实现 **distributed data Parallel training** 
实现 **optimizer state sharding** 优化器状态分片
Benchmark and profile the implementations 对实现进行基准测试和分析

## A3
在以前的 runs 上定义一个训练 API  （Hyperparameter ——> loss）
提交 training jobs (under a FLOPs budget) and gather data points
在 data points 上实施 scaling law

## A4
将 爬虫爬到的 HTML 转换为text
训练一个 分类器 将 harmful 和 quality 内容分开
用 MinHash 去重
在给定 token 的预算下 最小化困惑度

## A5
实施 supervised fine-tuning
实施 Direct Preference Optimization (DPO)
实施 Group Relative Preference Optimization (GRPO)



