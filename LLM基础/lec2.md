overview:
1. primitives needed 训练模型的基本要求
2. 从底层开始来优化训练循环
3. 关注 efficiency

这个Lec之后 要完成一个Assignment 标题为：
Building a Transformer LM

# Motivating_questions
训练一个 70B 参数的模型 数据有15Ttoken ，用1024个H100需要多久？
## **Question**: How long would it take to train a 70B parameter model on 15T tokens on 1024 H100s?

`hint: 70B 参数是700亿参数的意思，指的是模型里可训练的权重的个数`
`15T tokens = 15 Trillion tokens` 15 万亿个token 指你要为给模型训练的语料长度。也就是总共的训练数据的大小。

`total_flops = 6 * 70e9 * 15e12`  
*flops 意思是计算的次数 这是训练所需的总计算量。乘6的意思是每个参数在前向和反向传播中大概需要 6次浮点数运算。* 为什么是6 后面会进行计算

`mfu = 0.5` 
*Model FLOPs Utilization*  含义是模型在训练过程中真正用的算力/GPU理论峰值算力。如果一块 H100 的理论峰值是 **1979 TFLOP/s**，但训练时由于 **通信开销、显存带宽、调度效率、kernel 实现等**原因，实际只能跑到一半的效率。

`flops_per_day = h100_flop_per_sec * mfu * 1024 * 60 * 60 * 24`  =4.37e+22
`days = total_flops / flops_per_day` = 143.92

## **Question**: What's the largest model that can you can train on 8 H100s using AdamW (naively)?
8 张H100 上 用 AdamW 最大可以训练多大的模型？

`h100_bytes = 80e9` 一张H100 的显存是 80G。 因为 `1 GB ≈ 1e9 bytes`，所以写成 `80e9`（= 80 × 10^9 字节）。

`bytes_per_parameter = 4 + 4 + (4 + 4) # parameters, gradients, optimizer state` 
因为我们一般用 float32，因此一个数字是 32/8 = 4  byte
对于一个参数，**参数本身**要存，用一个float。**梯度**要存，反向传播时，每个参数都会算出对应的梯度，并保存在内存里。
**优化器状态 (AdamW 特有的)**要存
- Adam/AdamW 需要为每个参数维护两个额外的状态：
    - **一阶动量 m**（类似于历史梯度的指数平均）
    - **二阶动量 v**（历史平方梯度的指数平均）

`num_parameters = (h100_bytes * 8) / bytes_per_parameter` 
这样就算出 8张H100 能训练的最大的模型。

### 混合精度训练（mixed precision）的存储细节
默认情况下，我们使用float32(4bytes) 来存参数和梯度。
参数：4 byte
梯度：4 byte
Adam 状态： 8 Byte
一共16 Bytes

但是更常见的做法是 用bf16 来存储：
参数： 2 bytes
梯度： 2 bytes
Adam状态：8 Bytes
这样就变成 2 + 2 + 4 +4 = 12 Bytes了 比 float 减少了25%

但是我们在实际训练里，为了避免精度丢失，通常还保留一份 float32的parameter 权重。这样总开销又回到了 16 Bytes

另外一个问题是：这里没有把 activation 算进去。激活的显存开销 和 batch size，序列长度，层数相关。如果batch size很大 或者样本token很长。激活值可能比参数本身占用的显存还大。实际上激活值参数显存的主要瓶颈。


# Memory Accounting 内存计算
## Tensor 基础
创建tensor 
```python
x = torch.tensor([[1., 2, 3], [4, 5, 6]]) 

x = torch.zeros(4, 8) # 4x8 matrix of all zeros

x = torch.ones(4, 8) # 4x8 matrix of all ones 

x = torch.randn(4, 8) # 4x8 matrix of iid Normal(0, 1) samples 
```
分配未初始化的值：
```python
x = torch.empty(4,8)
```
注意，这里不是张量里面没有value。而是分配了一块内存，但是不对这块内存做初始化。所以tensor的值是 原本内存中残留数。通常看起来像是随机数。
创建完参数tensor之后，要对其进行初始化
```
nn.init.trunc_normal_(x, mean=0, std=1, a=-2, b=2)
```
*trunc= truncated（截断）normal 正态分布， _结尾 = in-place 操作（直接修改张量x本身，不返回新的张量）*
作用是 用一个截断正态分布 来初始化x 。也是就说 随机数来自 均值为0标准差是1的分布，但是会被限制在 -2，2 之间，超过范围的值会被丢弃并重新采样。

## Tensor的内存
### float32
32位浮点数
![[Pasted image 20250929195256.png]]
深度学习中默认使用32位的浮点数。 
**sign(符号位)** : 1 bit  0表示正数，1表示负数
**Exponent(指数位)**：存的是移码(biased exponent) 。 `真实的指数= 存的指数-127` 127就是bias
**Fraction(尾数/小数位)**: 23bits  存的是小数部分(mantissa)，前面有个隐含的1(没写出来) 所以有效位是 1+23 = 24。
> [!tip]- 为什么要这样设计？为什么是23位但是有效位却是24位？
> 1.xxxxx... × 2^e
> 前导的 1 是必然存在的（除非是 denormalized，这时没有隐含 1）。
> 既然必然存在，就没必要存储，直接省下 1 bit 用于扩大 fraction 精度。

```python
x = torch.zero(4,8)
assert x.dype == torch.float32
assert x.numel() == 4*8 # number of elements
assert x.element_size() == 4 # Float is 4 bytes
assert get_memory_usage(x) == 4 * 8 * 4 # 128 bytes
# 这个get_memory_usage(x) 是
```
