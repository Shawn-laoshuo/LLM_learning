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
# 这个get_memory_usage(x) 是 元素的个数*元素的size，最后得到结果
```
GPT-3(175B) 的隐藏维度 $d_{model}$  = 12288
$W_1$ 的形状 (12288,49152)
$W2$的形状(49152,12288)
两个矩阵都是 2.3 GB，一个就已经很大了

### float16
16位浮点数
![[Pasted image 20250929212929.png]]
相对于 32位浮点数 具体细节有所变化，但是细节没太大变化。
每个数字的大小  是 16bits/8 = 2 byte 
因此
```python
assert x.element_size() == 2
```
float16相对于 float32 的优点是运算速度快了。但是缺点是精度有所下降了。尤其是对小数的影响很大
>[!question]- 为什么特意提到小数？难道对大数的影响更小？
>float32 有 8bit指数 可以表示 从 $10^{-38}$ 到 $10^{38}$ 
>float16 只有 5bit指数 只能表示从 $10^{-5}$ 到 $10^{5}$
>用float16表示大数的时候，还能靠指数往上撑
>![[Pasted image 20250929214713.png]]
>![[Pasted image 20250929214739.png]]

```python
x = torch.tensor([1e-8],dtype=torch.float16)
assert x =0
```
这里出现了 underflow 下溢出。

### bfloat16
![[Pasted image 20250929215022.png]]
Google Brain 用 bfloat brain floating point  达到了 既拥有 float32的精度 也拥有float16的内存大小。


### fp8
![[Pasted image 20250929215139.png]]


## Compute Accounting 计算
### 把tensor 发送到GPU
![[Pasted image 20250930101001.png]]
我们需要通过PCI BUS 将tensor 从cpu发送到GPU。

### tensor operation 张量操作
大多数张量是通过在其他张量上 进行运算而创作的。每个操作都占用一些内存和有自己的计算结果。

#### tensor_storage 张量在内存中的存储方式
我们用 这个tensor来举例子
```python
x = torch.tensor([
[0., 1, 2, 3],
[4, 5, 6, 7],
[8, 9, 10, 11],
[12, 13, 14, 15],
])
```
![[Pasted image 20250930102055.png]]
>[!note] 数据在内存中到底是怎么存储的？
可以看到 数据在内存中是以**一整条**的方式来存储的。
但是我们用Pytorch操作数据的时候 是以张量的方式来操纵数据的。我们能通过张量的方式来操纵的原因在于，Pytorch这个库的作用其实是 作为**指针来帮助我们分配memory**，并且用metadata 来描述如何获得数据中的任意一个元素。

```python
assert x.stride(0) == 4
```
如果要转到下一行，(dim0)，需要跳过存储中的四个元素。
>[!note]- 在Pytorch里，stride 表述**在某个维度上移动一步，在底层内存中要跨多少个元素** 它决定了张量是怎么映射到底层连续内存的。

如果要转到下一列,(dim1)， 需要跳过存储中的一个元素。

#### tensor 切片

>[!note] Pytorch 的视图 view
Pytorch中许多操作只是提供tensor的不同视图(view)。这不是make a copy

取第一行
```python
y = x[0]
assert torch.equaly(y,torch.tensor([1.,2,3]))
assert same_storage(x,y)
```

```python
def same_storage(x: torch.Tensor, y: torch.Tensor):
	return x.untyped_storage().data_ptr() == y.untyped_storage().data_ptr()
```


y是取的x的第一行。 y的值肯定是和x的第一行是相等的。
同时 因为y取的是x的视图。y和x的内存肯定也是一块。

取第一列
```python
y = x[:1]
assert torch.equal(y,torch.tensor([2,5]))
assert same_storage(x,y)
```

用view 改变张量的 形状
view $2*3$ matrix as $3*2$:
```python
y = x.view(3,2) #将张量从2*3 改到3*2
assert torch.equal(y,torch.tensor([1,2],[3,4],[5,6]))
assert same_storage(x,y)
```

同样的 转置 transpose 矩阵同样不会对底层数据做修改
```python
y = x.transpose(1,0)
assert torch.equal(y,torch.tensor([1,4],[2,5],[3,6]))
assert same_storage(x,y)
```

在x矩阵中做修改，会营销到y矩阵。因为x和y是共享同一个内存的
```python
x[0][0] = 100
assert y[0][0] == 100
```

contiguous 和 non-contiguous的概念
```python
x = torch.tensor([1.,2,3],[4,5,6])
y = x.transpose(1,0)
assert not y.is_contigous() # x是内存上连续的一块，转置后 stride的方式变了，y就不是连续的了
try:
	y.view(2,3) # view 在内存是 contiguous的时候才能成功。否则不会成功的。
	assert False
```
根据这段代码，转置后由于 stride 的方向改变，因此转置后的矩阵在内存上不再是一整块连续的内存。

下面的代码可以让 tensor 强制变成 contiguous 的
```python
y = x.transpose(1,0).contiguous().view(2,3)
assert not same_storage(x,y)
```
我们可以看到 用  `.contiguous`方法将转置后的矩阵强制连续存储之后。和之前的x不是同一款存储空间了。


#### tensor element-wise 
element-wise 操纵 ，据说这个操作比较耗显存
加减 平方 开分号 乘除 都用 element-wise 的操作。

triu 用 upper triangular part 取矩阵的上三角
在计算 casual Attention mask(因果注意掩码) 的时候 很有用。`M[i,j]` 是i对j的 contribution

#### tensor matrix multiplication
参数一般用`w` 来表示。在运算中 `w` 一般是一个 矩阵的形式。一行是一个 w，多行 w 摞在一起就成了一个 `W` 矩阵。这个`W` 矩阵和 输出数据相乘，就构成了一个线性变换。
那么情况是这样，`W` 现在是一个 `[32,2]`的矩阵，输出数据应该是啥样的呢？可以确认的是 输出数据的最后一个维度一定是 32。而且这个32 还应该是 一条数据点的维度。

结合我们的推理和实际情况，那输出的数据 就应该是 `[N,C,H,W]`  
其中 N 是batch； C是 channel；H是高度，W是宽度。大概就是这样
![[Pasted image 20250930135427.png]]


#### torch的 einops **Einstein Operations**
>[!question] 什么是 einops
>`einops` 是一个 Python 库，全称 **Einstein Operations**。
>它提供了一种 **用爱因斯坦求和约定 (Einstein summation notation)** 来写张量的重排 / 合并 / 拆分操作。
>- 常用于 **深度学习** 里替代 `view`、`reshape`、`transpose`、`permute` 这些繁琐的 API。
>- `einops` 提供更直观的 **字符串语法**，比如 `"batch height width channel -> batch channel height width"`

>[!question] sequence数据送入模型是怎么做的？
>- N: Batch   一个batch里面有多个样本
>- seq: Sequence 一句话里面有多少个token
>- head: Attention Heads 在多头注意力里面 把hidden 维度切成多个头
>- hidden: 每个head 自己的 embedding维度。

>[!example] 例子：
> batch N = 2（两句话）
> seq = 5（每句话 5 个 token）
> head = 4（注意力头数）
> hidden = 16（每个 head 的维度）

输入的模型维度就是 shape = `(2, 5, 4, 16)`

对于一个 tensor 表示方式可以是：
`x = torch.ones(2, 2, 1, 3) # batch seq heads hidden`

也可以用 jaxtyping 的方式用来给张量加上 **类型注解 (type annotation)**：
`x: Float[torch.Tensor, "batch seq heads hidden"] = torch.ones(2, 2, 1, 3)`

##### einsum **Einstein summation convention**（爱因斯坦求和约定）来做矩阵乘法
>[!note] einsum 可以让矩阵乘法更简单。
>```python
>x: Float[torch.Tensor, "batch seq1 hidden"] = torch.ones(2, 3, 4) 
>y: Float[torch.Tensor, "batch seq2 hidden"] = torch.ones(2, 3, 4) 
>#没有 einsum的情况下
>z = x @ y.transpose(-2,-1) # batch, sequence,sequence
>#有einsum的情况下
>z = enisum(x,y,"batch seq1 hidden,batch seq2 hidden -> batch seq1 seq2")
>#重复的东西可以用 ... 省略
>z = enisum(x,y,"... seq1 hidden,... seq2 hidden -> ... seq1 seq2")
>```


在做矩阵乘法的时候，我们用 booking的记录来理清 数据之间的关系。


##### eniops_reduce():
reduce 方法可以将数据 通过 sum,mean,max,min 的方法减少一个维度。

>[!example]
>```python
>#old way
>y = x.mean(dim=-1) 
>#new way
>y = reduce(x,"... hidden -> ...","sum")
>```

>[!example]- 执行结果
>```python
>x = [
   [ [1, 2, 3, 4],     [5, 6, 7, 8],     [9, 10, 11, 12] ],   # 第一句话 (batch 0)
   [ [13,14,15,16],    [17,18,19,20],    [21,22,23,24] ]     # 第二句话 (batch 1)
]
>```
>batch =2 seq=3 hidden =4
>执行结果
>```python
>y = [
   [ 2.5,  6.5, 10.5 ],   # 第一句话
   [14.5, 18.5, 22.5 ]    # 第二句话
]
>```

##### einops_rearrange
目前来说最重要的一个方法，这个在 transformer运算的注意力计算中会出现。

>[!note] 具体什么时候会用到呢
>有时候张量的某一维，**其实是由两个（或更多）逻辑维度拼在一起的**，而我们需要把它拆开、单独操作。
>```python
>x: Float[torch.Tensor, "batch seq total_hidden"] = torch.ones(2, 3, 8)
>```
>这个时候 total_hidden 就是两个hidden层合起来的。
>```python
>x = rearrange(x, "... (heads hidden1) -> ... heads hidden1", heads=2)
>```
>这个时候 x就变成 [batch,seq,head,hidden1] 了
>```python
>w: Float[torch.Tensor,"hidden1 hidden2"] = torch.one(4,4)
>```
>执行 transformation
>```python
>x = enisum(x,w,"... hidden1, hidden1 hidden2 -> ... hidden2")
>```
>把 heads 和 hidden2 组合回一起
>```python
>x = rearrange(x,"... heads hidden2 -> ... (heads hidden2)")
>```

#### tensor 的计算开销
FLOP (floating-point operation) 代表一个基本运算。基本运算的含义是，加减和乘除。
FLOPs: 表示总共的 运算量
FLOP/s: 表示每秒进行的 FLOP运算。用来衡量hardware的运算速度。

>[!note] Linear model 的FLOP 运算量是怎么计算的
>设置几个参数
>B  Number of points 数据的数量
>D Dimension of points 点的维度
>K Number of output 输出的数量
>```python
>x = torch.ones(B,D,device=device)
>w = torch.randn(D,K,device = device)
>y = x@w
>```
>在矩阵运算中，我们有 $x[i,j]*w[j,i]$ 的运算。以及乘法的结果累加的运算。
>因此每一对 (i,j,k) 都会有一次加法，一次乘法。
>那么总共个flops就是
>```python
>actual_num_flops = 2*B*D*K
>```

从这里我们可以看到 线性层，如果不包括 加bias的话。 forwar 计算是 $2*参数量*数据量$ 的



>[!Note]- 其他运算的 FLOPs
>element-wise 在$m*n$ 的矩阵上。需要 O(m,n) 的FLOPs
>在 mn 的矩阵上 就需要 mn 次的 FLOPs
>深度学习中常见的操作有 **矩阵乘法，卷积，激活，归一化，dropout**
>矩阵乘法最耗算力，因为激活函数，归一化，element-wise 运算 都是 O(N) 级别，只对每个元素做一次简单操作。
>矩阵乘法是 O(N^3) 的级别 随着矩阵变大 计算量爆炸增长。
>卷积本质上是矩阵乘法的展开特例，因此和矩阵乘法计算成本差不多。
>因此如果矩阵够大，**矩阵乘法是最耗算力的操作**，其它操作的计算开销相比几乎可以忽略。


