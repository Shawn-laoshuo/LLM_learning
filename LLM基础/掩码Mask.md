## Mask 在Transformer中的意义
1. 防止信息泄露：因为Transformer是自回归的，生成的结果以历史信息为依据。在训练过程中，如果模型能访问未来的token，这就造成的信息泄露，这损害了训练过程。因为数据被提起看到了，那训练的loss就会变得特别小。在推理的时候，模型的效果就会很差
2. 避免无关的信息干扰：我们输入的自然语言数据，通常长度不是一样的，因此我们就需要讲其填充然后送入模型。但是实际上Padding的操作并没有任何有效信息，为了屏蔽无效信息的干扰，就应该把Padding的数据遮盖
3. 训练目标的需要：BERT的训练任务就是双向理解，它也是一个Masked LM（遮挡语言模型），也就是从文本的左右两端结合上下文来预测被Masked的token。那Mask就是必须的了。

## Mask 在Attention 中的作用机理

$$
\mathrm{Attention}(Q,K,V)
=\operatorname{softmax}\!\left(\frac{QK^{\mathsf T}}{\sqrt{d_k}}+M\right)V
$$
是Query矩阵维度是 batch_size , seq_len, $d_{k}$

K是key矩阵维度是 batch_size, seq_len, $d_{k}$

V是value矩阵 维度是 batch_size,seq_len, $d_{k}$

$d_{k}$是key 矩阵的维度。点积结果的数值规模会和维度大小相关。

$M$是Mask矩阵。形状与QK矩阵之积相同

对于需要进行屏蔽的位置，令 $M_{ij}$ = - $\infty$ 经过 softmax 函数之后就会变成一个接近于0的数
比如
softmax的公式是:
softmax里面是一个向量

$$
\operatorname{softmax}(\mathbf{x})_i =
\frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}} ,    i =1,2,\dots,n
$$
当 $x_{i}$ 趋近于负无穷的时候，
$e^{-\infty}$ = 0  也就是说这个分量的softmax结果是0。 当这个位置被mask为0的时候，其他位置会被重新归一化，加起来仍然会等于1

不被mask的部分 $M_{ij}$ =0 正常地参与注意力计算。
## Mask的主要类型
### Padding mask
批量训练的时候，需要将不同长度的序列对齐成相同长度，对多余位置进行padding(多用0或者其他token)。这些位置没有有效信息，为了避免干扰注意力计算，用padding Mask将padding的位置进行屏蔽

#### 计算方式
输入序列有效长度为 $L$ 其他位置是Padding 如果某个token 在索引 $j$ 处是Padding 那么就 让
$M_{ij} =-\infty$  
在构造QK乘积之后 根据输入序列长度来填充Mask矩阵

#### 适用位置
Encoder 自注意力 编码器处理输入序列的时候，屏蔽所有padding
Decoder 自注意力 解码器处理目标序列同样需要屏蔽padding
Cross-Attention 若源序列中有Padding 也需要屏蔽

### Causal Mask
也叫做未来帧屏蔽 或者自回归掩码。解码阶段 解码器需要一次只生成当前的token，不能访问还没生成的未来token，以确报预测是自回归的，符合语言生成的因果顺序。

Look-head Mask一般是一个上三角或则下三角矩阵。对于长度L的序列而言，下标(i,j) 表示第i个token 是否能看到第j个token。
若 $j$ >$i$ 则代表未来位置，需要屏蔽 ，若 $j$ <= $i$ 则代表历史位置，不需要屏蔽。
$$
\begin{bmatrix}
\color{black}{(0,0)} & \color{blue}{(0,1)} & \color{blue}{(0,2)} & \color{blue}{(0,3)} \\
\color{red}{(1,0)} & \color{black}{(1,1)} & \color{blue}{(1,2)} & \color{blue}{(1,3)} \\
\color{red}{(2,0)} & \color{red}{(2,1)} & \color{black}{(2,2)} & \color{blue}{(2,3)} \\
\color{red}{(3,0)} & \color{red}{(3,1)} & \color{red}{(3,2)} & \color{black}{(3,3)}
\end{bmatrix}
$$

可以看到 上三角是0 下三角是-$\infty$ 
蓝色的是需要mask的位置，红色的是不需要被Mask的部分。
### MLM Masked Language Model Mask 
#### 具体做法
在输入序列中替换：将一部分token 替换为[MASK]
Mask 矩阵
损失计算：只对被Mask的部分计算损失，其余位置不会贡献损失。


## Mask的实现细节
1. Shape 对齐：需要保证Mask与注意力分数的形状一致
2. 填充：将需要屏蔽的位置设置成一个非常大的复数。然后在softmax之前 加到注意力分数上，这样在softmax之后，这些位置的分数就会变成一个非常小的数，softmax 之后就会变成一个接近于0的数。

## 代码实现

```
import torch

def create_padding_mask(seq,pad_tokeen=0):
    """
    输入seq:(batch_size,seq_len)
    输出 mask:(batch_size,1,1,seq_len)
    """
    #等于pa d




```