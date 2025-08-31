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

## Mask的主要类型

## Mask的实现细节


