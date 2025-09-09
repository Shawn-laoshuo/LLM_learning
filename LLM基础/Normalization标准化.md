# Normalization是什么

Normalization通过标准化隐藏层的输出，来稳定训练过程。达到避免梯度爆炸和梯度消失，同时加速训练的目的。
# Batch Normalization
## BN的提出
模型的输入是一个batch一个batch输入的，模型每接受一个bathch数据，都会进行参数更新。但是由于bath之间数据分布并不相同，这导致网络各层的输入分布在训练过程中不断发生漂移。 这会让模型的训练变得困难，收敛速度变慢，甚至出现梯度爆照和梯度消失的问题。
=======
通过对激活值和特征张量进行规范化处理，达到缓解梯度消失，加速模型收敛，增强模型泛化能力的效果。
# Batch Normalization
在模型训练过程中，一般都是采用mini-batch的方式进行训练，而Batch Normalization就是对每个mini-batch进行归一化处理。
因为每个batch 的数据分布不同，这可能会导致网络各层输入分布在训练过程中不断漂移。这会使模型训练变得困难，甚至出现梯度消失和梯度爆炸的情况。

BN的核心思想就是归一化。具体步骤是：
1. **在激活函数之前**，对每个通道，统计该通道在一个 batch 内所有样本和空间位置的均值与方差，并将这些数据归一化为均值0、方差1的分布。
2. 再加上可学习的缩放因子$\gamma$和偏移量$\beta$，用于恢复部分信息。

## BN的主要思路和算法过程
MLP的输出维度一般是 [N,C,W,H]。分别代表 batchsize，channel，weight，High。
我们的目的是在channel维度上进行归一化。也就说把batch维度，weight维度，high维度加起来，计算均值

计算均值
$$
\mu_c = \frac{1}{N\times H \times W} \sum_{n=1}^{N} \sum_{h=1}^{H} \sum_{w=1}^{W} x_{n, c, h, w}
$$
计算方差
$$
\sigma_c^2 = \frac{1}{N\times H \times W} \sum_{n=1}^{N} \sum_{h=1}^{H} \sum_{w=1}^{W} (x_{n, c, h, w} - \mu_c)^2
$$

归一化
$$
\hat{x}_{n, c, h, w} = \frac{x_{n, c, h, w} - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}}
$$

缩放和平移
$$
y_{n, c, h, w} = \gamma_c \hat{x}_{n, c, h, w} + \beta_c
$$

其中 $\gamma_c$ 和$\beta_c$ 是可学习的参数，用于自动调整模型分布。
>>>>>>> origin/main
# Layer Normalization
在输入长度固定的时候，自然可以使用BN进行归一化。但是在NLP任务中，输入长度是不固定的，即使用了padding，有效信息的位置也是不一样的。此时这个时候BN就不再适用了。因为BN是在计算channel方向的均值和方差来进行调整，但是当长度不一样，有padding的时候。这个计算就没有意义了。

## LN的计算公式：
$$
y = \frac{x - \operatorname{E}(x)}{\sqrt{\operatorname{Var}(x) + \epsilon}} \cdot \gamma + \beta
$$

$\operatorname{E}(x)$ 是输入 $x$ 的期望，$\operatorname{Var}(x)$ 是输入 $x$ 的方差，$\epsilon$ 是一个小的常数，用于避免除零错误。
$\gamma$ 和 $\beta$ 是可学习的参数，用于自动调整模型分布。

## LN的特点
1. LN的优势在于LN不依赖于Batch，适用于batch size很小或者 batchsize为1的情况。

2. 对序列建模效果明显

3. 在CNN中LN的效果弱于BN。





# Instance Normalization

# Group Normalization

# RMS Normalization

# pRMS Normalization

# Deep Normalization

# BN & LN & IN & GN 的对比

# Post-LN 与 Pre-LN的对比

