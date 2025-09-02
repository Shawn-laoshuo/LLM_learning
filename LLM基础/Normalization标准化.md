# Normalization是什么
Normalization通过标准化隐藏层的输出，来稳定训练过程。达到避免梯度爆炸和梯度消失，同时加速训练的目的。
# Batch Normalization
## BN的提出
模型的输入是一个batch一个batch输入的，模型每接受一个bathch数据，都会进行参数更新。但是由于bath之间数据分布并不相同，这导致网络各层的输入分布在训练过程中不断发生漂移。 这会让模型的训练变得困难，收敛速度变慢，甚至出现梯度爆照和梯度消失的问题。
# Layer Normalization

# Instance Normalization

# Group Normalization

# RMS Normalization

# pRMS Normalization

# Deep Normalization

# BN & LN & IN & GN 的对比

# Post-LN 与 Pre-LN的对比

