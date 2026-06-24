This assignment have  four parts，需要实现implement 的内容为
1. Byte-pair encoding (BPE) tokenizer.  字节对编码分词器
2. Transformer Language Model 
3. The cross-entropy loss function and the AdamW optimizer
4. 包括支持序列化 和 loading 模型的 training loop

需要run 的part为：
1. 在TinyStories dataset上训练 BPE tokenizer
2. 运行训练好的BPE tokenizer ，将TinyStories dataset 转换为 整数ID
3. 用LM 生成example 然后评估 perplexity 困惑度
4. 在OpenWeb 数据集上训练模型 然后评估perplexity 困惑度(PPL)

# 2 BPE Byte-pair encoding tokenizer
## 2.1 The Unicode Standard   （Unicode 标准）
Unicode 是一个 用于将 文本 映射(map) 成`code point` (integer格式) 的 text encoding standard (文本编码标准)。 
>[!note] Unicode
>	截止到2024年9月，Unicode 已经更新到 Unicode 16. 这个标准定义了来自 168个字符体系的 154,998 个字符。例如 字母`s` 在 Unicode 标准中 的 `code point` 是115。 115 经常会被标记为 `U+0073` 。 其中`U+` 是   一个惯例前缀，0073 是115 的十六进制表示(hex)。
>	在Python中`ord()` 函数，可以将Unicode的字符转换为 整数表示。`chr()` 函数会反过来，将整数转换为字符



Problem (unicode1): Understanding Unicode (1 point) 
(a) What Unicode character does chr(0) return? Deliverable: A one-sentence response. 
```python
>>> chr(0)
'\x00'
>>> 
```
`chr() `的作用是把 整数(Unicode码点)转换为 字符。
`chr(0)` 对应 Unicode 码点 0，也就是 空字符（null character），写作 `\x00 `或 \0。
它是不可见的，打印出来什么都看不到，但它确实是一个字符
>[!notice]- \x  是怎么回事
为什么是 `\x00` 呢 
因为这是一个十六进制表示法 hex
\x 是 Python（以及 C、Java 等大多数语言）里的转义前缀，意思是："后面跟的是十六进制数字"。
>同样的 还有\n 换行 \t Tab 键。等情况

(b) How does this character’s string representation (__repr__()) differ from its printed representation? Deliverable: A one-sentence response.
```python
>>> print(chr(0))

>>> repr(chr(0))
"'\\x00'"
```
上面说过了 0 这个码点 对应的是空字符 null character。 
`print`直接输出字符的原始字节，空字符` \x00 `是不可见字符，终端不会渲染它，所以你什么都看不到（但它确实输出了）

而 `repr() `的目标是让你看清楚这个对象到底是什么，所以它会把不可见字符转义成可读形式。
也就是显示可读表示。
>[!AI]- 从底层原理理解 `print` 与 `repr` 的本质（AI知识卡片）
>在 `CPython` 中，`print` 与 `repr` 的分水岭在于它们调用了不同的对象协议函数。
>
>1. 核心哲学：谁是受众？
>	- `print()`（基于 `__str__`）：面向用户（User-friendly）。人来阅读
>	  目标是“可读性”，强调字符的功能（换行、响铃、退格等）。
>	- `repr()`（基于 `__repr__`）：面向开发者（Debug-friendly）。对象 _official_ 的字符串表示
>	  目标是“无歧义性（Unambiguous）”，强调对象的本体表示。
>	  理想标准：`eval(repr(obj)) == obj`
>
>2. CPython 调用路径
>	- `print(obj)` → `PyObject_Str(obj)`
>	  若未定义 `__str__`，则回退调用 `__repr__`
>	  最终将得到的字符串编码（如 UTF-8）为字节流
>	  → 交给底层 `write()` 写入 `stdout`
>
>	- `repr(obj)` → `PyObject_Repr(obj)`
>	  底层 C 函数（如 `unicode_repr`）逐字符扫描
>	  若发现不可打印字符（如 U+0000）
>	  → 生成字符串字面量（如 `'\\x00'`）
>	  → 再写入 `stdout`
>
>3. 数据流向差异（以 `\n` 为例）
>	- 内存中：U+000A（换行符）
>
>	- `print` 路径：
>	  U+000A → UTF-8 编码 → 0x0A → `write(fd, "\n", 1)`
>	  → 终端解释为“执行换行动作”
>
>	- `repr` 路径：
>	  U+000A → 判断为不可见字符
>	  → 生成 `'\\n'`
>	  → `write(fd, "\\n", 2)`
>	  → 终端显示字符 `\` 和 `n`
>
>4. 本质区别
>	- `print` 关注字符的功能（执行）
>	- `repr` 关注字符的本体（表征）
>
>5. 小实验
>	- `python3 -c "print('\x07')"` → 你会听到蜂鸣，但看不到字符
>	- `python3 -c "print(repr('\x07'))"` → 你会看到 `'\\x07'`，但不会听到声音


(c) What happens when this character occurs in text? It may be helpful to play around with the following in your Python interpreter and see if it matches your expectations: ]

```python
>>> chr(0) 
>>> print(chr(0)) 
>>> "this is a test" + chr(0) + "string" 
>>> print("this is a test" + chr(0) + "string")
```

## 2.2  Unicode encodings

## 2.5 Experiment with BPE Tokenizer Training
要实现train_bpe.py 这个文件。
```python
def train_bpe(
      input_path: str,
      vocab_size: int,
      special_tokens: list[str]
  ) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]
```

`input_path` 是训练语料的路径。 
`vocab_size` 表示最终需要的词表的大小。 这个size必须是要远大于256的。 因为有初始的256个byte token
>[!note]- 关于vocab_size 的问题 256这个数字是怎么来的
>下面可以看到 `b\x00`, `b\x01` 等 
>`\x00` 是十六进制的 00  十进制的0
> `\x01` 是十六进制的 01  十进制的1
> 而python里面的 b''  表示bytes 也就是字节序列。一个字节的取值范围
> 一个字节 byte 是有8个位置的，也就是能存储8个数字。
> 所以 11111111 最大的数字对应十进制是255.
> 一个byte 是8个bits 
> 一个bits 只能取 0 或者1 也就是 高电平或者低电平 
> 然后8个bits 就是 一个bytes


`special_tokens` 特殊token的字符串列表。作用是把这几个token 直接加入词库，另外是在训练前用他们先把语料切开。也就是 pre-tokenization.

返回的形式大概是
`vocab:dict[int,bytes]` token id 到 字符串映射

`merges:list[tuple[bytes,bytes]]`

```json
  (
    {0: b'\x00', 1: b'\x01', ..., 256: b'<|endoftext|>',257: b'th', ...},
    [(b't', b'h'), (b'th', b'e'), ...]
  )
```

`merge` 
```json
(b`h`,b'e')
意思是把 h和e merge成一个新的token b和h
```

整体的流程就是
找到出现频率最多的两个字符 然后把他俩合并称一个，也就是放到merge里面。
同时把`new_token` 放到vocab 词表里面
进行多次，完成这个步骤。


# 3 Transformer Language Model Architecture
language model 用 一组整数ID作为输入。比如 `[1,2,3,4,5]`。格式大概是 `[batch_size,sequece_length]`。 第一个维度是句子的数量，第二个维度是句子的长度。 这个整数是tokenizor层的结果了。

输出 一个 或者一个 batch的 normalized probability distribution。

在训练语言模型的过程中，我们计算生成的文本和实际文本之间的 cross-entropy loss。以此来进行梯度下降。

	在推理的过程中，我们会从 sequence中 得到 next-word distribution，也就是 下一个文字的分布情况。然后生成 下一个文字的token。 循环往复，得到语言模型的输出结果。

## 3.1 Transformer LM
### 3.1.1 Token Embeddings
经过Embedding 层， token从整数，经历一个查表的过程，被转换成离散的数据。
数据的格式也就是从
`[batch_size,sequece_length]` 转换为 `[batch_size,sequence_length,d_model]`
其中 `batch_size` 自然就是 句子的数量，而`sequence_length` 是每句中所包含的token数量。最后 `d_model`  是每个token所转换成的维度的 `embedding_dim` 的向量。

### 3.1.2 Pre-norm Transformer Block
pre-norm 风格的 Transformer Block.
embedding 做之后，拿到了一堆向量。这些向量都是 `[batch_size,sequence_length,d_model]` 长这样的。 输入和输出长得一样，但是实际上他们被 normalization 了。

## 3.2 Output Normalization and Embedding
在经过多个 transformer block的变换之后，获得的activation 激活值，会被转换成一个概率分布。 这个在这个概率分布抽样后会得到的我们需要的token。

在transformer中，用的一般都是 pre-norm 也就是数据在被计算之前，先把数据标准化，然后进行计算。

## 3.3 Remark: Batching, Einsum and Efficient Computation
Enisum 用来做矩阵乘法，这样我们不需要复杂的转置，通过矩阵的notation 标记，也就是这个维度代表什么数据，就可以免去复杂的转置计算。

## 3.4 Basic Building Blocks: Linear and Embedding Modules
实施 线性层和 embedding层。

### 3.4.1 Parameter Initialization
参数初始化。初始化的方式会很大程度上影响模型的训练效果，在这次的作业中，我们的初始化方式为

- Linear weights: $\mathcal{N}\left(\mu = 0, \sigma^2 = \frac{2}{d_{\text{in}} + d_{\text{out}}}\right)$ truncated at $[-3\sigma, 3\sigma]$.

- Embedding: $\mathcal{N}\left(\mu = 0, \sigma^2 = 1\right)$ truncated at $[-3, 3]$.

- RMSNorm: $1$
### 3.4.2 Linear Module
Linear Module
这个类 继承自
`class Linear(nn.Module):`
这个`nn.Module` 类 初始化 weight
在 `forward` 方法中，输入的数据 x乘上 `self.weight`

### 3.4.3 Embedding Module
embedding module 的用处，简单来说就是查表
讲 `[batch_size,sequence_length]` 转换成 `[batch_size,sequence_size,d_model]` 这个格式。

大概流程是 从 embedding matrix（`self.weight`）这个大表上，通过 token id 来查询。



## 3.4 Pre-Norm Transformer Block
有人发现把 norm的操作转移到从输出层之前(post-norm)转移到 the input of each sub-layer，效果会比较好一些。
现在这个作业要implement 的就是 pre-norm Transformer。
### 3.4.1 Root Mean Square Layer Normalization
也就是  `RMSNorm`  公式为：
$\mathrm{RMSNorm}(a_i) = \frac{a_i}{\mathrm{RMS}(a)} g_i$
其中gi 是一个可学习的参数。

### 3.4.2 Position-Wise Feed-Forward Network

传统Transformer的结构，是两层 线性层中间夹着一个 ReLu层。
线性层长这样：
$y = Wx + b$
中间夹着一个ReLu

也就是 第一层
$h1 = W_{1}x+b_{1}$
中间夹着一个ReLu
$\mathrm{ReLU}(x) = \max(0, x)$

>[!AI] Latex语法 \mathrm
>在数学公式里，用“正文直立字体”来显示内容，而不是默认的斜体变量字体 \mathrm 
>\cdot LaTeX 里的**乘号（·）**，表示“点乘 / 普通乘法”， centered dot（居中的点
>在Latex规则中，公式是正体 用Romas字体，变量是斜体 italic


那最后的公式就是
$\mathrm{FFN}(x) = W_2 \cdot \mathrm{ReLU}(W_1 x + b_1) +b_2$

这是在 Transformer 原始论文中的做法，现代的大模型 一般会使用  **SwiGLU** activation function  并且配合 Gate Linear Unit 也就是用 门控机制来做到这些。
$$
\mathrm{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1+e^{-x}}
$$
$\sigma(x)$ 这里 是一个$\mathrm{Sigmod}$ 函数，这个激活函数 就是 $x$ 再乘上一个Sigmod

>[!notice]- Sigmoid 函数探秘
>Sigmoid 函数是法国数学家 让皮埃尔 在描述人口增长模式是使用的一个函数。 初期人口增长十分迅速，中期呈指数型增长，后期增长缓慢。 该函数因其优秀的性质，在早期深度学习中被 Hinton Leecun 等机器学习始祖们广泛使用。 该函数在输入值很大的时候，值接近1，在输入很小的时候，值接近0。它的性质就很适合做 Gate 门控机制，比如想让比重是0.8 那就输入一个较大的值，想让比重是0.99 就输入一个特别大的值，反之亦然。但是由于Sigmod 多次求导之后，输入的函数值会变得极大或及小，这可能会造成 梯度下降或者梯度消失，因此在深度学习中 ReLU 会更常用一些。 Sigmoid 函数的导数是 $\sigma'(x)= \sigma(x)(1-\sigma(x))$   也就是σ(x)=0.8 σ′(x)=0.8(1−0.8)=0.16 越求导数字越小 当时数字很大的时候。越求导数字还越大。 
>这个函数的导师也特别有意思，它很优雅。 因为它能自己表示自己，这就是很优雅。

SiLU 是 输入x 乘上 Sigmoid，多了这个乘法，Sigmoid函数就变成一个门控机制了。
SiLU 相对于ReLU函数大概长这样
![[Pasted image 20260507153833.png]]

从函数性质上来看，SiLU 和ReLU 长得差不多，但是SiLU在 原点0 附近 更加丝滑，不像 ReLU 在零点处很尖锐。 尖锐这个性质，说明ReLU 在零点处是 not differentiable 的  还是不可微分(having a differential)的。 忘记不可微分和不可导的区别了。

>[!notice] 深度学习中 为什么不区分可微分和可导。
>首先 可微 having a differential 的含义是 函数局部足够光滑，可以被线性近似。核心含义是可以被线性近似
>	例如在一元函数里 $f(x+\Delta(x)) = f(x) + A\Delta(x)+ o(\Delta(x))$ A 就是微分, $A\Delta(x)$ 是线性主部， $o(\Delta(x))$  部分是误差。  如果一个函数处处都是光滑的 那它就是可微的。
>	
>而可导的含义就是 可以求出导数。更严格一点来说 就是这个函数在某一点的导数存在。
>	可导的定义是这个极限存在，核心关注点 是变化率有没有稳定下来(缩小区间 观察平均变化率是不是稳定在一个值)
>	 $$f'(x) = \lim_{\Delta x \to0} \frac{f(x+\Delta x)- f(x)}{\Delta x}$$
>在一元函数里 函数满足 $f(x + \Delta x)- f(x) = A \Delta x + o(\Delta x)$
>然后这个函数除上 $\Delta x$ 结果就是 $A + \frac{o(\Delta x)}{\Delta x}$  且后面这部分 $\frac{o(\Delta x)}{\Delta x}$ 是趋近于0 的。 那 $f'(x)$ 的值 就是 趋近于 A的。 所以在一元函数 中 可微和可导是等价的。

最后经过Gated Linear Unit 线性门口单元 以及SiLu的处理，大模型的SwiGLU 的最终形态为
GLU 通常是 $x$ 的线性变换 经过 $\mathrm{Sigmoid}$ 函数处理后。然后和另外一个线性层 做element-wise 乘积。

$$
\mathrm{FFN(x)} = \mathrm{SwiGLU(x,W_{1},W_{2},W_{3})} = \mathrm{W_{2}(SiLU(W_{1})\odot W_{3}\mathrm{x}) }
$$
### 3.4.3 Relative Positional Embeddings
#### RoPE
相对位置编码 
目的在于 在Attention 机制计算相关性的时候，能感知token的相对距离。

RNN模型是根据时间步串行处理序列，天然携带词语的先后顺序信息。但是Transformer 的 self-attention机制是对各token进行相关性的并行计算。单纯的attention score 只是token之间的匹配程度，并不包含token的绝对位置或相对距离。因此transformer 需要通过额外引入位置编码来表示信息。

比如在计算 
```
我 去 银行 办 业务的时候
```
在注意力机制里每个token都生成自己的 $q$ $k$ $v$
也就是对于第i个个 token 的输入向量$x_{i}$ 他们会经过三个线性投影矩阵:
$$
\begin{aligned}
q_{i} = W_q x_i \\
k_i = W_k x_i \\
v_i = W_v x_i
\end{aligned}
$$

$$
\begin{aligned}
q_{\text{我}},\ k_{\text{我}},\ v_{\text{我}} \\
q_{\text{今天}},\ k_{\text{今天}},\ v_{\text{今天}} \\
q_{\text{银行}},\ k_{\text{银行}},\ v_{\text{银行}}
\end{aligned}
$$

每个token 里面都有自己的 query key  value。

每一个token都会用自己的 $q$  和所有 $k$ 和 $v$  做匹配，然后得到注意力权重
$$
\mathrm{score_{ij}} = q_{i}^T k_{j} 
$$

$$
a_{ij} = softmax(score_{ij})
$$

过一遍softmax 函数之后得到注意力权重，最后用这些权重对所有token 的 $v_{j}$  加权求和。
$$
o_{i} = \sum_j \alpha_{ij} v_j
$$

$o_{i}$ 是第i个token 经过Attention计算之后得到的输出向量。原始的Attention机制计算了token直接的相关性，再根据注意力权重 $v_{ij}$ 加权求和。在这个过程中我们没有显式包含token的位置信息 ij 或者相对距离 i-j  因此我们原始的Attention计算本身不能感知到token直接的相对位置关系。

最直接的位置编码方式是绝对位置编码。它告诉模型每个 token 在序列中的绝对位置，例如第 1 个位置、第 2 个位置、第 3 个位置。

但是在语言建模中，很多关系并不只依赖绝对位置，而更依赖两个 token 之间的相对距离。例如：
$$
[  
\text{我 去 银行}  
]
$$
和：
$$
[  
\text{今天 上午 我 去 银行}  
]
$$
在这两个句子中，“去”和“银行”的绝对位置发生了变化，但是它们的相对关系基本没有变化：银行都在“去”的后面，并且距离很近。

因此，相对位置编码关注的不是“某个 token 是第几个”，而是“两个 token 之间相隔多远、谁在谁前面”。也就是说，它希望 attention 在计算 token 相关性时，不仅考虑内容相关性：
$$
[  
q_i^\top k_j  
]
$$
还能够感知相对位置信息：
$$
[  
j-i  
]
$$
这就是引入相对位置编码的动机。

RoPE，即 Rotary Position Embedding，是一种相对位置编码方法。它不是直接给 token embedding 加一个位置向量，而是在计算 attention score 之前，对 (q) 和 (k) 做位置相关的旋转。

对于第 (i) 个 token，先正常计算 query 和 key：
$$

q_i = W_q x_i  

$$
$$
k_i = W_k x_i  
$$

然后 RoPE 根据 token 的位置 (i)，对 (q_i) 和 (k_i) 做旋转：

$$
q_i' = R^i q_i  
$$

$$  
k_i' = R^i k_i  
$$

之后 attention score 不再使用原始的 (q_i) 和 (k_j)，而是使用旋转后的 (q_i') 和 (k_j')：

$$  
\text{score}_{ij}=(q_i')^\top k_j'  
$$

也就是：

$$  
\text{score}_{ij}=(R^i q_i)^\top(R^j k_j)  
$$

由于第 (i) 个 token 和第 (j) 个 token 使用的旋转角度不同，它们做点积时会受到旋转角度差的影响。而这个角度差与相对距离 (j-i) 有关。因此，RoPE 可以把相对位置信息注入到 attention score 中。

那么RoPE 如何对向量做旋转？

RoPE 不是把整个 (d) 维向量一次性旋转，而是把向量每两个维度分成一组，每一组看作一个二维向量进行旋转。

假设某个 query 向量是 6 维：

$$  
q_i = [q_1,q_2,q_3,q_4,q_5,q_6]  
$$

RoPE 会把它分成三组：

$$  
(q_1,q_2),\quad(q_3,q_4),\quad(q_5,q_6)  
$$

每一组都看成一个二维向量。对于第 (k) 组二维向量：

$$  
(q_{2k-1}, q_{2k})  
$$

RoPE 会使用角度 $\theta_{i,k}$ 对它进行旋转： 也就是 $R_{k}^i$ 
$$R_{k}^{i} = 
\begin{pmatrix} 
\cos\theta_{i,k} & -\sin\theta_{i,k} \\ 

\sin\theta_{i,k} & \cos\theta_{i,k} 
\end{pmatrix}
$$

这里的旋转角度$\theta$不是随便设定的，而是由 token 位置 $i$ 和维度组编号 $k$ 决定：
$$  
\theta_{i,k} = \frac{i}{\Theta^{(2k-2)/d}}  
$$
i是token位置 k是维度组编号。
比如q是6维，那么分成三组 k=1 k=2 k=3 三组。


然后后面的计算就是用第i个位置 第k组的小旋转矩阵 $R_{k}^{i}$ 去旋转去旋转 query 向量里的第 kkk 组二维向量。
$$
\begin{pmatrix}
q'_{2k-1}\\
q'_{2k}
\end{pmatrix}
=
R_{k}^{i} 
\begin{pmatrix}
q_{2k-1}\\
q_{2k}
\end{pmatrix}
$$
$$
\begin{pmatrix}
q'_{2k-1}\\
q'_{2k}
\end{pmatrix}
=
\begin{pmatrix}
\cos\theta_{i,k} & -\sin\theta_{i,k}\\
\sin\theta_{i,k} & \cos\theta_{i,k}
\end{pmatrix}
\begin{pmatrix}
q_{2k-1}\\
q_{2k}
\end{pmatrix}
$$

展开后就是：

$$
q'_{2k-1}
=
q_{2k-1}\cos\theta_{i,k}
-
q_{2k}\sin\theta_{i,k}
$$

$$
q'_{2k}
=
q_{2k-1}\sin\theta_{i,k}
+
q_{2k}\cos\theta_{i,k}
$$

其中，(i) 表示 token 的位置，(k) 表示第几组二维向量。因此，不同位置、不同维度组会使用不同的旋转角度。

$R_i$ 最后被展示成这样
$$
R^i =
\begin{pmatrix}
R_1^i & 0 & 0 & \cdots & 0 \\
0 & R_2^i & 0 & \cdots & 0 \\
0 & 0 & R_3^i & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & R_{d/2}^i
\end{pmatrix}
$$

Ri 是一个特殊的对角矩阵，**实际实现时不要真的构造这个矩阵**。 
整个q' k'是被动态计算出来的，并不需要构造整改 $R_i$ 

### 3.4.4 Scaled Dot - Product Attention  点积注意力

>[!notice]- 分布
>分布的定义就是把 100% 的可能性（或者说数字 1），分摊给几个不同的选项
>一个合格的概率分布，必须满足两个铁律：
>非负性：每个选项分到的概率必须在 $0$ 到 $1$ 之间（即 0% 到 100%）。你不能说“这图有 -20% 的概率是只猫”。
>归一化（Normalized）**：所有选项的概率加起来，必须**完完全全等于 1（即 100%）。
>Softmax 的作用就是把输出的原始分数通常叫Logits ，归一化为概率分布

#### Masking
在注意力计算中，本来每个token的q 会和所有token 的k 算相关性分数，然后用这些分数加权汇总对应的v。
但是我们的任务是 根据前面已知的token 预测下一个token 或者 根据两侧已知的token 预测中间位置的未知token。因此 模型就不能看到所有token 也就是不能把所有token 都用来做注意力计算，否则需要被预测的token就被泄露出来了。
为了达到训练模型预测token 的目的，某些token 需要被mask。
Masking 这个机制是让某些 query 不允许看某些 key。但是我们又不能把这个 token删掉，因此我们用Masking 机制。
Mask 本质上是一个布尔矩阵：
$$
M \in \{\text{True},\text{False}\}^{n \times m}
$$
其中$M_{ij}$  是布尔值，表示 第i个 query 能否看到 j。 True 是允许注意，False 是不允许注意。
在计算 Attention 时，首先会得到 Attention Score：
$$
S = \frac{QK^\top}{\sqrt{d_k}}
$$

假设某一行分数为 
$$
[5, 2, 7]
$$
如果对应的Mask 是
$$
[True, True, False]
$$
 那就需要把False 位置的分数改成 $-\infty$

然后得到的分数就是
$$
[5,2,-\infty]
$$

然后再执行 softmax

$$
\mathrm{softmax}([5,2,-\infty])
$$

由于
$$
e^{-\infty} = 0
$$
结果是 
$$
[0.95,0.05,0]
$$
出于False 的 Attention 计算结果就是0了。


### 3.4.5 Causal Multi-Head Self-Attention  带因果 Mask 的多头自注意力
单头注意力，就是 token 和token 之间做注意力计算。每个 token 用一整条向量去做一次 attention。
而多头注意力，是把每个token拆成多个向量子空间，每个head 在自己的子空间里做 attention。
例如
$$
Q = \left[
\begin{array}{cccccccc}
q_{11} & q_{12} & q_{13} & q_{14} & q_{15} & q_{16} & q_{17} & q_{18} \\
q_{21} & q_{22} & q_{23} & q_{24} & q_{25} & q_{26} & q_{27} & q_{28} \\
q_{31} & q_{32} & q_{33} & q_{34} & q_{35} & q_{36} & q_{37} & q_{38} \\
q_{41} & q_{42} & q_{43} & q_{44} & q_{45} & q_{46} & q_{47} & q_{48}
\end{array}
\right]
\begin{array}{l}
\leftarrow \text{token 1} \\
\leftarrow \text{token 2} \\
\leftarrow \text{token 3} \\
\leftarrow \text{token 4}
\end{array}
$$
```
[q11, q12, q13, q14 | q15, q16, q17, q18]
        head 1              head 2
```
然后可以切分成两个 head。前四个 给个head1 后四个给了head2。

K 和V 同样要 切不同的head。
每个 head 内部，使用自己的 Qi,Ki,ViQ_i,K_i,V_i单独做 attention 计算。
**head 之间不互相做 attention**。 不是 head 1 attend to head 2，而是每个 head 在自己的子空间里，对所有 token 做 self-attention。最后，把所有 head 的输出拼接起来：$\mathrm{Concat}(head1,head2)$
所以 self-attention 相对于 multi-head attention 是没有复杂度上的变化的。

算完多个 head的attention之后，再乘上一个$W_{O}$，把多个 head 拼接后的结果再融合一次，得到最终输出。
最后的公式就是
$$
\operatorname{MultiHeadSelfAttention}(x)
=
W_O \operatorname{MultiHead}(W_Q x,\; W_K x,\; W_V x)
$$


## 3.5 The full Transformer LM
大模型不仅有一层Transformer，而是多个 Transformer block 叠起来做到的。
Transformer block 包括 两个'sub-layer'  
第一个 sub-layer 是 MHA 多头注意力计算模块
第二个 sub-layer 是 SwiGLU feed-forward 计算模块
每个block 的职责就是 过着两个模块。 先做 attention 计算 每个token直接相互看，然后再对每个token做非线性变换(SwiGLU Feed-Forward Network 激活函数)。 
Residual Add 的含义就是这个动作
$$
y = x + \text{MultiHeadSelfAttention}(\text{RMSNorm}(x))
$$
输入的值 加上attention的结果 这叫残差连接 防止梯度消失或梯度下降。
> [!note]- 为什么叫残差连接呢
> **残差（Residual）的含义**：在数学中，残差通常指“观测值与估计值之差”。
> **连接的逻辑**：在 Transformer 中，我们将层输入 $x$ 看作“基准信息”。我们让注意力层只去计算 $x$ 需要**修正、补充或提取的细节**（即那个“残差部分”：$\text{MultiHeadSelfAttention}(\text{RMSNorm}(x))$）。
> **叠加**：最后通过 $y = x + \text{残差}$，模型输出的就是“原始信息 + 修正部分”。这就像是给原始信息做了一次“精修”和“润色”。


# 4 Training a Transformer LM

Transformer的训练要素
Loss Function 这里选用 Cross-entropy
Optimizer  优化器 这里介绍了AdamW
Training Loop 我们需要工具来 load data ，save check points， manage training。这里就用Pytorch 自带的组件来实现了。


## 4.1 Cross-entropy loss
核心思想在于，LM一直在猜下一个词，猜中的概率越高(给Ground Truth 分配的概率越高) Loss越小。
猜中的概率越低(给Ground Truth 分配的概率越低) Loss 越大。

假设训练集里面只有一句话
$$
[我,爱,北京]
$$
Tokenizer 之后会变成一串。数字，这个是根据词表来的。
$$
[101,102,103]
$$
语言模型在训练的时候，就是根据前面的词预测下一个词,
也就是 

|输入|目标|
|---|---|
|我|爱|
|我 爱|北京|


公式描述也就是
$$
\log p_{\theta} (x_{i+1}|x_{1:i})
$$
