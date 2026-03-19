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

# Implementing the linear module 实施Linear 层
