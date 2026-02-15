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

# BPE Byte-pair encoding tokenizer

Problem (unicode1): Understanding Unicode (1 point) 
(a) What Unicode character does chr(0) return? Deliverable: A one-sentence response. 
```python
>>> chr(0)
'\x00'
>>> 
```

(b) How does this character’s string representation (__repr__()) differ from its printed representation? Deliverable: A one-sentence response.
```python

```


(c) What happens when this character occurs in text? It may be helpful to play around with the following in your Python interpreter and see if it matches your expectations: ]

```python
>>> chr(0) 
>>> print(chr(0)) 
>>> "this is a test" + chr(0) + "string" 
>>> print("this is a test" + chr(0) + "string")
```



# Implementing the linear module 实施Linear 层
