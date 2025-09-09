首先要做的是分词
```
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

encoded_input = tokenizer("Hello, I'm a single sentence!")
print(encoded_input)
```

这里会把输出的文字做分词处理

```
{'input_ids': [101, 8667, 117, 1000, 1045, 1005, 1049, 2235, 17662, 12172, 1012, 102], 
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```
结果是这样
`input_ids` 是 输出的id
`token_type_ids` 这告诉模型 当前的输入哪个是句子A 哪个输入是句子B
'attention_mask'  这个标出了哪个token 应该参与注意力计算哪个不应该。应该是padding的部分

我们可以吧`input_ids`  解码回去看看啥样
```
tokenizer.decode(encoded_input["input_ids"])
```

```
"[CLS] Hello, I'm a single sentence! [SEP]"
```
`[CLS]` （classification token） 放在序列的开头，用来汇总整个序列的信息
`[SEP]` （separator token） 分隔不同句子，或标记序列的结尾
这两个是 **特殊 token**，是 BERT 类模型里专门引入的


## Padding
可以看到这里是没有padding的
padding 用 
`tokenizer(text,padding=True,return_tensors="pt")`
padding = bool 这个参数来控制

padding的部分 会在attention_mask 里面变成0，而不是1。


## Truncating inputs
截断输入
可以通过 
`tokenizer(...,truncation=Ture,max_length=int)`
来截断输入
所获得的输出就是这个
```
{'input_ids': tensor([[  101,  1731,  1132,  1128,   102],
         [  101,  1045,  1005,  1049,   102]]), 
 'token_type_ids': tensor([[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]]), 
 'attention_mask': tensor([[1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1]])}
```

## Adding special tokens
**BERT 类模型** 在预训练时规定了输入格式：
 开头要有 `[CLS]`（代表整句话）
结尾要有 `[SEP]`（表示句子结束或分隔多句）
既然模型在预训练时就是这样学的，那么推理/下游任务时也必须保持一致，否则模型“对不上号”。 
所以 tokenizer 会 **自动帮你加上这些特殊 token**。
```
encoded_input = tokenizer("How are you?")
print(encoded_input["input_ids"])
# [101, 1731, 1132, 1128, 136, 102]
101 → [CLS]

1731 → "How"

1132 → "are"

1128 → "you"

136 → "?"

102 → [SEP]

tokenizer.decode(...) 会把数字翻译回带 [CLS]、[SEP] 的文本。
```

```
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
Tokenize 后：


encoded_sequences = [
    [101, 1045, 1005, 2310, ..., 1012, 102],   # 第一句
    [101, 1045, 5223, 2023, ..., 999, 102]     # 第二句
]
每个句子都是一串 input_ids（整数序列），前后加了 [CLS]、[SEP]。
```