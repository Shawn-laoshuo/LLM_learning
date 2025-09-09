
# Transformer 类的特点
1. 低代码调用
2. 所有模型都是基于 Pytorch 的 `nn.Module` 类调用的
3. 没有跨文件的抽象。代码容易理解
[[]]


# Transformer类通用的处理数据的流程

## Preprocessing with a tokenizer 分词
`from transformers import AutoModel`
`from transformers import AutoTokenizer`
`from transformers import AutoModelForSequenceClassification`
NLP任务的输出数据是一段文本。
第一步要做的事是分词，也是把词切成子词 用到的是 Transformers 类里面的 `AutoTokenizer` 
```
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```
`AutoTokenizer.from_pretrained(checkpoint)` 这个代码加载了分词模型


```
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)
```
padding 填充， return_tensors 这个参数写的是 pt 意思是pytorch

结果是这样的：

```
{
    'input_ids': tensor([
        [  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172, 2607,  2026,  2878,  2166,  1012,   102],
        [  101,  1045,  5223,  2023,  2061,  2172,   999,   102,     0,     0,     0,     0,     0,     0,     0,     0]
    ]), 
    'attention_mask': tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
}
```

- `input_ids`：这是 **分词 + 映射成整数 id** 的结果。
- `attention_mask`：告诉模型哪些位置要计算，哪些是填充。
所以分词其实是是将文本切成小块，然后用index表示。首先把原始文本切分成更小的单元，可以是词，子词，字符。然后映射成索引。每个token在 词表中都有一个编号
为什么要用索引呢？因为神经网络不能处理字符串。用整数索引（id）后，模型就能通过 **embedding matrix** 把它转成向量（模型真正能理解的形式）。

因此 **分词 = 切小块（tokenize）+ 用索引号表示（numericalize）**。
[[]]

## 模型内部流程

```
from transformers import AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)
```
这一步是在加载模型，`AutoModel` 是一个万能入口，会根据 checkpoint自动选择 合适的底层类，比如BertModel GPT 等
然后返回一个模型对象。
里面包含 Transformer **base**（embedding 层 + 多层 self-attention + feed-forward），输出 hidden states。
**注意：这里没有任务 head。** 后面  AutoModelForSequenceClassification 这有个分类头。

![[Pasted image 20250909135532.png]]
没有任务head的意思是 AutoModel 加载的模型 只输出到 hidden state 这一步。没有head，head通常是 Linear 层+激活函数。

```
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)
```
```
torch.Size([2, 16, 768])
```
可以看到输出是这个 
batch = 2 
sequence  length = 16
Hidden size = 768
返回是的一个高维向量，因为最后一个值是 768维的。

如果加上分类头，效果就是：

```
from transformers import AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)
```

```
print(outputs.logits.shape)
```

```
torch.Size([2, 2])
```
可以看到 有了head 后面输出的就是 一个logit
```
tensor([[-1.5607,  1.6123],
        [ 4.1692, -3.3464]], grad_fn=<AddmmBackward>)
```

logit 后面经过softmax 会变成预测的概率

```
import torch

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)


tensor([[4.0195e-02, 9.5980e-01],
        [9.9946e-01, 5.4418e-04]], grad_fn=<SoftmaxBackward>)
```
`model.config.id2label` 存这模型配置
id2label 写着 模型的数值输出映射为标签的方式。
[[Transformer Encoding Text]]

## 模型使用
### loading and saving 模型

```
model.save_pretrained("directory_on_my_computer")
```
这个方法会把 **模型的权重参数** 和 **配置文件** 一起保存到指定目录里。
一般有
`pytorch_model.bin`  PyTorch 权重
`config.json`配置文件（包含 hidden_size, num_labels, id2label 等信息）
`tokenizer_config.json`, `vocab.txt`, `merges.txt`（如果你同时保存 tokenizer）

加载一个 saved 模型 用 from_pretrained() 方法

```
from transformers import AutoModel

model = AutoModel.from_pretrained("directory_on_my_computer")
```
还可以直接推送到 hugging face hub
```
from huggingface_hub import notebook_login

notebook_login()

model.push_to_hub("my-awesome-model")
```
用 push_to_hub 的方法

```
from transformers import AutoModel
model = AutoModel.from_pretrained("your-username/my-awesome-model")
```
