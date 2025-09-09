# 推理
## Text generation
`generate()` 是一个方法 。
`transformers.generation.utils.GenerationMixin`  中定义
实际流程是
`AutoModelForCausalLM` 返回一个具体类 `GPT2LMHeadModel`。
`GPT2LMHeadModel` 继承了 `GenerationMixin`。
`generate()` 来自 `GenerationMixin`

所有具备生成能力的模型类（比如 `AutoModelForCausalLM`, `AutoModelForSeq2SeqLM`）都会继承这个方法。
`model.generate()` 输入token序列，模型会继续生成新的 token IDs 
输出会是一个 **张量 tensor** 里面是完整的 token序列 
需要再 tokenizer.decode() 才能看到真正的自然语言回答。

相关的是 `model.chat()`  作者在 `trust_remote_code=True` 的时候 给模型对象挂了一个自定义的便捷方法。内部其实还是会调用 `generate()`，只是帮你自动完成 
1. 把 `messages` 用 `apply_chat_template` 
2. 拼成 prompt；插入 `<image>` 标记、
3. 处理 `pixel_values` 
4. 调 `generate()`自动 decode 回文本
（有的还会帮你维护 `history`）
**输出**：  
直接就是模型的回答（字符串），而不是 token 张量。


### Common Options of generate()
可以定义输出长度

Decoding策略
	默认的策略是 greedy search 也就是选最 相似的token。修改 GenerationConfig 允许我们使用不同的策略。

Padding side
	 padding side默认是右边。但是模型是从左往右生成的。那模型生成的上一个 token就会是空的token。每次生成都要“跳过”一堆 `<pad>`，最后一个有效 token的位置不对齐，效率低。
	 可以定义生成的位置