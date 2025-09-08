# Chat models
这里没啥可说的，就是配置不同模型，值得注意的是 可以配置Hugging face上的模型。应该能把hugging face上的模型跑在本地 然后用langchain调用

# Prompt Templates

这个功能会把 user input 和 prompt 一起送进 模型。

在官方的示例中

```
from langchain_core.prompts import ChatPromptTemplate  
  
system_template = "Translate the following from English into {language}"  
  
prompt_template = ChatPromptTemplate.from_messages(  
[("system", system_template), ("user", "{text}")]  
)
```

## `langchain_core` 和  `langchain` 的区别
 `langchain_core`
- **内容**：
    - **基础抽象类和接口**：`Runnable`、`BaseChatModel`、`BaseRetriever`、`BaseTool`
    - **消息和文档的数据结构**：`HumanMessage`、`AIMessage`、`Document`
    - **Prompt 模板相关**：`PromptTemplate`、`ChatPromptTemplate`
- **特点**：
    - **不包含任何具体集成**（比如 OpenAI、FAISS、Slack 这些都没有）。
    - 体积小，稳定，专注于 **定义统一接口**。
    - 让其他模块（`langchain`, `langchain-community`, `langchain-openai`）基于它来实现功能。

langchian 
- **定位**：LangChain 的 **应用层主包**。
- **内容**：
    - 内置一些常用 **Chain / Agent / Utility**。
    - 提供对 `langchain-core` 的封装，让开发者更容易上手。
    - 提供 `init_chat_model()`、`LLMChain`、`ConversationChain` 等高级工具。
这样import langchian_core 是不能调用高级功能的。


## {language} 和 {text}
这里是占位符，是等着用户输入的



## ChatPromptTemplate 对象和 from_message方法
- `ChatPromptTemplate` 是 **LangChain Core** 里的一个类。
- 它的作用：帮你**定义一个“对话提示模板”**，里面可以有多个角色消息（system / user / assistant），还可以放变量占位符 `{}`。
- 最终生成的不是普通字符串，而是 **标准化的消息对象列表**（`SystemMessage`、`HumanMessage` 等），直接能喂给 ChatModel。


- `from_messages` 是 `ChatPromptTemplate` 提供的一个 **类方法 (classmethod)**。
- 它用来快速创建一个 **ChatPromptTemplate 对象**，输入是一组消息模板。
语法：
`ChatPromptTemplate.from_messages([     (角色, 模板字符串) ])`
在你代码里：
`prompt_template = ChatPromptTemplate.from_messages(     [("system", system_template), ("user", "{text}")] )`
等价于说：

- 定义一个 `system` 消息，它的内容是 `system_template`（里面有 `{language}` 占位符）。
- 定义一个 `user` 消息，它的内容是 `{text}`（用户输入的文本）。
- 把这两个合并，生成一个 **ChatPromptTemplate 对象**，赋值给 `prompt_template`。


## 用 .invoke 对 PromptTemplate 填充模板

```
prompt = prompt_template.invoke({"language": "Italian", "text": "hi!"})  
  
prompt
```