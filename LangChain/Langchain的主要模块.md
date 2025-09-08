# Core
Langchian所有的基本功能都在这里。
[[Chat models and prompts]]
[[Semantic search engine]]
[[# Classify Text into Labels]]
[[# Build an Extraction Chain]]

- **Runnable**：统一接口，所有组件（ChatModel、Retriever、Tool 等）都继承它。
    - `.invoke()` → 单次调用
    - `.batch()` → 批量调用
    - `.stream()` → 流式调用
    - `.ainvoke()` → 异步调用
- **Message 对象**：对话的标准结构
    - `HumanMessage`（用户）、`AIMessage`（模型）、`SystemMessage`（系统提示）
    - 支持 OpenAI 格式（`{"role": "user", "content": "Hello"}`）和直接字符串输入。

# Components
## Chat models
- 定义：对话模型的封装，继承自 Runnable。
- 初始化：
    - 推荐用 `init_chat_model(model_name, model_provider=...)`，屏蔽不同厂商差异。
    - 示例：
        `model = init_chat_model("gpt-4o-mini", model_provider="openai")`
- 调用：
    - `model.invoke("Hello")`
    - `model.stream("Hello")` → 流式输出
- Provider：OpenAI（`langchain-openai`）、Anthropic、Google、HuggingFace 等单独包。

## Retrievers
- 定义：统一的“检索接口”，返回相关文档（Document）。
- 常见类型：
    - **向量检索**：基于 Vector Store（FAISS、Chroma、Pinecone…）。
    - **关键词检索**：BM25、TF-IDF。
    - **外部检索**：Google Search、Wikipedia、Arxiv 等。
    - **混合检索**：向量 + 关键词。
- 联系：Vector Store `.as_retriever()` → Retriever；但 Retriever 也可以独立实现。
	- 也就是说 Vector Store可以变成。

## Tools
- 定义：LLM 可以调用的外部函数/能力。
- 分类：
    - **Search 工具**：Google、Tavily、Bing、DuckDuckGo…
    - **Web Browsing 工具**：Playwright、Hyperbrowser（模拟点击、抓取网页内容）。
    - **Productivity 工具**：Notion、Slack、GitHub、Trello、Figma。
    - **Database 工具**：SQLDatabase、SparkSQL、Cassandra。
    - **Code Interpreter 工具**：Python 沙盒执行器（返回执行结果）。
- 特点：可以自己封装成 Tool，供 Agent 使用

## Document loaders
- 定义：把外部数据源加载为 LangChain 的 `Document(page_content, metadata)`。
- 数据源类型：
    - **文件**：PDF (PyPDF, PDFPlumber)、Word、TXT、CSV、Markdown。
    - **网页**：BeautifulSoup、RecursiveURL、Sitemap、AgentQL、Hyperbrowser。
    - **云存储**：AWS S3、Google Drive、Dropbox、OneDrive、腾讯 COS、华为 OBS。
    - **生产力工具**：Notion、Slack、GitHub、Trello、Figma、Roam。
    - **特殊格式**：Arxiv、PubMed、Zotero、Bilibili（标题、字幕、弹幕）。

## Vector stores
- 定义：向量数据库，用于存储 embedding 向量 + 文档
- 功能：
    - 添加文档（embedding + metadata）。
    - 相似度搜索（语义检索）。
- 常见实现：FAISS、Chroma、Pinecone、Weaviate、Milvus、Qdrant
- 联系：Vector Store 是数据层，Retriever 是接口层。

## Embedding models
- 定义：把文本（或图片、音频）转换成向量表示。
- 用途：
    - 存入 Vector Store，支持语义检索。
    - 计算相似度、聚类、分类。
- Provider：
    - OpenAI (`text-embedding-3-small/large`)
    - HuggingFace (`sentence-transformers` 系列、本地 BERT)
    - Cohere、Google、国产大模型（智谱、文心、通义）。
- 使用：
    `from langchain_openai import OpenAIEmbeddings embeddings = OpenAIEmbeddings() vector = embeddings.embed_query("hello world")`