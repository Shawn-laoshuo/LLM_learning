## Documents 和 Document Loaders 
langchian提供了 document 类的抽象
document 类有三个 attributes
`page_content`: 表示内容，是一个字符串
`meta_data`：随便定义的元数据
`id` （可选）document的 id

```
from langchain_core.documents import Document

documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
]
```
这个代码定义了Document对象，可以看到这个document其实是一个list 里面有不同的Document对象


加载documents
```
from langchain_community.document_loaders import PyPDFLoader

file_path = "../example_data/nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

print(len(docs))
```
`PyPDFLoader` 看起来是加载pdf用的。他的作用就是 读取 PDF 文件，把里面的每一页内容，封装成一个或多个 `Document` 对象。

`loader = PyPDFLoader(file_path)` - 创建一个 PDF 加载器对象，告诉它：要处理的文件是这个路径下的 PDF。这个时候 `loader` 还没真正把文件内容读出来。

`docs = loader.load()` 到这一步 开始执行加载操作，结果 `docs` 会是一个 **列表 (list)**，里面是多个 `Document` 对象。结果是每一夜 PDF会是一个 Document。 page_content 是这一页的文字内容。metadata 是自定义数据

```
print(f"{docs[0].page_content[:200]}\n")  
print(docs[0].metadata)
```
```
Table of Contents  
UNITED STATES  
SECURITIES AND EXCHANGE COMMISSION  
Washington, D.C. 20549  
FORM 10-K  
(Mark One)  
☑ ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(D) OF THE SECURITIES EXCHANGE ACT OF 1934  
FO  
  
{'source': '../example_data/nke-10k-2023.pdf', 'page': 0}
```
python的printf 会格式化打印字符串，把变量的值一起打印到字符串里面来。这个操作是打印第0页 .page_content 的前200个字符串。并且打印完了换行。
值得注意的是
```
.page_content
```
这个是langchain的Document的对象自带的属性。存储文档的正文文本（一个字符串）


## Splitting
加载了文档，PDF是一页一页的，这样不好用。我们的目的是通过用户query找到我们所需要的数据，因此需要把一页一页的PDF做一些切分。
langchain中就有个 text splitters 方法可以做到这些。

```
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

len(all_splits)
```
`RecursiveCharacterTextSplitter` 是一种递归文档切分方法。
`chunk_size=1000, chunk_overlap=200`  意味着每1000个字符进行一次切分，然后200个字符是重复的，这样能避免上下文丢失的情况。

## Embedding 
切分完的数据是非结构化数据。为了查询这些非结构化数据，我们向量化和计算向量相似度的方法。

```
import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
```
这段代码可以配置向量化模型，能进行embedding。代码配置的是OpenAI的embedding 模型。

这里的 `embeddings` 对象是OpenAIEmbeddings 类的实例对象。
```
vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)

assert len(vector_1) == len(vector_2)
print(f"Generated vectors of length {len(vector_1)}\n")
print(vector_1[:10])
```
然后调用 embeddings对象也就是 OpenAIEmbeddings 对象的 embed_query 方法，把输入的一段文本 `all_splits[0].page_content` 也就是存储的文档 送给OpenAI的embedding模型

## Vector stores
向量存储
在Embedding 完之后，我们下一步将向量 存储到向量数据库中去。
向量存储经常和Embedding 一起初始化，这样也更便于决定 text data是如何被送入embedding 
在 LangChain 里，`VectorStore`（向量数据库/向量存储）是一个抽象概念，用来存放 **文本对应的 embedding 向量**。

以我用过的 FAISS 为例（FAISS vector database 和FAISS不是很一样 FAISS是一种检索方法。FAISS+一个文档存储，就把这个FAISS变成一个轻量数据库了。）
```
import faiss  
from langchain_community.docstore.in_memory import InMemoryDocstore  
from langchain_community.vectorstores import FAISS  
  
embedding_dim = len(embeddings.embed_query("hello world"))  
index = faiss.IndexFlatL2(embedding_dim)  
  
vector_store = FAISS(  
embedding_function=embeddings,  
index=index,  
docstore=InMemoryDocstore(),  
index_to_docstore_id={},  
)
```

简化一下
```
from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)

```
- `InMemoryVectorStore`：LangChain 自带的一个“玩具级”向量库。  
    它不用 FAISS、Milvus、Pinecone 这些外部依赖，直接把 **embedding 向量存到 Python 内存里（list/dict）**。
- `embeddings`：传入的 embedding 模型（比如 `OpenAIEmbeddings`），用来把文本转向量。


```
ids = vector_store.add_documents(documents=all_splits)
```
**VectorStore 对象的方法**，用来往向量库里添加一批文档。
`add_documents` = **把 Document 列表批量存入向量库**。
这样你以后就能通过 **ID** 或 **相似度搜索** 找回文档。

## Usage
Embedding的结果 一般是稠密向量结果，这个结果适合用做相似度计算。
[[稠密向量和稀疏向量]]

```
results = vector_store.similarity_search(  
"How many distribution centers does Nike have in the US?"  
)  
  
print(results[0])
```
```
page_content='direct to consumer operations sell products through the following number of retail stores in the United States:
U.S. RETAIL STORES NUMBER
NIKE Brand factory stores 213 
NIKE Brand in-line stores (including employee-only stores) 74 
Converse stores (including factory stores) 82 
TOTAL 369 
In the United States, NIKE has eight significant distribution centers. Refer to Item 2. Properties for further information.
2023 FORM 10-K 2' metadata={'page': 4, 'source': '../example_data/nke-10k-2023.pdf', 'start_index': 3125}
```
`.similarity_search` 来自 vector_store 的方法 也就是 FAISS 或 InMemoryVectorStore 的方法。
```
vector_store.similarity_search(query: str, k: int = 4) -> List[Document]
```
- 输入 query 文本 → 转 embedding → 在库里检索最相似的文档 → 返回 `Document` 列表。

Async 异步版本：
```
results = await vector_store.asimilarity_search("When was Nike incorporated?")  
  
print(results[0])
```

返回分数
```
results = vector_store.similarity_search_with_score("What was Nike's revenue in 2023?")  
doc, score = results[0]  
print(f"Score: {score}\n")  
print(doc)
```
`.similarity_search_with_scor`
可以看到方法不一样了 这个方法是返回score分数的。

返回结果
```
embedding = embeddings.embed_query("How were Nike's margins impacted in 2023?")  
  
results = vector_store.similarity_search_by_vector(embedding)  
print(results[0])
```

`similarity_search_by_vector` 的输入是一个向量，返回最相似的文档 Document对象，默认不带相似度分数。

`similarity_search_with_score` 这个内部会自动调用初始化时绑定的embedding模型，把query 转换成向量，然后再在向量库里面做检索

## Retrievers
VectorStore 只是存储和进行相似度检索的一个对象，支持许多不同的存储和相似度检索。提供 .similarity_search() 这样的函数 但是不是来自 Runnable 
Retrievers 继承自 Runnable 支持 
- `.invoke()`（单次调用）
- `.batch()`（批量调用）
- `.ainvoke()`（异步调用）  
    这些让它能无缝接到 **Chain / Agent / LangGraph** 里

### 自己构造 Retrievers 用@chain包装
核心
```
from typing import List

from langchain_core.documents import Document
from langchain_core.runnables import chain


@chain
def retriever(query: str) -> List[Document]:
    return vector_store.similarity_search(query, k=1)


retriever.batch(
    [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ],
)
```
核心是调用 `vector_store.similarity_search` 来进行相似度检索
`@chain` 装饰器 把普通的Python函数 包装成Runnable对象 这样它就能和langchain 内置的 Retriever函数行为一致了
```
## LangChain 的 `@chain` 做了什么

- `@chain` 接收你定义的普通函数 `def retriever(query): ...`。
    
- 内部会把它 **包装成一个 Runnable 对象**（比如 `RunnableLambda`）。
    
- 这个对象已经实现了 `.invoke()`、`.batch()`、`.ainvoke()` 等方法。
    
- 所以，虽然你没写 `.invoke()`，但你拿到的 **已经不是原始函数了**，而是一个 Runnable 实例。
  Python 不需要你写 `.invoke()`，因为装饰器把函数替换成了一个类实例，这个类本身就带有 `.invoke()` 等方法。
  
这个时候 retriever不再是原来的Python函数，而是一个 RunnableLambda的实例。
但是实例内部保存了写的函数逻辑 作为 self.func 
同时它继承了Runnable的标准接口 .invoke() .batch() .ainvoke()
```
[[装饰器]]
想**快速自定义**（混合检索、过滤、rerank）→ 用 `@chain` 包个函数最灵活。

### 将vector_store 直接改造成Retriever
```
retriever = vector_store.as_retriever(  
search_type="similarity",  
search_kwargs={"k": 1},  
)  
  
retriever.batch(  
[  
"How many distribution centers does Nike have in the US?",  
"When was Nike incorporated?",  
],  
)
```
直接用 `as.retriever`的方法