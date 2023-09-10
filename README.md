[toc]

# Chatbot-KB项目介绍
------------------
该项目旨在构建一个与本地知识库相结合的聊天问答系统。能够根据用户提出的问题，调用本地知识库中已有的知识作为参考，生成相应带有知识背景的回答。
Chatbot-KB项目以ChatGLM2-6B作为模型基座生成问题的回答。利用langchain框架的部分模块，结合句子向量嵌入模型和FAISS向量数据库分别进行知识库文档的分割/分块,知识块的向量化，以及知识向量的存储和查询。
项目的实现思路如下：
1. 加载文件，读取文件中的知识文本
2. 对知识文本进行分块处理，包括句子层面的切分，得到一系列知识块文本。
3. 使用文本嵌入模型text2vec-large-chinese对知识块进行特征提取和建模，获得表示该知识块的嵌入向量。接着将所有嵌入向量存储到FAISS向量数据库中。
4. 当用户输入query，经过文本嵌入模型获取其embedding，接着在FAISS向量数据库中根据向量相似度进行检索，返回k个最相关的嵌入向量及其对应的知识块，作为问题回复的先验知识
5. 将先验知识和query一同输入部署在本地的ChatGLM2-6B模型基座，生成相应的回复

![实现思路](image/langchain+chatglm.png)

## 项目运行流程
-------------------
本项目运行环境： Ubuntu 22.04系统，Python 3.8.5，CUDA 12.2

- 首先准备好知识文本文件text.txt，放置在目录下，接着终端中运行vector_prompt.py，将知识文本向量化并以FAISS向量数据库的格式进行存储（vs目录下的index.faiss文件）
- 在终端中运行run.py,加载Chatbot-KB聊天系统，可在终端实现聊天问答。（如果要实现流式输出则运行run_stream.py）

## 代码及各个功能模块介绍
-------------------
### vector_prompt.py
- 定义了TextSpliter类，继承自langchain中的CharacterTextSplitter类，用于文档的切分，以及定义切分的规则。
- 定义了config_vs类，加载文本嵌入模型text2vec-large-chinese，其中store_vs()方法存储向量数据库，load_vs()方法加载已经生成的向量数据库，get_knowledge()方法输入用户query和k，返回k个最相关的文本。
### model.py
- 加载部署在本地的ChatGLM2-6B模型，并一次性返回response结果
### run_stream.py
- 运行问答系统，流式输出response结果
### run.py
- 运行问答系统，一次性输出response结果




