# rag.py
"""
RAG 系统核心逻辑

职责：
1. 加载向量数据库
2. 构建 Retriever
3. 组装 Prompt + LLM
4. 提供可流式输出的 RAG Chain
"""

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from config import (
    VECTOR_DB_DIR,
    EMBED_MODEL,
    LLM_MODEL,
    TOP_K,
    TEMPERATURE
)
from prompts import RAG_PROMPT

def build_rag_chain(streaming: bool = False):
    print("🔧 初始化 RAG 系统...")

    # ---------- 1. Embedding ----------
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    # ---------- 2. 向量数据库 ----------
    print("📚 加载向量数据库:", VECTOR_DB_DIR)
    vectorstore = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings
    )

    # ---------- 3. Retriever ----------
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": TOP_K}
    )

    # ---------- 4. LLM ----------
    llm = OllamaLLM(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        streaming=streaming
    )

    # ---------- 5. RAG Chain ----------
    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    print("✅ RAG 系统初始化完成")
    return rag_chain