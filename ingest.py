# ingest.py
"""
文档入库模块

功能：
1. 加载 data/ 目录下的文档
2. 文本切分
3. 调用 Ollama Embedding
4. 构建并持久化 Chroma 向量库
"""

import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

from config import (
    DATA_DIR,
    EMBED_MODEL,
    VECTOR_DB_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

def ingest():
    print("=" * 60)
    print("📥 开始构建本地向量数据库")
    print("=" * 60)

    # ---------- 1. 加载文档 ----------
    print("\n📂 扫描文档目录:", DATA_DIR)
    documents = []

    for file in Path(DATA_DIR).glob("*.txt"):
        print(f"   - 读取文件: {file.name}")
        loader = TextLoader(str(file), encoding="utf-8")
        documents.extend(loader.load())

    if not documents:
        raise RuntimeError("❌ 未找到任何文档，请检查 data/ 目录")

    print(f"✅ 共加载 {len(documents)} 个原始文档")

    # ---------- 2. 文本切分 ----------
    print("\n✂️  文档切分中...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )

    chunks = splitter.split_documents(documents)
    print(f"✅ 切分得到 {len(chunks)} 个文本块")

    # ---------- 3. 初始化 Embedding ----------
    print("\n🧠 初始化 Embedding 模型:", EMBED_MODEL)
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    # ---------- 4. 构建向量数据库 ----------
    print("\n📦 构建 Chroma 向量库...")
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR
    )

    print("🎉 向量数据库构建完成")
    print("📁 存储位置:", VECTOR_DB_DIR)

if __name__ == "__main__":
    ingest()