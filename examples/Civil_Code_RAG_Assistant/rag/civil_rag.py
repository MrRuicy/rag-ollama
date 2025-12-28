"""
民法典专用RAG引擎
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# 添加模块路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from langchain_ollama import OllamaEmbeddings, OllamaLLM
    from langchain_chroma import Chroma
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.documents import Document
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请先安装依赖: pip install -r requirements_civil.txt")
    sys.exit(1)

# 导入专用配置和Prompt
from examples.Civil_Code_RAG_Assistant.configs.civil_config import (
    CIVIL_VECTOR_DB_DIR,
    EMBED_MODEL,
    LLM_MODEL,
    TOP_K,
    TEMPERATURE,
    MAX_TOKENS,
    RETRIEVAL_METHOD,
    MMR_DIVERSITY,
    SCORE_THRESHOLD,
    LEGAL_TERMS_MAPPING
)

from examples.Civil_Code_RAG_Assistant.prompts.civil_prompts import (
    get_prompt_template
)

class CivilCodeRAG:
    """民法典RAG系统 - 完全兼容版"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.rag_chain = None
        self.initialized = False
        
        if self.verbose:
            print("🔧 初始化民法典RAG系统...")
    
    def initialize(self):
        """初始化所有组件"""
        if self.initialized:
            return
        
        # 1. 初始化Embeddings
        if self.verbose:
            print(f"   🤖 加载Embedding模型: {EMBED_MODEL}")
        self.embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        
        # 2. 加载向量数据库
        if self.verbose:
            print(f"   📚 加载向量数据库: {CIVIL_VECTOR_DB_DIR}")
        try:
            self.vectorstore = Chroma(
                persist_directory=CIVIL_VECTOR_DB_DIR,
                embedding_function=self.embeddings,
                collection_name="civil_code_collection"
            )
            count = self.vectorstore._collection.count()
            if self.verbose:
                print(f"   ✅ 加载成功，包含 {count} 个法律条文")
        except Exception as e:
            print(f"❌ 加载向量数据库失败: {e}")
            print("   请先运行入库程序: python processors/civil_ingest.py")
            raise
        
        # 3. 配置检索器 - 使用 Chroma 的内置方法
        if self.verbose:
            print(f"   🔍 配置检索器")
        
        # 4. 初始化LLM
        if self.verbose:
            print(f"   ⚖️  初始化法律模型: {LLM_MODEL}")
        self.llm = OllamaLLM(
            model=LLM_MODEL,
            temperature=TEMPERATURE,
            num_predict=MAX_TOKENS,
            top_p=0.9,
            repeat_penalty=1.1,
            num_ctx=4096,
            # stop=["【重要提示】", "---"]  # 添加明确的停止词
        )
        
        # 5. 构建RAG Chain
        self.rag_chain = self._build_rag_chain()
        
        self.initialized = True
        if self.verbose:
            print("✅ 民法典RAG系统初始化完成")
    
    def _retrieve_documents(self, query: str) -> str:
        """检索文档，根据配置使用不同方法"""
        try:
            if RETRIEVAL_METHOD == "mmr":
                # MMR检索
                docs = self.vectorstore.max_marginal_relevance_search(
                    query=query,
                    k=TOP_K,
                    fetch_k=TOP_K * 3,
                    lambda_mult=MMR_DIVERSITY
                )
            elif RETRIEVAL_METHOD == "similarity_score_threshold":
                # 带相似度阈值检索
                docs_with_scores = self.vectorstore.similarity_search_with_score(
                    query=query,
                    k=TOP_K * 2
                )
                docs = []
                for doc, score in docs_with_scores:
                    if score >= SCORE_THRESHOLD:
                        docs.append(doc)
                    if len(docs) >= TOP_K:
                        break
            else:
                # 默认相似度检索
                docs = self.vectorstore.similarity_search(
                    query=query,
                    k=TOP_K
                )
            
            # 合并文档内容
            if not docs:
                return "未找到相关法律条文。"
            
            context_parts = []
            for i, doc in enumerate(docs):
                # 添加法条号和内容
                article_num = doc.metadata.get('article_number', f'第{i+1}条')
                content = doc.page_content.strip()
                context_parts.append(f"【{article_num}】{content}")
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            print(f"❌ 检索失败: {e}")
            return "检索法律条文时出现错误。"
    
    def _build_rag_chain(self, prompt_mode: str = "detailed"):
        """构建RAG Chain - 简单直接的方式"""
        prompt_template = get_prompt_template(prompt_mode)
        
        def rag_pipeline(question: str) -> str:
            """完整的RAG流水线"""
            # 1. 检索
            context = self._retrieve_documents(question)
            
            # 2. 构建提示词
            prompt = prompt_template.format(context=context, question=question)
            
            # 3. 生成回答
            return self.llm.invoke(prompt)
        
        return rag_pipeline
    
    def query(self, question: str, stream: bool = False):
        """查询民法典"""
        if not self.initialized:
            self.initialize()
        
        if self.verbose:
            print(f"❓ 问题: {question}")
            print("🔍 检索相关法条...")
        
        try:
            # 简单版本不支持流式，可以后续添加
            response = self.rag_chain(question)
            yield response
            
        except Exception as e:
            error_msg = f"❌ 查询失败: {str(e)}"
            print(error_msg)
            yield error_msg
    
    def get_retrieved_documents(self, query: str, k: Optional[int] = None) -> List[Document]:
        """获取检索到的文档（用于调试）"""
        if not self.initialized:
            self.initialize()
        
        k = k or TOP_K
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            return docs
        except Exception as e:
            print(f"❌ 获取检索文档失败: {e}")
            return []
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        if not self.initialized:
            return {"status": "未初始化"}
        
        count = self.vectorstore._collection.count() if self.vectorstore else 0
        
        return {
            "status": "运行中",
            "vector_count": count,
            "embedding_model": EMBED_MODEL,
            "llm_model": LLM_MODEL,
            "retrieval_method": RETRIEVAL_METHOD,
            "top_k": TOP_K,
            "temperature": TEMPERATURE
        }
    
    # 在 CivilCodeRAG 类中添加以下方法
    def query_simple(self, question: str) -> str:
        """非流式查询，直接返回完整答案"""
        if not self.initialized:
            self.initialize()
        
        if self.verbose:
            print(f"❓ 问题: {question}")
            print("🔍 检索相关法条...")
        
        try:
            # 构建完整提示词
            context = self._retrieve_documents(question)
            prompt = self._build_simple_prompt(context, question)
            
            # 调用LLM
            response = self.llm.invoke(prompt)
            return response
            
        except Exception as e:
            return f"❌ 查询失败: {str(e)}"
        
    def _build_simple_prompt(self, context: str, question: str) -> str:
        """构建简单提示词"""
        return f"""请基于以下民法典条文回答问题：
    
    【相关法律条文】
    {context}
    
    【用户问题】
    {question}
    
    请以专业法律顾问的身份回答，要求：
    1. 引用具体法条号
    2. 解释法律含义
    3. 给出实践建议
    4. 最后注明"注：本回答仅供参考，具体案件请咨询专业律师"
    
    【回答】：
    """
    

# 兼容原有接口
def build_civil_code_chain(streaming: bool = False):
    """创建民法典RAG链"""
    rag = CivilCodeRAG(verbose=True)
    rag.initialize()
    
    def query_function(question: str):
        return next(rag.query(question, stream=False))
    
    return query_function