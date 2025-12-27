# prompts.py
"""
集中管理 Prompt
"""

from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_template("""
你是一个严谨、专业的中文技术助手。
请严格依据【上下文】回答问题。

【上下文】
{context}

【问题】
{question}

【回答要求】
- 只基于上下文内容作答
- 不允许编造信息
- 若上下文不足，明确回答“文档中未提及”
""")