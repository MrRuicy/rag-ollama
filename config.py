# config.py
"""
系统统一配置文件
所有可调参数只允许出现在这里
"""

# ========== Ollama 服务 ==========
OLLAMA_BASE_URL = "http://localhost:11434"

# ========== 模型配置 ==========
# 向量化模型（必须与入库阶段一致）
EMBED_MODEL = "nomic-embed-text:latest"

# 本地大模型
LLM_MODEL = "qwen2.5:1.5b"

# ========== 向量数据库 ==========
VECTOR_DB_DIR = "./chroma_db"

# ========== 文档处理 ==========
DATA_DIR = "./data"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80

# ========== 检索参数 ==========
TOP_K = 3

# ========== 推理参数 ==========
TEMPERATURE = 0.3