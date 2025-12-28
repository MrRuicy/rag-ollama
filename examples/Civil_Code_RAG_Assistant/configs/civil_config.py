"""
民法典RAG助手专用配置
"""

import os
from pathlib import Path

# 获取当前模块所在目录
CIVIL_MODULE_DIR = Path(__file__).parent.parent

# ========== 路径配置 ==========
# 数据目录（可以指向examples内的data，也可以使用外部目录）
CIVIL_DATA_DIR = os.getenv("CIVIL_DATA_DIR", str(CIVIL_MODULE_DIR / "data"))
# 如果不存在，使用项目根目录的data
if not os.path.exists(CIVIL_DATA_DIR):
    CIVIL_DATA_DIR = "./data"

# 向量数据库目录（独立存储，避免与通用系统冲突）
CIVIL_VECTOR_DB_DIR = str(CIVIL_MODULE_DIR / "chroma_db_civil")

# ========== Ollama 服务 ==========
OLLAMA_BASE_URL = "http://localhost:11434"

# ========== 模型配置（民法典专用） ==========
# 向量化模型 - 中文法律文本优化
EMBED_MODEL = "all-minilm:latest"
# 备选: "bge-m3:latest", "bge-large-zh:latest", "moka-ai/m3e-base"

# 本地大模型 - 需要较强的法律理解和推理能力
LLM_MODEL = "qwen2.5:1.5b"  # 推荐7B版本
# 备选: "qwen2.5:1.5b", "llama3.2:3b", "deepseek-coder:6.7b"

# ========== 文档处理参数（针对民法典优化） ==========
# 民法典特点：章节结构清晰，法条独立性强
CHUNK_SIZE = 800  # 较大的chunk保持法条完整性
CHUNK_OVERLAP = 150  # 增加重叠确保法条间衔接
SPLIT_STRATEGY = "by_section"  # 按章节分割优先

# ========== 检索参数（法律查询优化） ==========
TOP_K = 5  # 法律问题可能需要引用多个法条
RETRIEVAL_METHOD = "similarity_score_threshold"  # 最大化相关性与多样性
MMR_DIVERSITY = 0.3  # 较低多样性，优先相关性
SCORE_THRESHOLD = 0.7  # 相似度阈值



# ========== 推理参数 ==========
TEMPERATURE = 0.1  # 非常低，法律回答需要确定性
MAX_TOKENS = 2000  # 增加，法律解释可能需要较长回答
TOP_P = 0.9
REPEAT_PENALTY = 1.1

# ========== Web服务配置 ==========
CIVIL_HOST = "127.0.0.1"
CIVIL_PORT = 8001  # 使用不同端口，避免与通用系统冲突

# ========== 民法典结构配置 ==========
CIVIL_CODE_SECTIONS = [
    "总则", "物权", "合同", "人格权", "婚姻家庭", 
    "继承", "侵权责任", "附则"
]

# 法律术语扩展词典
LEGAL_TERMS_MAPPING = {
    "违约": ["违约责任", "违约赔偿", "合同违约", "违反合同"],
    "侵权": ["侵权行为", "侵权责任", "损害赔偿", "侵害权益"],
    "继承": ["遗产继承", "法定继承", "遗嘱继承", "继承权"],
    "离婚": ["离婚协议", "离婚诉讼", "夫妻财产分割", "子女抚养"],
    "合同": ["合同订立", "合同履行", "合同解除", "合同无效"],
    "财产": ["财产权", "财产分割", "财产保护", "财产继承"],
}

# ========== 日志配置 ==========
LOG_LEVEL = "INFO"
LOG_FILE = str(CIVIL_MODULE_DIR / "civil_rag.log")

def get_all_configs():
    """返回所有配置的字典"""
    return {
        key: value for key, value in globals().items() 
        if not key.startswith('_') and not callable(value)
    }

def print_config_summary():
    """打印配置摘要"""
    print("=" * 60)
    print("民法典RAG助手配置摘要")
    print("=" * 60)
    print(f"📁 数据目录: {CIVIL_DATA_DIR}")
    print(f"📁 向量库目录: {CIVIL_VECTOR_DB_DIR}")
    print(f"🤖 Embedding模型: {EMBED_MODEL}")
    print(f"🤖 LLM模型: {LLM_MODEL}")
    print(f"🔧 分割策略: {SPLIT_STRATEGY}")
    print(f"🔍 检索方法: {RETRIEVAL_METHOD}")
    print(f"🌡️  温度参数: {TEMPERATURE}")
    print(f"🚀 Web服务: http://{CIVIL_HOST}:{CIVIL_PORT}")
    print("=" * 60)