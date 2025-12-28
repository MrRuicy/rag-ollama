#!/usr/bin/env python3
"""
民法典RAG助手一键启动脚本
"""

import os
import sys
import subprocess
import webbrowser
from pathlib import Path

# 添加模块路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent.parent))

def check_dependencies():
    """检查依赖"""
    print("🔍 检查依赖...")
    
    try:
        import fastapi
        import uvicorn
        import langchain
        import chromadb
        print("✅ 基础依赖检查通过")
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请先运行: pip install -r requirements_civil.txt")
        return False
    
    # 检查Ollama服务
    try:
        import requests
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama服务运行正常")
        else:
            print("❌ Ollama服务异常")
            return False
    except:
        print("⚠️  无法连接到Ollama服务")
        print("请确保Ollama服务已启动: ollama serve")
        return False
    
    return True

def check_data():
    """检查数据文件"""
    print("📁 检查数据文件...")
    
    from examples.Civil_Code_RAG_Assistant.configs.civil_config import CIVIL_DATA_DIR
    
    data_dir = Path(CIVIL_DATA_DIR)
    if not data_dir.exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        print("请创建目录并放入民法典PDF文件")
        return False
    
    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ 未找到PDF文件，请将民法典PDF放入: {data_dir}")
        return False
    
    # 查找民法典文件
    civil_file = None
    for pdf in pdf_files:
        if "民法典" in pdf.name or "civil" in pdf.name.lower():
            civil_file = pdf
            break
    
    if civil_file:
        print(f"✅ 找到民法典文件: {civil_file.name}")
        return True
    else:
        print(f"⚠️  找到PDF文件但未明确命名为'民法典': {[p.name for p in pdf_files]}")
        print("将使用第一个PDF文件")
        return True

def check_vector_db():
    """检查向量数据库"""
    print("📦 检查向量数据库...")
    
    from examples.Civil_Code_RAG_Assistant.configs.civil_config import CIVIL_VECTOR_DB_DIR
    
    db_path = Path(CIVIL_VECTOR_DB_DIR) / "chroma.sqlite3"
    if db_path.exists():
        print(f"✅ 向量数据库已存在: {db_path}")
        
        # 检查是否为空
        try:
            import chromadb
            from examples.Civil_Code_RAG_Assistant.configs.civil_config import EMBED_MODEL
            from langchain_ollama import OllamaEmbeddings
            
            embeddings = OllamaEmbeddings(model=EMBED_MODEL)
            client = chromadb.PersistentClient(path=CIVIL_VECTOR_DB_DIR)
            collection = client.get_collection("civil_code_collection")
            count = collection.count()
            
            if count > 0:
                print(f"   包含 {count} 个法律条文向量")
                return True
            else:
                print("⚠️  向量数据库为空")
                return False
                
        except Exception as e:
            print(f"⚠️  检查向量数据库失败: {e}")
            return False
    else:
        print("❌ 向量数据库不存在")
        print("请先运行入库程序: python processors/civil_ingest.py")
        return False

def run_ingest_if_needed():
    """如果需要，运行入库程序"""
    from examples.Civil_Code_RAG_Assistant.configs.civil_config import CIVIL_VECTOR_DB_DIR
    
    db_path = Path(CIVIL_VECTOR_DB_DIR) / "chroma.sqlite3"
    
    if not db_path.exists():
        print("\n📚 检测到未构建知识库，开始入库...")
        try:
            from examples.Civil_Code_RAG_Assistant.processors.civil_ingest import main as ingest_main
            
            # 设置参数
            sys.argv = ["civil_ingest.py"]  # 模拟命令行参数
            
            # 运行入库
            ingest_main()
            
            # 检查是否成功
            if db_path.exists():
                print("✅ 知识库构建完成")
                return True
            else:
                print("❌ 知识库构建失败")
                return False
                
        except Exception as e:
            print(f"❌ 入库过程出错: {e}")
            return False
    
    return True

def main():
    """主函数"""
    print("=" * 70)
    print("民法典RAG助手 - 一键启动")
    print("=" * 70)
    
    # 步骤1: 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 步骤2: 检查数据
    if not check_data():
        sys.exit(1)
    
    # 步骤3: 检查或构建知识库
    if not run_ingest_if_needed():
        sys.exit(1)
    
    # 步骤4: 检查向量数据库
    if not check_vector_db():
        print("\n💡 提示: 如果需要重新构建知识库，请运行:")
        print("python processors/civil_ingest.py --force")
        sys.exit(1)
    
    # 步骤5: 启动Web服务
    print("\n🚀 启动民法典智能助手服务...")
    
    from examples.Civil_Code_RAG_Assistant.configs.civil_config import CIVIL_HOST, CIVIL_PORT
    
    # 导入并运行Web应用
    from examples.Civil_Code_RAG_Assistant.web.civil_app import run_server
    
    # 可选：自动打开浏览器
    import threading
    import time
    
    def open_browser():
        time.sleep(3)  # 等待服务启动
        webbrowser.open(f"http://{CIVIL_HOST}:{CIVIL_PORT}")
    
    # 在新线程中打开浏览器
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # 运行服务器
    run_server()

if __name__ == "__main__":
    main()