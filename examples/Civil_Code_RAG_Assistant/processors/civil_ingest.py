"""
民法典专用文档处理器
"""

import os
import re
import sys
import time
import warnings
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import requests

warnings.filterwarnings("ignore")

# 添加模块路径，确保可以导入项目根目录的通用模块
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from langchain_community.document_loaders import UnstructuredPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_ollama import OllamaEmbeddings
    from langchain_chroma import Chroma
    from langchain_community.vectorstores.utils import filter_complex_metadata
    from langchain_core.documents import Document
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请先安装依赖: pip install -r requirements_civil.txt")
    sys.exit(1)

# 导入专用配置
from examples.Civil_Code_RAG_Assistant.configs.civil_config import (
    CIVIL_DATA_DIR,
    CIVIL_VECTOR_DB_DIR,
    EMBED_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SPLIT_STRATEGY,
    CIVIL_CODE_SECTIONS
)

# 参考通用文件添加的辅助函数
def sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    清理metadata，确保所有值都是ChromaDB支持的类型
    ChromaDB支持的类型：str, int, float, bool, None
    参考通用文件的实现
    """
    sanitized = {}
    for key, value in metadata.items():
        if value is None:
            sanitized[key] = None
        elif isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        elif isinstance(value, list):
            # 列表转换为逗号分隔的字符串
            if all(isinstance(item, (str, int, float, bool)) for item in value):
                sanitized[key] = ", ".join(str(item) for item in value)
            else:
                sanitized[key] = str(value)
        else:
            # 其他类型转换为字符串
            sanitized[key] = str(value)
    return sanitized

class CivilCodeIngestor:
    """民法典专用文档处理器"""
    
    def __init__(self):
        self.stats = {
            "total_pages": 0,
            "total_articles": 0,
            "total_chunks": 0,
            "sections_found": [],
            "start_time": 0,
            "end_time": 0,
            "ollama_retries": 0
        }
        
        # 民法典结构正则模式
        self.section_patterns = [
            r'^第[一二三四五六七八九十]+编\s+([^\s]+)',  # 第X编 章节名
            r'^第[一二三四五六七八九十]+章\s+([^\s]+)',  # 第X章 章节名
            r'^第[一二三四五六七八九十]+节\s+([^\s]+)',  # 第X节 章节名
        ]
        
        self.article_pattern = r'^第[零一二三四五六七八九十百千]+条\s*(.*)'
        
        # Ollama 服务配置
        self.ollama_host = "http://127.0.0.1:11434"
        self.max_retries = 3  # 最大重试次数
    
    def check_ollama_service(self) -> bool:
        """检查 Ollama 服务是否正常运行"""
        try:
            print("🔍 检查 Ollama 服务状态...")
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=10)
            if response.status_code == 200:
                print("✅ Ollama 服务运行正常")
                
                # 检查模型是否已下载
                models = response.json().get("models", [])
                model_names = [model.get("name") for model in models]
                
                # 检查完整模型名或基础名
                model_found = False
                for model_name in model_names:
                    if EMBED_MODEL == model_name or EMBED_MODEL.startswith(model_name.split(':')[0]):
                        model_found = True
                        print(f"✅ 模型 '{EMBED_MODEL}' 已找到 (实际名称: {model_name})")
                        break
                
                if not model_found and model_names:
                    print(f"❌ 模型 '{EMBED_MODEL}' 未找到")
                    print(f"   可用模型: {', '.join(model_names)}")
                    print(f"\n💡 请下载模型: ollama pull {EMBED_MODEL.split(':')[0]}")
                    return False
                elif not model_names:
                    print("⚠️  未找到任何模型")
                    print(f"💡 请下载模型: ollama pull {EMBED_MODEL.split(':')[0]}")
                    return False
                    
                return True
            else:
                print(f"❌ Ollama API 返回错误: {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            print("❌ 无法连接到 Ollama 服务")
            print("💡 请确保 Ollama 已启动:")
            print("   1. 在终端运行: ollama serve")
            print("   2. 或者 Windows: 启动 Ollama 应用")
            return False
        except Exception as e:
            print(f"❌ 检查 Ollama 服务时出错: {e}")
            return False
    
    def start_ollama_service(self) -> bool:
        """尝试启动 Ollama 服务"""
        print("🔄 尝试启动 Ollama 服务...")
        
        try:
            # 根据不同操作系统尝试启动
            if sys.platform == "win32":
                # Windows: 尝试启动 Ollama 应用
                result = subprocess.run(
                    ["ollama", "serve"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode == 0:
                    print("✅ Ollama 服务启动成功")
                    time.sleep(5)  # 等待服务完全启动
                    return True
                else:
                    print(f"❌ 启动失败: {result.stderr}")
                    return False
            else:
                # Linux/Mac: 使用 systemd 或直接启动
                result = subprocess.run(
                    ["systemctl", "--user", "start", "ollama"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    print("✅ 通过 systemctl 启动 Ollama")
                    time.sleep(5)
                    return True
                else:
                    print("⚠️  systemctl 启动失败，尝试直接启动...")
                    # 后台启动 ollama serve
                    subprocess.Popen(
                        ["ollama", "serve"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    print("✅ Ollama 服务已在后台启动")
                    time.sleep(10)  # 给更多时间启动
                    return True
                    
        except FileNotFoundError:
            print("❌ Ollama 未安装")
            print("💡 请先安装 Ollama: https://ollama.com/download")
            return False
        except Exception as e:
            print(f"❌ 启动 Ollama 服务失败: {e}")
            return False
    
    def initialize_embeddings_with_retry(self) -> Optional[OllamaEmbeddings]:
        """初始化 Embedding 模型，带重试机制"""
        max_retries = self.max_retries
        base_delay = 2  # 基础延迟2秒
        
        for attempt in range(max_retries):
            try:
                print(f"🧠 初始化 Embedding 模型 ({attempt + 1}/{max_retries}): {EMBED_MODEL}")
                
                # 根据 langchain-ollama 版本调整参数
                # 新版本可能不支持 timeout 参数，使用更简单的初始化
                try:
                    # 尝试不带 timeout 参数
                    embeddings = OllamaEmbeddings(
                        model=EMBED_MODEL,
                        base_url=self.ollama_host
                    )
                except TypeError as e:
                    if "unexpected keyword argument 'timeout'" in str(e):
                        # 如果 timeout 参数不被支持，使用更简单的初始化
                        print("   检测到不支持 timeout 参数，使用简化初始化...")
                        embeddings = OllamaEmbeddings(model=EMBED_MODEL)
                    else:
                        raise e
                
                # 简单测试连接
                print("   测试模型连接...")
                test_vector = embeddings.embed_query("测试连接")
                
                if not test_vector or len(test_vector) == 0:
                    raise ValueError("Embedding 返回空向量")
                
                print(f"✅ Embedding 模型可用，向量维度: {len(test_vector)}")
                return embeddings
                
            except requests.exceptions.ConnectionError as e:
                self.stats["ollama_retries"] += 1
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # 指数退避
                    print(f"⚠️  连接失败，{delay}秒后重试... (错误: {str(e)[:100]})")
                    time.sleep(delay)
                    
                    # 如果是第一次失败，尝试重启服务
                    if attempt == 0:
                        if not self.check_ollama_service():
                            self.start_ollama_service()
                else:
                    print(f"❌ 连接重试 {max_retries} 次均失败")
                    return None
                    
            except Exception as e:
                self.stats["ollama_retries"] += 1
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    error_msg = str(e)
                    print(f"⚠️  嵌入失败，{delay}秒后重试... (错误: {error_msg[:100]})")
                    time.sleep(delay)
                    
                    # 检查是否是特定错误
                    if "extra_forbidden" in error_msg and "timeout" in error_msg:
                        print("   检测到 timeout 参数问题，尝试重新初始化...")
                else:
                    print(f"❌ 嵌入重试 {max_retries} 次均失败: {e}")
                    return None
        
        return None
    
    def find_civil_code_pdf(self) -> Optional[Path]:
        """查找民法典PDF文件"""
        data_dir = Path(CIVIL_DATA_DIR)
        
        if not data_dir.exists():
            print(f"❌ 数据目录不存在: {data_dir}")
            return None
        
        # 搜索PDF文件
        pdf_files = list(data_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"❌ 在 {data_dir} 中未找到PDF文件")
            return None
        
        # 优先查找名称包含"民法典"的文件
        for pdf in pdf_files:
            if "民法典" in pdf.name or "civil" in pdf.name.lower():
                return pdf
        
        # 如果没有找到明确命名的，使用第一个PDF
        print(f"⚠️  未找到明确命名为'民法典'的PDF，使用第一个找到的文件")
        return pdf_files[0]
    
    def extract_legal_structure(self, content: str) -> List[Dict[str, Any]]:
        """提取民法典结构"""
        lines = content.split('\n')
        structure = []
        current_section = "总则"
        current_chapter = ""
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # 检测章节
            for pattern in self.section_patterns:
                match = re.match(pattern, line)
                if match:
                    section_name = match.group(1)
                    if "编" in line:
                        current_section = section_name
                        if section_name not in self.stats["sections_found"]:
                            self.stats["sections_found"].append(section_name)
                    elif "章" in line:
                        current_chapter = section_name
                    
                    structure.append({
                        "type": "section",
                        "name": section_name,
                        "line": i,
                        "full_path": f"{current_section} - {current_chapter} - {section_name}" 
                                      if current_chapter else f"{current_section} - {section_name}"
                    })
                    break
            
            # 检测法条
            match = re.match(self.article_pattern, line)
            if match:
                article_content = match.group(1)
                structure.append({
                    "type": "article",
                    "number": line.split('条')[0] + '条',
                    "content_preview": article_content[:50] + "..." if len(article_content) > 50 else article_content,
                    "line": i,
                    "section": current_section,
                    "chapter": current_chapter,
                    "is_article": True
                })
                self.stats["total_articles"] += 1
        
        return structure
    
    def split_by_legal_structure(self, content: str) -> List[Document]:
        """按法律结构分割文档"""
        print("   使用法律结构感知分割策略...")
        
        structure = self.extract_legal_structure(content)
        
        if not structure:
            print("   ⚠️  未能检测到法律结构，使用通用分割")
            return self.split_generic(content)
        
        lines = content.split('\n')
        chunks = []
        current_chunk_lines = []
        
        # 基础元数据（参考通用文件的实现）
        current_metadata = sanitize_metadata({
            "document_type": "民法典",
            "law_type": "civil",
            "country": "中国",
            "year": "2021",
            "content_type": "law_document"
        })
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # 检查是否为结构标记
            is_structure = False
            for struct_item in structure:
                if struct_item["line"] == i:
                    is_structure = True
                    
                    # 保存当前chunk（如果有足够内容）
                    if current_chunk_lines and len(''.join(current_chunk_lines)) > 100:
                        chunk_text = '\n'.join(current_chunk_lines)
                        doc = Document(
                            page_content=chunk_text,
                            metadata=current_metadata.copy()
                        )
                        chunks.append(doc)
                        current_chunk_lines = []
                    
                    # 更新metadata（参考通用文件的sanitize_metadata）
                    if struct_item["type"] == "article":
                        current_metadata.update(sanitize_metadata({
                            "article_number": struct_item["number"],
                            "section": struct_item["section"],
                            "chapter": struct_item.get("chapter", ""),
                            "content_type": "law_article",
                            "is_law_article": True
                        }))
                    elif struct_item["type"] == "section":
                        current_metadata.update(sanitize_metadata({
                            "section_name": struct_item["name"],
                            "full_path": struct_item.get("full_path", ""),
                            "content_type": "section_header"
                        }))
                    
                    current_chunk_lines.append(line)
                    break
            
            if not is_structure and line:
                current_chunk_lines.append(line)
        
        # 处理最后一个chunk
        if current_chunk_lines and len(''.join(current_chunk_lines)) > 100:
            chunk_text = '\n'.join(current_chunk_lines)
            doc = Document(
                page_content=chunk_text,
                metadata=current_metadata.copy()
            )
            chunks.append(doc)
        
        return chunks
    
    def split_generic(self, content: str) -> List[Document]:
        """通用分割（备用）"""
        print("   使用通用分割策略...")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n第", "\n第", "\n\n", "\n", "。", "！", "？", "；", "，", " "],
            length_function=len,
        )
        
        doc = Document(page_content=content)
        chunks = splitter.split_documents([doc])
        
        # 添加法律相关元数据（参考通用文件的sanitize_metadata）
        base_metadata = sanitize_metadata({
            "document_type": "民法典",
            "law_type": "civil",
            "content_type": "generic_split",
            "country": "中国",
            "year": "2021"
        })
        
        for chunk in chunks:
            chunk.metadata.update(base_metadata)
        
        return chunks
    
    def _extract_pdf_metadata(self, pdf_path: Path) -> Dict[str, Any]:
        """提取PDF文件元数据（参考通用文件的实现）"""
        try:
            stat = pdf_path.stat()
            metadata = sanitize_metadata({
                "source": str(pdf_path.name),
                "filename": pdf_path.name,
                "extension": ".pdf",
                "file_size": stat.st_size,
                "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "content_type": "pdf",
                "document_type": "民法典",
                "law_type": "civil"
            })
            return metadata
        except Exception as e:
            print(f"   ⚠️  提取PDF元数据失败: {e}")
            return sanitize_metadata({
                "source": str(pdf_path.name),
                "filename": pdf_path.name,
                "extension": ".pdf",
                "content_type": "pdf"
            })
    
    def _load_pdf_with_retry(self, pdf_path: Path) -> List[Document]:
        """加载PDF文件，带重试机制（参考通用文件的多格式处理）"""
        pdf_metadata = self._extract_pdf_metadata(pdf_path)
        
        # 尝试不同的加载策略
        strategies = [
            # 策略1: 使用fast策略（对文本型PDF效果好）
            ("fast", {"mode": "elements", "strategy": "fast"}),
            # 策略2: 使用hi_res策略（对扫描PDF效果好）
            ("hi_res", {"mode": "elements", "strategy": "hi_res"}),
            # 策略3: 使用paged策略（分页加载）
            ("paged", {"mode": "paged", "strategy": "auto"})
        ]
        
        for strategy_name, loader_params in strategies:
            try:
                print(f"   尝试策略: {strategy_name}")
                loader = UnstructuredPDFLoader(str(pdf_path), **loader_params)
                documents = loader.load()
                
                # 添加元数据到每个文档
                for doc in documents:
                    doc.metadata.update(pdf_metadata)
                    # 确保metadata被清理
                    doc.metadata = sanitize_metadata(doc.metadata)
                
                print(f"      ✅ {strategy_name}策略成功，得到 {len(documents)} 个元素")
                return documents
                
            except Exception as e:
                print(f"      ⚠️  {strategy_name}策略失败: {str(e)[:100]}")
                continue
        
        print("❌ 所有PDF加载策略都失败")
        return []
    
    def process_pdf(self, pdf_path: Path) -> List[Document]:
        """处理PDF文件 - 改进版"""
        print(f"📄 处理文件: {pdf_path.name}")
        print(f"   文件大小: {pdf_path.stat().st_size / 1024:.1f} KB")
        
        try:
            # 加载PDF（使用带重试的加载器）
            documents = self._load_pdf_with_retry(pdf_path)
            
            if not documents:
                print("❌ PDF加载失败")
                return []
            
            print(f"✅ PDF解析成功，得到 {len(documents)} 个元素")
            
            # 合并文本内容
            full_text = ""
            for doc in documents:
                if hasattr(doc, 'page_content'):
                    text = doc.page_content.strip()
                    if text:
                        full_text += text + "\n\n"
            
            print(f"   提取文本长度: {len(full_text)} 字符")
            
            # 根据策略分割
            if SPLIT_STRATEGY == "by_section":
                chunks = self.split_by_legal_structure(full_text)
            else:
                chunks = self.split_generic(full_text)
            
            self.stats["total_chunks"] = len(chunks)
            
            # 为每个chunk添加唯一ID（参考通用文件的实现）
            for i, chunk in enumerate(chunks):
                chunk_metadata = chunk.metadata.copy()
                chunk_metadata["chunk_id"] = f"civil_{i:04d}"
                chunk_metadata["chunk_index"] = i
                chunk_metadata["total_chunks"] = len(chunks)
                chunk.metadata = sanitize_metadata(chunk_metadata)
            
            return chunks
            
        except Exception as e:
            print(f"❌ PDF处理失败: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def check_dependencies(self) -> bool:
        """检查必要的依赖（参考通用文件的实现）"""
        print("🔍 检查依赖...")
        
        required_packages = [
            "langchain_community",
            "langchain_text_splitters",
            "langchain_ollama",
            "chromadb",
            "unstructured"
        ]
        
        missing_required = []
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                missing_required.append(package)
        
        if missing_required:
            print("❌ 缺少必需依赖:")
            for package in missing_required:
                print(f"   - {package}")
            print(f"\n请运行: pip install {' '.join(missing_required)}")
            return False
        
        # 检查PDF专用依赖
        try:
            import pdfminer
            import unstructured_pytesseract
        except ImportError:
            print("⚠️  PDF处理依赖不完整，建议安装:")
            print("    pip install unstructured[pdf] pdfminer.six unstructured_pytesseract")
        
        print("✅ 依赖检查通过")
        return True
    
    def test_embeddings_directly(self) -> bool:
        """直接测试 Embeddings 而不初始化整个流程"""
        print("\n🧪 直接测试 Ollama Embeddings...")
        
        try:
            # 最简单的初始化方式
            embeddings = OllamaEmbeddings(model=EMBED_MODEL)
            
            # 测试小文本
            test_text = "民法典第一条"
            print(f"   测试文本: '{test_text}'")
            
            vector = embeddings.embed_query(test_text)
            
            if vector and len(vector) > 0:
                print(f"✅ 嵌入成功，向量维度: {len(vector)}")
                return True
            else:
                print("❌ 嵌入返回空向量")
                return False
                
        except Exception as e:
            print(f"❌ 嵌入测试失败: {e}")
            return False
    
    def ingest(self, force_recreate: bool = False) -> bool:
        """主入库函数"""
        self.stats["start_time"] = time.time()
        
        print("=" * 70)
        print("📚 民法典知识库构建系统")
        print("=" * 70)
        
        # 检查依赖
        if not self.check_dependencies():
            return False
        
        # 检查 Ollama 服务
        print("\n🔍 检查 Ollama 服务...")
        if not self.check_ollama_service():
            print("\n🔄 尝试自动启动 Ollama 服务...")
            if not self.start_ollama_service():
                print("❌ 无法启动 Ollama 服务，请手动启动")
                print("💡 Windows: 双击 Ollama 应用图标")
                print("💡 Linux/Mac: 运行 'ollama serve'")
                return False
        
        # 直接测试 Embeddings
        if not self.test_embeddings_directly():
            print("❌ Embeddings 测试失败，无法继续")
            return False
        
        # 查找PDF文件
        pdf_path = self.find_civil_code_pdf()
        if not pdf_path:
            return False
        
        # 处理PDF
        chunks = self.process_pdf(pdf_path)
        if not chunks:
            print("❌ 未生成有效的文本块")
            return False
        
        # 显示统计信息
        print(f"\n📊 处理统计:")
        print(f"   生成文本块: {self.stats['total_chunks']}")
        print(f"   检测到法条数: {self.stats['total_articles']}")
        if self.stats["sections_found"]:
            print(f"   发现章节: {', '.join(self.stats['sections_found'][:5])}")
            if len(self.stats["sections_found"]) > 5:
                print(f"             ...等 {len(self.stats['sections_found'])} 个章节")
        
        # 初始化Embedding（带重试）
        print(f"\n🧠 初始化Embedding模型: {EMBED_MODEL}")
        embeddings = self.initialize_embeddings_with_retry()
        
        if not embeddings:
            print("❌ Embedding模型初始化失败，尝试简化初始化...")
            try:
                # 尝试最简单的初始化
                embeddings = OllamaEmbeddings(model=EMBED_MODEL)
                test_vector = embeddings.embed_query("测试")
                print(f"✅ 简化初始化成功，向量维度: {len(test_vector)}")
            except Exception as e:
                print(f"❌ 简化初始化也失败: {e}")
                print("💡 可能的解决方案:")
                print("   1. 确保 Ollama 服务正在运行")
                print("   2. 检查模型是否存在: ollama list")
                print("   3. 下载模型: ollama pull " + EMBED_MODEL.split(':')[0])
                print("   4. 尝试其他模型: nomic-embed-text, all-minilm, mxbai-embed-large")
                return False
        
        # 构建向量数据库
        print(f"\n📦 构建向量数据库...")
        print(f"   存储位置: {CIVIL_VECTOR_DB_DIR}")
        
        try:
            # 创建目录
            os.makedirs(CIVIL_VECTOR_DB_DIR, exist_ok=True)
            
            # 过滤metadata中的复杂类型（参考通用文件的重要步骤）
            print(f"   🧹 过滤metadata中的复杂类型...")
            filtered_chunks = filter_complex_metadata(chunks)
            print(f"   过滤后剩余 {len(filtered_chunks)} 个文本块")
            
            # 检查是否已有数据库（参考通用文件的实现）
            chroma_db_path = os.path.join(CIVIL_VECTOR_DB_DIR, "chroma.sqlite3")
            
            if os.path.exists(chroma_db_path) and not force_recreate:
                print("   🔄 检测到已有向量库，进行增量更新...")
                
                vectorstore = Chroma(
                    persist_directory=CIVIL_VECTOR_DB_DIR,
                    embedding_function=embeddings
                )
                
                # 分批添加文档，避免内存问题
                batch_size = 50
                total_batches = (len(filtered_chunks) + batch_size - 1) // batch_size
                
                for batch_idx in range(total_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, len(filtered_chunks))
                    batch = filtered_chunks[start_idx:end_idx]
                    
                    print(f"     添加批次 {batch_idx + 1}/{total_batches} ({len(batch)} 个文档)...")
                    vectorstore.add_documents(documents=batch)
                
                operation = "更新"
            else:
                if os.path.exists(chroma_db_path):
                    print("   强制重建向量库...")
                else:
                    print("   创建新向量库...")
                
                # 使用 from_documents 但分批处理
                batch_size = 100
                if len(filtered_chunks) > batch_size:
                    print(f"   ⚠️  文档数量较多 ({len(filtered_chunks)})，分批处理...")
                    # 分批创建
                    vectorstore = None
                    total_batches = (len(filtered_chunks) + batch_size - 1) // batch_size
                    
                    for batch_idx in range(total_batches):
                        start_idx = batch_idx * batch_size
                        end_idx = min((batch_idx + 1) * batch_size, len(filtered_chunks))
                        batch = filtered_chunks[start_idx:end_idx]
                        
                        print(f"     处理批次 {batch_idx + 1}/{total_batches} ({len(batch)} 个文档)...")
                        
                        if batch_idx == 0:
                            # 第一批创建数据库
                            vectorstore = Chroma.from_documents(
                                documents=batch,
                                embedding=embeddings,
                                persist_directory=CIVIL_VECTOR_DB_DIR,
                                collection_name="civil_code_collection"
                            )
                        else:
                            # 后续批次添加到现有数据库
                            vectorstore.add_documents(documents=batch)
                else:
                    # 文档少，直接创建
                    vectorstore = Chroma.from_documents(
                        documents=filtered_chunks,
                        embedding=embeddings,
                        persist_directory=CIVIL_VECTOR_DB_DIR,
                        collection_name="civil_code_collection"
                    )
                
                operation = "创建"
            
            # 验证（参考通用文件）
            count = vectorstore._collection.count()
            print(f"✅ 向量数据库{operation}完成")
            print(f"   存储向量数: {count}")
            
            # 显示示例
            print(f"\n📝 示例法条:")
            sample_chunks = filtered_chunks[:min(3, len(filtered_chunks))]
            for i, chunk in enumerate(sample_chunks):
                article_num = chunk.metadata.get('article_number', '未知法条')
                preview = chunk.page_content[:100].replace("\n", " ")
                print(f"   {i+1}. {article_num}: {preview}...")
            
        except Exception as e:
            print(f"❌ 向量数据库失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 检查是否是内存问题
            if "memory" in str(e).lower() or "out of memory" in str(e).lower():
                print("\n💡 内存不足建议:")
                print("   1. 增加系统内存")
                print("   2. 减少批量大小")
                print("   3. 使用更小的嵌入模型")
                print("   4. 分批处理文档")
            
            return False
        
        self.stats["end_time"] = time.time()
        duration = self.stats["end_time"] - self.stats["start_time"]
        
        print(f"\n🎉 民法典知识库构建完成!")
        print(f"   总耗时: {duration:.1f} 秒")
        print(f"   处理速度: {self.stats['total_chunks']/max(duration, 0.1):.1f} 块/秒")
        print(f"   Ollama重试次数: {self.stats['ollama_retries']}")
        
        return True

def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="民法典知识库构建工具")
    parser.add_argument("--force", "-f", action="store_true", help="强制重建向量库")
    parser.add_argument("--check", "-c", action="store_true", help="仅检查依赖和文件")
    parser.add_argument("--test", "-t", action="store_true", help="测试模式，只解析不存储")
    parser.add_argument("--model", "-m", type=str, help="指定使用的Ollama模型")
    parser.add_argument("--test-embeddings", "-e", action="store_true", help="只测试Embeddings功能")
    
    args = parser.parse_args()
    
    # 如果指定了模型，更新配置
    if args.model:
        global EMBED_MODEL
        EMBED_MODEL = args.model
        print(f"📝 使用指定模型: {EMBED_MODEL}")
    
    if args.check:
        # 检查依赖和文件
        ingestor = CivilCodeIngestor()
        ingestor.check_dependencies()
        
        # 检查 Ollama 服务
        ingestor.check_ollama_service()
        
        pdf_path = ingestor.find_civil_code_pdf()
        if pdf_path:
            print(f"\n✅ 找到民法典PDF: {pdf_path}")
            print(f"   文件大小: {pdf_path.stat().st_size / 1024:.1f} KB")
        else:
            print("❌ 未找到民法典PDF")
        
        return
    
    if args.test_embeddings:
        # 只测试 Embeddings
        ingestor = CivilCodeIngestor()
        ingestor.check_ollama_service()
        ingestor.test_embeddings_directly()
        return
    
    if args.test:
        # 测试模式：只解析不存储
        ingestor = CivilCodeIngestor()
        pdf_path = ingestor.find_civil_code_pdf()
        if pdf_path:
            chunks = ingestor.process_pdf(pdf_path)
            if chunks:
                print(f"\n🧪 测试模式结果:")
                print(f"   解析成功，生成 {len(chunks)} 个文本块")
                print(f"   检测到 {ingestor.stats['total_articles']} 个法条")
                
                # 显示前3个chunk
                for i, chunk in enumerate(chunks[:3]):
                    print(f"\n   --- Chunk {i+1} ---")
                    print(f"   Metadata: {chunk.metadata}")
                    print(f"   Content preview: {chunk.page_content[:200]}...")
        return
    
    # 执行入库
    ingestor = CivilCodeIngestor()
    success = ingestor.ingest(force_recreate=args.force)
    
    if success:
        print("\n💡 下一步:")
        print("   1. 启动服务: python run_civil.py")
        print("   2. 访问: http://127.0.0.1:8001")
    else:
        print("\n❌ 构建失败，请检查:")
        print("   1. Ollama 服务是否运行")
        print("   2. 模型是否已下载: ollama list")
        print("   3. 内存是否充足")
        print("   4. 系统资源是否足够")
        sys.exit(1)

if __name__ == "__main__":
    main()