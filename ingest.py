# ingest.py
"""
文档入库模块 - 支持多格式文档
修复版本：修复编码问题和metadata过滤
"""

import os
import re
import sys
import warnings
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# 忽略某些警告
warnings.filterwarnings("ignore")

# 添加过滤metadata的工具
from langchain_community.vectorstores.utils import filter_complex_metadata

try:
    # LangChain文档加载器
    from langchain_community.document_loaders import (
        TextLoader,
        UnstructuredMarkdownLoader,
        PythonLoader,
        CSVLoader,
        UnstructuredHTMLLoader,
        JSONLoader,
        UnstructuredWordDocumentLoader,
        UnstructuredPowerPointLoader,
        UnstructuredExcelLoader,
        UnstructuredPDFLoader,
        UnstructuredFileLoader,
        DirectoryLoader
    )
    # 文本分割器
    from langchain_text_splitters import (
        RecursiveCharacterTextSplitter,
        MarkdownHeaderTextSplitter,
        Language
    )
    # 向量化模型
    from langchain_ollama import OllamaEmbeddings
    from langchain_chroma import Chroma
    
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保已安装所有依赖：pip install langchain-community langchain-text-splitters langchain-ollama langchain-chroma")
    sys.exit(1)

# OCR依赖（可选）
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# 自定义配置文件
try:
    from config import (
        DATA_DIR,
        EMBED_MODEL,
        VECTOR_DB_DIR,
        CHUNK_SIZE,
        CHUNK_OVERLAP
    )
except ImportError:
    print("❌ 找不到 config.py，请确保 config.py 文件存在")
    sys.exit(1)

def sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    清理metadata，确保所有值都是ChromaDB支持的类型
    ChromaDB支持的类型：str, int, float, bool, None
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

class MultiFormatDocumentProcessor:
    """多格式文档处理器"""
    
    # 支持的文档格式映射到对应的加载器
    SUPPORTED_FORMATS = {
        # 文本文件 - 支持多种编码
        '.txt': lambda file_path: TextLoader(file_path, encoding='utf-8', autodetect_encoding=True),
        '.md': lambda file_path: UnstructuredMarkdownLoader(file_path, mode="elements"),
        '.markdown': lambda file_path: UnstructuredMarkdownLoader(file_path, mode="elements"),
        
        # 代码文件
        '.py': lambda file_path: PythonLoader(file_path),
        '.js': lambda file_path: TextLoader(file_path, encoding='utf-8', autodetect_encoding=True),
        '.java': lambda file_path: TextLoader(file_path, encoding='utf-8', autodetect_encoding=True),
        '.cpp': lambda file_path: TextLoader(file_path, encoding='utf-8', autodetect_encoding=True),
        '.c': lambda file_path: TextLoader(file_path, encoding='utf-8', autodetect_encoding=True),
        
        # 数据文件
        '.csv': lambda file_path: CSVLoader(file_path, encoding='utf-8'),
        '.json': lambda file_path: JSONLoader(
            file_path=file_path,
            jq_schema='.',
            text_content=False,
            json_lines=False
        ),
        
        # 标记语言
        '.html': lambda file_path: UnstructuredHTMLLoader(file_path),
        '.htm': lambda file_path: UnstructuredHTMLLoader(file_path),
        
        # 配置文件
        '.yaml': lambda file_path: TextLoader(file_path, encoding='utf-8', autodetect_encoding=True),
        '.yml': lambda file_path: TextLoader(file_path, encoding='utf-8', autodetect_encoding=True),
        
        # Office文档
        '.docx': lambda file_path: UnstructuredWordDocumentLoader(file_path, mode="elements"),
        '.doc': lambda file_path: UnstructuredWordDocumentLoader(file_path, mode="elements"),
        '.pptx': lambda file_path: UnstructuredPowerPointLoader(file_path, mode="elements"),
        '.ppt': lambda file_path: UnstructuredPowerPointLoader(file_path, mode="elements"),
        '.xlsx': lambda file_path: UnstructuredExcelLoader(file_path, mode="elements"),
        '.xls': lambda file_path: UnstructuredExcelLoader(file_path, mode="elements"),
        
        # PDF文档 - 使用fast策略，对文本型PDF效果好
        '.pdf': lambda file_path: UnstructuredPDFLoader(
            file_path, 
            mode="elements",
            strategy="fast"
        ),
    }
    
    # 图片格式（需要OCR）
    IMAGE_FORMATS = {
        '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif'
    }
    
    def __init__(self):
        """初始化文档处理器"""
        self.stats = {
            "total_files": 0,
            "processed_files": 0,
            "failed_files": 0,
            "total_chunks": 0,
            "start_time": None,
            "end_time": None
        }
        
        # 检查OCR可用性
        self.ocr_available = OCR_AVAILABLE
        
    def _get_file_extension(self, file_path: str) -> str:
        """获取文件扩展名（小写）"""
        return Path(file_path).suffix.lower()
    
    def _extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """提取文件元数据"""
        path = Path(file_path)
        try:
            stat = path.stat()
            
            # 基本元数据
            metadata = {
                "source": str(path.relative_to(DATA_DIR)) if DATA_DIR in str(path) else str(path),
                "filename": path.name,
                "extension": self._get_file_extension(file_path),
                "file_size": stat.st_size,
                "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "directory": str(path.parent.relative_to(DATA_DIR)) if DATA_DIR in str(path) else str(path.parent),
            }
            
            # 根据扩展名添加内容类型
            ext = metadata["extension"]
            if ext in ['.py', '.js', '.java', '.cpp', '.c']:
                metadata["content_type"] = "code"
            elif ext == '.md':
                metadata["content_type"] = "markdown"
            elif ext in ['.csv', '.json', '.xlsx', '.xls']:
                metadata["content_type"] = "data"
            elif ext in ['.docx', '.doc', '.pptx', '.ppt']:
                metadata["content_type"] = "office"
            elif ext == '.pdf':
                metadata["content_type"] = "pdf"
            elif ext in self.IMAGE_FORMATS:
                metadata["content_type"] = "image"
            else:
                metadata["content_type"] = "text"
                
            return sanitize_metadata(metadata)
            
        except Exception as e:
            print(f"   ⚠️  提取元数据失败 {file_path}: {e}")
            # 返回最小元数据
            return sanitize_metadata({
                "source": str(path),
                "filename": path.name,
                "extension": self._get_file_extension(file_path),
            })
    
    def _load_text_file(self, file_path: str) -> List:
        """加载文本文件"""
        try:
            extension = self._get_file_extension(file_path)
            
            # 检查是否支持该格式
            if extension not in self.SUPPORTED_FORMATS:
                print(f"   ⚠️  不支持的文件格式: {extension}")
                return []
            
            print(f"      🔧 使用加载器处理: {extension}")
            # 获取加载器
            loader_func = self.SUPPORTED_FORMATS[extension]
            loader = loader_func(file_path)
            
            # 加载文档
            documents = loader.load()
            
            # 添加元数据
            base_metadata = self._extract_metadata(file_path)
            for doc in documents:
                if hasattr(doc, 'metadata'):
                    # 清理原始metadata
                    if doc.metadata:
                        doc.metadata = sanitize_metadata(doc.metadata)
                    # 更新基础元数据
                    doc.metadata.update(base_metadata)
                else:
                    doc.metadata = base_metadata
            
            print(f"      ✅ 加载成功，得到 {len(documents)} 个文档片段")
            return documents
            
        except ImportError as e:
            print(f"   ❌ 导入错误: {e}")
            # 提示安装依赖
            if extension == '.pdf':
                print("      请安装PDF支持: pip install unstructured[pdf] pdfminer.six")
            elif extension == '.docx':
                print("      请安装Word支持: pip install unstructured[docx]")
            return []
            
        except Exception as e:
            print(f"   ❌ 加载文件失败: {e}")
            
            # 特殊处理：尝试用不同编码打开文本文件
            if extension == '.txt':
                print("      🔄 尝试其他编码...")
                try:
                    # 尝试常见编码
                    encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1', 'cp1252']
                    for encoding in encodings:
                        try:
                            with open(file_path, 'r', encoding=encoding) as f:
                                content = f.read()
                            # 使用TextLoader重新加载
                            from langchain_community.document_loaders import TextLoader
                            loader = TextLoader(file_path, encoding=encoding)
                            documents = loader.load()
                            
                            # 添加元数据
                            base_metadata = self._extract_metadata(file_path)
                            for doc in documents:
                                doc.metadata = base_metadata
                            
                            print(f"      ✅ 使用 {encoding} 编码加载成功")
                            return documents
                        except UnicodeDecodeError:
                            continue
                    print("      ❌ 尝试所有编码均失败")
                except Exception as e2:
                    print(f"      ❌ 备用加载也失败: {e2}")
            
            return []
    
    def _load_image_with_ocr(self, file_path: str) -> List:
        """使用OCR加载图片文件"""
        if not self.ocr_available:
            print(f"   ⚠️  OCR功能未启用，跳过图片文件")
            return []
        
        try:
            # 使用pytesseract直接OCR
            import pytesseract
            from PIL import Image
            
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image, lang='chi_sim+eng')
            
            if not text.strip():
                print(f"   ⚠️  图片中未识别到文字")
                return []
            
            from langchain_core.documents import Document
            documents = [Document(
                page_content=text,
                metadata=self._extract_metadata(file_path)
            )]
            
            print(f"   ✅  OCR识别成功，提取 {len(text)} 字符")
            return documents
            
        except Exception as e:
            print(f"   ❌ 图片OCR失败: {e}")
            return []
    
    def _smart_text_splitter(self, documents: List) -> List:
        """智能文本分割器"""
        all_chunks = []
        
        for doc in documents:
            content = doc.page_content
            metadata = doc.metadata.copy()  # 复制metadata
            extension = metadata.get("extension", "")
            
            try:
                if extension == '.md':
                    # Markdown文件按标题分割
                    headers_to_split_on = [
                        ("#", "Header 1"),
                        ("##", "Header 2"),
                        ("###", "Header 3"),
                    ]
                    markdown_splitter = MarkdownHeaderTextSplitter(
                        headers_to_split_on=headers_to_split_on,
                        strip_headers=False
                    )
                    chunks = markdown_splitter.split_text(content)
                elif extension == '.py':
                    # Python代码按函数/类分割
                    python_splitter = RecursiveCharacterTextSplitter.from_language(
                        language=Language.PYTHON,
                        chunk_size=CHUNK_SIZE,
                        chunk_overlap=CHUNK_OVERLAP
                    )
                    chunks = python_splitter.split_documents([doc])
                else:
                    # 通用文本分割器
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=CHUNK_SIZE,
                        chunk_overlap=CHUNK_OVERLAP,
                        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
                        length_function=len,
                    )
                    chunks = splitter.split_documents([doc])
            except Exception as e:
                print(f"      ⚠️  分割失败，使用备用分割器: {e}")
                # 备用分割器
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                    separators=["\n\n", "\n", " ", ""],
                )
                chunks = splitter.split_documents([doc])
            
            # 为每个块添加元数据
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_id"] = f"{metadata.get('filename', 'doc')}_{i}"
                chunk_metadata["chunk_index"] = i
                chunk_metadata["total_chunks"] = len(chunks)
                chunk.metadata = sanitize_metadata(chunk_metadata)
            
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def process_file(self, file_path: str) -> List:
        """处理单个文件"""
        extension = self._get_file_extension(file_path)
        filename = Path(file_path).name
        
        print(f"   📄 处理文件: {filename}")
        
        # 判断文件类型
        if extension in self.IMAGE_FORMATS:
            documents = self._load_image_with_ocr(file_path)
        else:
            documents = self._load_text_file(file_path)
        
        if not documents:
            self.stats["failed_files"] += 1
            return []
        
        # 智能分割
        chunks = self._smart_text_splitter(documents)
        
        # 更新统计
        self.stats["processed_files"] += 1
        self.stats["total_chunks"] += len(chunks)
        
        print(f"      ✅ 成功分割为 {len(chunks)} 个文本块")
        return chunks

def check_dependencies():
    """检查必要的依赖"""
    print("🔍 检查依赖...")
    
    required_packages = [
        "langchain",
        "langchain_community",
        "langchain_text_splitters",
        "chromadb",
        "unstructured",
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
    
    print("✅ 基础依赖检查通过")
    return True

def ingest():
    """主入库函数"""
    print("=" * 70)
    print("📥 多格式文档向量化入库系统")
    print("=" * 70)
    
    # 检查依赖
    if not check_dependencies():
        return
    
    # 初始化处理器
    processor = MultiFormatDocumentProcessor()
    processor.stats["start_time"] = datetime.now()
    
    # ---------- 1. 扫描文档目录 ----------
    print(f"\n📂 扫描文档目录: {DATA_DIR}")
    
    if not os.path.exists(DATA_DIR):
        print(f"❌ 目录不存在: {DATA_DIR}")
        print("   请创建data/目录并放入文档")
        return
    
    # 收集所有支持的文件
    all_files = []
    supported_extensions = set(processor.SUPPORTED_FORMATS.keys()) | processor.IMAGE_FORMATS
    
    for ext in supported_extensions:
        pattern = f"**/*{ext}"
        files = list(Path(DATA_DIR).glob(pattern))
        all_files.extend(files)
    
    # 去重
    all_files = list(set(all_files))
    
    if not all_files:
        print("❌ 未找到任何支持的文档")
        return
    
    processor.stats["total_files"] = len(all_files)
    print(f"✅ 找到 {len(all_files)} 个文档文件")
    
    # ---------- 2. 处理所有文件 ----------
    print("\n🔄 开始处理文档...")
    all_chunks = []
    
    for file_path in all_files:
        chunks = processor.process_file(str(file_path))
        if chunks:
            all_chunks.extend(chunks)
    
    # ---------- 3. 检查处理结果 ----------
    print(f"\n{'='*60}")
    print("📊 处理统计:")
    print(f"   总文件数: {processor.stats['total_files']}")
    print(f"   成功处理: {processor.stats['processed_files']}")
    print(f"   失败文件: {processor.stats['failed_files']}")
    print(f"   生成文本块: {processor.stats['total_chunks']}")
    
    if processor.stats['total_chunks'] == 0:
        print("❌ 未生成任何文本块")
        return
    
    # ---------- 4. 初始化 Embedding ----------
    print(f"\n🧠 初始化 Embedding 模型: {EMBED_MODEL}")
    try:
        embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        # 简单测试
        test_vector = embeddings.embed_query("test")
        print(f"✅ Embedding模型可用，向量维度: {len(test_vector)}")
    except Exception as e:
        print(f"❌ Embedding模型初始化失败: {e}")
        print("   请确保Ollama服务运行且模型已下载")
        return
    
    # ---------- 5. 构建向量数据库 ----------
    print(f"\n📦 构建 Chroma 向量库...")
    print(f"   存储位置: {VECTOR_DB_DIR}")
    
    try:
        # 创建存储目录
        os.makedirs(VECTOR_DB_DIR, exist_ok=True)
        
        # 过滤metadata中的复杂类型
        print(f"   🧹 过滤metadata中的复杂类型...")
        from langchain_community.vectorstores.utils import filter_complex_metadata
        filtered_chunks = filter_complex_metadata(all_chunks)
        print(f"   过滤后剩余 {len(filtered_chunks)} 个文本块")
        
        # 检查是否已有向量库
        chroma_db_path = os.path.join(VECTOR_DB_DIR, "chroma.sqlite3")
        if os.path.exists(chroma_db_path):
            print("   🔄 检测到已有向量库，进行增量更新...")
            # 加载现有向量库
            vectorstore = Chroma(
                persist_directory=VECTOR_DB_DIR,
                embedding_function=embeddings
            )
            # 添加新文档
            vectorstore.add_documents(documents=filtered_chunks)
            print(f"   ✅ 增量更新完成")
        else:
            # 创建新向量库
            vectorstore = Chroma.from_documents(
                documents=filtered_chunks,
                embedding=embeddings,
                persist_directory=VECTOR_DB_DIR
            )
            print(f"   ✅ 新建向量库完成")
        
        # 统计信息
        collection_count = vectorstore._collection.count()
        print(f"   存储向量数: {collection_count}")
        
    except Exception as e:
        print(f"❌ 向量数据库构建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ---------- 6. 完成统计 ----------
    processor.stats["end_time"] = datetime.now()
    duration = (processor.stats["end_time"] - processor.stats["start_time"]).total_seconds()
    
    print(f"\n🎉 入库完成!")
    print(f"   总耗时: {duration:.1f} 秒")
    print(f"   平均速度: {processor.stats['total_chunks']/max(duration, 0.1):.1f} 块/秒")
    print(f"   存储位置: {VECTOR_DB_DIR}")

if __name__ == "__main__":
    # 简单命令行参数
    if len(sys.argv) > 1 and sys.argv[1] in ["--help", "-h"]:
        print("用法: python ingest.py")
        print("       python ingest.py --check  # 检查依赖")
    elif len(sys.argv) > 1 and sys.argv[1] == "--check":
        check_dependencies()
    else:
        ingest()