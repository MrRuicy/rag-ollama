"""
民法典RAG助手 - 简洁可扩展Web版
修复流式响应显示问题，专注于核心功能
"""
import sys
import logging
from pathlib import Path
from typing import Optional

# 添加模块路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn

# 导入专用配置
from examples.Civil_Code_RAG_Assistant.configs.civil_config import (
    CIVIL_HOST,
    CIVIL_PORT,
    print_config_summary,
    LOG_LEVEL
)

# 配置日志
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('civil_web.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("civil_web")

# 创建FastAPI应用
app = FastAPI(
    title="民法典智能助手",
    description="简洁版的民法典RAG问答系统",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# 全局RAG实例
rag_system: Optional['CivilCodeRAG'] = None

# 初始化函数
async def initialize_rag():
    """初始化RAG系统"""
    global rag_system
    
    try:
        from examples.Civil_Code_RAG_Assistant.rag.civil_rag import CivilCodeRAG
        
        logger.info("正在初始化民法典RAG系统...")
        rag_system = CivilCodeRAG(verbose=True)
        rag_system.initialize()
        logger.info("RAG系统初始化完成")
        
        # 打印系统信息
        info = rag_system.get_system_info()
        logger.info(f"系统信息: {info}")
        
    except Exception as e:
        logger.error(f"RAG系统初始化失败: {e}")
        rag_system = None
        raise

@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    print_config_summary()
    logger.info("民法典Web助手启动中...")
    
    # 初始化RAG系统
    try:
        await initialize_rag()
    except Exception as e:
        logger.error(f"启动失败: {e}")
        # 不退出，允许部分功能运行

# ========== 核心API路由 ==========

@app.get("/", response_class=HTMLResponse)
async def index():
    """首页 - 返回简洁的HTML页面"""
    html_content = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>民法典智能助手</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
                color: #333;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            .header {
                text-align: center;
                padding: 40px 20px;
                color: white;
            }
            .header h1 {
                font-size: 2.5rem;
                margin-bottom: 10px;
                text-shadow: 0 2px 4px rgba(0,0,0,0.2);
            }
            .header p {
                font-size: 1.1rem;
                opacity: 0.9;
                max-width: 600px;
                margin: 0 auto;
            }
            .main-content {
                display: grid;
                grid-template-columns: 1fr;
                gap: 30px;
                margin-top: 30px;
            }
            @media (min-width: 768px) {
                .main-content {
                    grid-template-columns: 2fr 1fr;
                }
            }
            .chat-card {
                background: white;
                border-radius: 20px;
                padding: 30px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.1);
            }
            .sidebar {
                background: rgba(255,255,255,0.95);
                border-radius: 20px;
                padding: 25px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.08);
            }
            .input-group {
                margin-bottom: 20px;
            }
            textarea {
                width: 100%;
                padding: 16px;
                border: 2px solid #e0e0e0;
                border-radius: 12px;
                font-size: 16px;
                resize: vertical;
                min-height: 100px;
                transition: border-color 0.3s;
            }
            textarea:focus {
                outline: none;
                border-color: #667eea;
            }
            .btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 14px 28px;
                border-radius: 12px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: transform 0.2s, box-shadow 0.2s;
                display: inline-flex;
                align-items: center;
                gap: 8px;
            }
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
            }
            .btn:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none !important;
            }
            .response-area {
                background: #f8f9fa;
                border-radius: 12px;
                padding: 20px;
                margin-top: 20px;
                min-height: 200px;
                max-height: 500px;
                overflow-y: auto;
                white-space: pre-wrap;
                font-family: 'Georgia', serif;
                line-height: 1.6;
            }
            .status {
                padding: 12px;
                border-radius: 8px;
                margin: 10px 0;
                font-size: 14px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .status.success {
                background: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }
            .status.error {
                background: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }
            .status.info {
                background: #d1ecf1;
                color: #0c5460;
                border: 1px solid #bee5eb;
            }
            .typing {
                display: flex;
                align-items: center;
                gap: 8px;
                color: #666;
                font-style: italic;
                margin: 10px 0;
            }
            .dot {
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: #667eea;
                animation: pulse 1.5s infinite;
            }
            .dot:nth-child(2) { animation-delay: 0.2s; }
            .dot:nth-child(3) { animation-delay: 0.4s; }
            @keyframes pulse {
                0%, 100% { opacity: 0.3; }
                50% { opacity: 1; }
            }
            .example-list {
                list-style: none;
                margin-top: 15px;
            }
            .example-item {
                padding: 12px 15px;
                margin-bottom: 8px;
                background: #f0f2f5;
                border-radius: 10px;
                cursor: pointer;
                transition: all 0.3s;
                border-left: 4px solid transparent;
            }
            .example-item:hover {
                background: #e3e6ea;
                border-left-color: #667eea;
                transform: translateX(5px);
            }
            .law-article {
                background: rgba(102, 126, 234, 0.05);
                border-left: 4px solid #667eea;
                padding: 15px;
                margin: 15px 0;
                border-radius: 0 8px 8px 0;
            }
            .law-article-title {
                font-weight: bold;
                color: #667eea;
                margin-bottom: 8px;
                display: flex;
                align-items: center;
                gap: 8px;
            }
            .system-info {
                font-size: 14px;
                color: #666;
                line-height: 1.5;
            }
            .system-info h4 {
                color: #444;
                margin: 15px 0 8px 0;
                font-size: 16px;
            }
            .footer {
                text-align: center;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid rgba(255,255,255,0.2);
                color: rgba(255,255,255,0.8);
                font-size: 14px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>⚖️ 民法典智能助手</h1>
                <p>基于RAG技术的《中华人民共和国民法典》专业问答系统</p>
            </div>
            
            <div class="main-content">
                <!-- 主聊天区域 -->
                <div class="chat-card">
                    <h2 style="margin-bottom: 20px; color: #444;">📝 法律咨询</h2>
                    
                    <div class="input-group">
                        <textarea 
                            id="questionInput" 
                            placeholder="请输入您的法律问题，例如：租房合同违约怎么办？离婚财产如何分割？..."
                            rows="3"></textarea>
                    </div>
                    
                    <div style="display: flex; gap: 10px; margin-bottom: 20px;">
                        <button class="btn" id="askButton">
                            <span>发送咨询</span>
                        </button>
                        <button class="btn" id="clearButton" style="background: #6c757d;">
                            <span>清空</span>
                        </button>
                    </div>
                    
                    <div id="statusArea"></div>
                    
                    <div class="response-area" id="responseArea">
                        <div style="text-align: center; padding: 40px 20px; color: #666;">
                            <div style="font-size: 48px; margin-bottom: 20px;">⚖️</div>
                            <h3 style="margin-bottom: 10px; color: #444;">欢迎使用民法典智能助手</h3>
                            <p>我是专业的法律助手，为您提供准确的民法典条文解释。</p>
                        </div>
                    </div>
                    
                    <div class="typing" id="typingIndicator" style="display: none;">
                        <div class="dot"></div>
                        <div class="dot"></div>
                        <div class="dot"></div>
                        <span>正在分析法律条文，请稍候...</span>
                    </div>
                </div>
                
                <!-- 侧边栏 -->
                <div class="sidebar">
                    <h3 style="margin-bottom: 20px; color: #444;">💡 使用指南</h3>
                    
                    <div class="system-info">
                        <h4>📋 示例问题</h4>
                        <ul class="example-list" id="exampleList">
                            <!-- 示例问题将由JavaScript动态加载 -->
                        </ul>
                        
                        <h4>⚙️ 系统状态</h4>
                        <div id="systemStatus">正在检测系统状态...</div>
                        
                        <h4>📊 系统信息</h4>
                        <p>• 知识库：民法典全文</p>
                        <p>• 处理方式：本地RAG</p>
                        <p>• 响应方式：流式回答</p>
                    </div>
                    
                    <div style="margin-top: 25px; padding: 15px; background: #f8f9fa; border-radius: 10px;">
                        <p style="font-size: 13px; color: #666; line-height: 1.5;">
                            <strong>⚠️ 免责声明：</strong><br>
                            本系统提供的法律信息仅供参考，不构成正式法律意见。具体案件请咨询专业律师。
                        </p>
                    </div>
                </div>
            </div>
            
            <div class="footer">
                <p>© 2024 民法典智能助手 | 基于 FastAPI + Ollama + RAG</p>
                <p style="margin-top: 5px; font-size: 13px; opacity: 0.7;">
                    <a href="/api/health" style="color: white; margin: 0 10px;">健康检查</a> |
                    <a href="/api/system/info" style="color: white; margin: 0 10px;">系统信息</a> |
                    <a href="/api/docs" style="color: white; margin: 0 10px;">API文档</a>
                </p>
            </div>
        </div>
        

        <script>
            document.addEventListener('DOMContentLoaded', function() {
                // 元素引用
                const questionInput = document.getElementById('questionInput');
                const askButton = document.getElementById('askButton');
                const clearButton = document.getElementById('clearButton');
                const responseArea = document.getElementById('responseArea');
                const typingIndicator = document.getElementById('typingIndicator');
                const statusArea = document.getElementById('statusArea');
                const exampleList = document.getElementById('exampleList');
                const systemStatus = document.getElementById('systemStatus');

                let currentEventSource = null;

                // 初始化
                async function initialize() {
                    // 加载示例问题
                    loadExamples();

                    // 检查系统状态
                    checkSystemStatus();

                    // 设置示例点击事件
                    exampleList.addEventListener('click', function(e) {
                        if (e.target.tagName === 'LI') {
                            questionInput.value = e.target.textContent;
                            questionInput.focus();
                        }
                    });

                    // 清空按钮
                    clearButton.addEventListener('click', function() {
                        questionInput.value = '';
                        responseArea.innerHTML = `
                            <div style="text-align: center; padding: 40px 20px; color: #666;">
                                <div style="font-size: 48px; margin-bottom: 20px;">⚖️</div>
                                <h3 style="margin-bottom: 10px; color: #444;">欢迎使用民法典智能助手</h3>
                                <p>我是专业的法律助手，为您提供准确的民法典条文解释。</p>
                            </div>
                        `;
                        showStatus('系统已就绪', 'success');
                    });

                    // 提问按钮
                    askButton.addEventListener('click', askQuestion);

                    // 回车键支持
                    questionInput.addEventListener('keydown', function(e) {
                        if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            askQuestion();
                        }
                    });
                }

                // 加载示例问题
                async function loadExamples() {
                    try {
                        const response = await fetch('/api/examples');
                        const data = await response.json();

                        exampleList.innerHTML = '';
                        data.examples.slice(0, 5).forEach(example => {
                            const li = document.createElement('li');
                            li.className = 'example-item';
                            li.textContent = example;
                            exampleList.appendChild(li);
                        });
                    } catch (error) {
                        console.error('加载示例失败:', error);
                    }
                }

                // 检查系统状态
                async function checkSystemStatus() {
                    try {
                        const response = await fetch('/api/health');
                        const data = await response.json();

                        if (data.status === 'healthy') {
                            systemStatus.innerHTML = `
                                <span style="color: #28a745;">✓ 系统运行正常</span>
                                <br><small>模型已加载，等待提问</small>
                            `;
                        } else {
                            systemStatus.innerHTML = `
                                <span style="color: #dc3545;">✗ 系统未就绪</span>
                                <br><small>正在初始化，请稍候...</small>
                            `;
                        }
                    } catch (error) {
                        systemStatus.innerHTML = `
                            <span style="color: #dc3545;">✗ 连接失败</span>
                            <br><small>无法连接到服务器</small>
                        `;
                    }
                }

                // 显示状态消息
                function showStatus(message, type = 'info') {
                    statusArea.innerHTML = `
                        <div class="status ${type}">
                            ${type === 'success' ? '✓' : type === 'error' ? '✗' : 'ℹ️'}
                            ${message}
                        </div>
                    `;
                }

                // 提问函数
                async function askQuestion() {
                    const question = questionInput.value.trim();

                    if (!question) {
                        showStatus('请输入问题', 'error');
                        return;
                    }

                    if (question.length < 3) {
                        showStatus('问题太短，请详细描述', 'error');
                        return;
                    }

                    // 禁用输入和按钮
                    askButton.disabled = true;
                    questionInput.disabled = true;
                    askButton.innerHTML = '<span>处理中...</span>';

                    // 显示加载状态
                    typingIndicator.style.display = 'flex';
                    showStatus('正在检索法律条文...', 'info');

                    // 清空之前的回答
                    responseArea.innerHTML = '';

                    // 如果有之前的连接，先关闭
                    if (currentEventSource) {
                        currentEventSource.close();
                    }

                    try {
                        // 创建EventSource连接
                        const encodedQuestion = encodeURIComponent(question);
                        currentEventSource = new EventSource(`/api/chat?q=${encodedQuestion}`);

                        currentEventSource.onopen = function() {
                            showStatus('已连接到服务器，正在生成回答...', 'info');
                        };

                        currentEventSource.onmessage = function(event) {
                            if (event.data === '[DONE]') {
                                // 结束处理
                                currentEventSource.close();
                                currentEventSource = null;

                                // 恢复界面
                                askButton.disabled = false;
                                questionInput.disabled = false;
                                questionInput.value = '';
                                askButton.innerHTML = '<span>发送咨询</span>';
                                typingIndicator.style.display = 'none';
                                showStatus('回答完成', 'success');

                                // 添加结束标记
                                const endMarker = document.createElement('div');
                                endMarker.style.cssText = `
                                    margin-top: 20px;
                                    padding-top: 15px;
                                    border-top: 1px solid #ddd;
                                    color: #666;
                                    font-size: 0.9em;
                                `;
                                endMarker.innerHTML = '✅ 回答结束。本回答基于《中华人民共和国民法典》相关条文，仅供参考。';
                                responseArea.appendChild(endMarker);

                                // 自动滚动
                                responseArea.scrollTop = responseArea.scrollHeight;
                            } else {
                                // 处理内容
                                let content = event.data;

                                // 检测法条引用
                                if (content.includes('《民法典》第')) {
                                    // 创建法条容器
                                    const articleDiv = document.createElement('div');
                                    articleDiv.className = 'law-article';
                                    articleDiv.innerHTML = `
                                        <div class="law-article-title">📖 法律条文</div>
                                        <div>${content}</div>
                                    `;
                                    responseArea.appendChild(articleDiv);
                                } else {
                                    // 普通内容
                                    const contentDiv = document.createElement('div');
                                    contentDiv.style.marginBottom = '10px';
                                    contentDiv.textContent = content;
                                    responseArea.appendChild(contentDiv);
                                }

                                // 自动滚动
                                responseArea.scrollTop = responseArea.scrollHeight;
                            }
                        };

                        currentEventSource.onerror = function() {
                            showStatus('连接出错，请重试', 'error');

                            // 恢复界面
                            askButton.disabled = false;
                            questionInput.disabled = false;
                            askButton.innerHTML = '<span>发送咨询</span>';
                            typingIndicator.style.display = 'none';

                            if (currentEventSource) {
                                currentEventSource.close();
                                currentEventSource = null;
                            }
                        };

                    } catch (error) {
                        showStatus('连接失败: ' + error.message, 'error');

                        // 恢复界面
                        askButton.disabled = false;
                        questionInput.disabled = false;
                        askButton.innerHTML = '<span>发送咨询</span>';
                        typingIndicator.style.display = 'none';
                    }
                }

                // 启动初始化
                initialize();
            });
        </script>
        



    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# ========== 核心流式API ==========

@app.get("/api/chat")
async def chat_stream(q: str):
    """
    流式聊天接口 - 修复版
    确保完整回答的流式传输
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="系统未初始化完成")
    
    if not q or len(q.strip()) < 2:
        raise HTTPException(status_code=400, detail="问题太短")
    
    logger.info(f"收到法律咨询: {q}")
    
    async def event_generator():
        """修复的流式生成器"""
        try:
            # 方法1：使用新增的query_simple方法（非流式但稳定）
            # response = rag_system.query_simple(q)
            # yield f"data:{response}\n\n"
            # yield "data:[DONE]\n\n"
            
            # 方法2：使用原query方法但修复流式处理
            buffer = ""
            for chunk in rag_system.query(q, stream=True):
                buffer += chunk
                
                # 按句子分割发送，避免逐字发送
                sentences = []
                current = ""
                for char in chunk:
                    current += char
                    if char in ['。', '！', '？', '；', '\n', '，'] and len(current) > 20:
                        sentences.append(current)
                        current = ""
                
                if current:
                    sentences.append(current)
                
                # 发送完整的句子
                for sentence in sentences:
                    if sentence.strip():
                        yield f"data:{sentence}\n\n"
            
            # 发送结束标记
            yield "data:[DONE]\n\n"
            
        except Exception as e:
            error_msg = f"❌ 查询失败: {str(e)}"
            logger.error(f"生成回答失败: {e}")
            yield f"data:{error_msg}\n\n"
            yield "data:[DONE]\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

# ========== 辅助API ==========

@app.get("/api/system/info")
async def system_info():
    """获取系统信息"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="系统未初始化")
    
    info = rag_system.get_system_info()
    return JSONResponse(content=info)

@app.get("/api/health")
async def health_check():
    """健康检查"""
    status = "healthy" if rag_system else "uninitialized"
    return JSONResponse(content={
        "status": status,
        "service": "civil_code_assistant",
        "rag_initialized": rag_system is not None
    })

@app.get("/api/examples")
async def get_examples():
    """获取示例问题"""
    examples = [
        "什么是违约责任？",
        "离婚需要什么条件？",
        "合同无效的情况有哪些？",
        "个人隐私权受到侵害怎么办？",
        "交通事故责任如何认定？",
        "房屋租赁合同要注意什么？",
        "遗嘱怎么写才有效？",
        "消费者权益受到侵害如何维权？",
        "产品质量问题怎么赔偿？",
        "夫妻共同债务如何认定？"
    ]
    return JSONResponse(content={"examples": examples})

@app.get("/api/debug/query")
async def debug_query(q: str, simple: bool = False):
    """调试接口：直接查询（非流式）"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="系统未初始化")
    
    try:
        if simple:
            response = rag_system.query_simple(q)
        else:
            response = next(rag_system.query(q, stream=False))
        
        return JSONResponse(content={
            "question": q,
            "answer": response,
            "mode": "simple" if simple else "streaming"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========== 主函数 ==========

def run_server(host: str = None, port: int = None):
    """运行Web服务器"""
    host = host or CIVIL_HOST
    port = port or CIVIL_PORT
    
    print("=" * 60)
    print("民法典智能助手 - Web服务启动")
    print("=" * 60)
    print(f"🌐 访问地址: http://{host}:{port}")
    print(f"📚 API文档: http://{host}:{port}/api/docs")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )

if __name__ == "__main__":
    run_server()