# app.py
"""
FastAPI Web 服务入口
"""

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import Request
from rag import build_rag_chain

app = FastAPI(title="Local RAG with Ollama")

# 初始化模板引擎
templates = Jinja2Templates(directory="templates")

# 初始化 RAG（全局只做一次）
rag_chain = build_rag_chain(streaming=True)

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    """
    渲染前端页面
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/chat")
def chat(q: str):
    """
    流式聊天接口（Server-Sent Events）
    """
    def generator():
        for token in rag_chain.stream(q):
            yield f"data:{token}\n\n"
        yield "data:[END]\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")

# 可选：添加静态文件支持（如果需要CSS/JS/图片等静态资源）
# app.mount("/static", StaticFiles(directory="static"), name="static")