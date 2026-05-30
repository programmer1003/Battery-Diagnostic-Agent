import os
import io

os.environ["PYTHONIOENCODING"] = "utf-8"

# --- 1. FastAPI 核心依赖 ---
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import uvicorn

# --- 2. 你的原有依赖 ---
import pandas as pd
import numpy as np
import torch
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from typing import TypedDict, Annotated, Sequence ,List
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool, Tool

from tf_gdc import TF_GDC  # 你的模型

# ==========================================
# 初始化：医院开机与模型挂载 (只执行一次)
# ==========================================
app = FastAPI(title="新能源电池诊断核心微服务 API", version="v1.0")

# 全局变量：取代 Streamlit 的 session_state
GLOBAL_DF = None
workflow_app = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"正在使用 {device} 加载 TF_GDC 模型...")

try:
    real_model = TF_GDC(win_size=512, input_c=3, patch_size=16)
    real_model.load_state_dict(torch.load('best_model.pth', map_location=device))
    real_model.eval()
    real_model.to(device)
    print("✅ TF-GDC 模型引擎点火成功！")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    real_model = None


# ==========================================
# 工具 1：深度学习探测仪 (完全复用你的代码)
# ==========================================
@tool
def analyze_battery_data(query: str) -> str:
    """当用户询问实时电池运行状态、当前数据是否有异常时，必须调用此工具。"""
    global GLOBAL_DF, real_model, device

    if GLOBAL_DF is None: return "未检测到CSV数据。"
    if real_model is None: return "模型未加载。"
    if len(GLOBAL_DF) < 512: return "数据不足512行。"

    try:
        df = GLOBAL_DF
        v_col = next((c for c in df.columns if 'volt' in c.lower()), df.columns[0])
        i_col = next((c for c in df.columns if 'current' in c.lower()), df.columns[1])
        t_col = next((c for c in df.columns if 'temp' in c.lower()), df.columns[2])

        window_data = df[[v_col, i_col, t_col]].tail(512).values
        mean = np.mean(window_data, axis=0)
        std = np.std(window_data, axis=0) + 1e-8
        normalized_window = (window_data - mean) / std

        input_tensor = torch.tensor(normalized_window, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            reconstructed_data, _, _ = real_model(input_tensor)
            mse_loss = torch.nn.functional.mse_loss(reconstructed_data, input_tensor).item()

        threshold = 0.65
        status = "🚨 异常" if mse_loss >= threshold else "✅ 正常"

        report = f"【TF-GDC 报告】实际误差: {mse_loss:.4f}, 判定: {status}"
        if mse_loss >= threshold:
            report += "\n[专家行动指令]：检测到误差超标！请立刻去知识库检索维修建议！"
        return report
    except Exception as e:
        return f"推理失败: {str(e)}"


# ==========================================
# 工具 2：LangGraph 大脑组装厂 (复用你的代码)
# ==========================================
def get_langgraph_chain(ensemble_retriever):
    llm = ChatOpenAI(
        openai_api_key="your_api_key_here",
        openai_api_base="https://api.deepseek.com",
        model_name="deepseek-chat",
        temperature=0.1
    )

    retriever_tool = Tool(
        name="search_battery_manual",
        description="遇到特定故障或需维修建议时，调用此工具搜索知识库。",
        func=lambda query: "\n\n".join([doc.page_content for doc in ensemble_retriever.invoke(query)])
    )

    tools = [analyze_battery_data, retriever_tool]
    llm_with_tools = llm.bind_tools(tools)

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]

    def call_model(state: AgentState):
        messages = state["messages"]
        system_prompt = HumanMessage(
            content="系统人设：你是一位资深新能源电池算法工程师。如果数据提示异常，务必去知识库寻找解决方案。")
        response = llm_with_tools.invoke([system_prompt] + messages)
        return {"messages": [response]}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")

    return workflow.compile()


# ==========================================
# 🌟 核心 API 接口定义区 🌟
# ==========================================

@app.post("/upload_csv", summary="1. 上传电池时序数据")
async def upload_csv(file: UploadFile = File(...)):
    """接收外部传来的 CSV 文件，并挂载到全局黑板 GLOBAL_DF 上"""
    global GLOBAL_DF
    try:
        content = await file.read()
        # 用 io.BytesIO 把二进制数据变成 Pandas 能读的格式
        GLOBAL_DF = pd.read_csv(io.BytesIO(content))
        return {"status": "success", "message": f"挂载成功！共 {len(GLOBAL_DF)} 行数据", "filename": file.filename}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/build_agent", summary="2. 上传 PDF 手册并组装大脑")
async def build_agent(file: UploadFile = File(...)):
    """接收单份 PDF 手册，切块打入向量库，并最终编译 LangGraph 状态机"""
    global workflow_app
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.retrievers import BM25Retriever
    from langchain_community.vectorstores import Chroma

    try:
        text = ""
        # 直接读取单份文件
        content = await file.read()

        # 读取 PDF
        for page in PdfReader(io.BytesIO(content)).pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted

        chunks = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_text(text)

        # 组装双路检索
        emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = Chroma.from_texts(chunks, emb, collection_name="api_db")
        chroma_retriever = db.as_retriever(search_kwargs={"k": 3})
        bm25_retriever = BM25Retriever.from_texts(chunks)
        bm25_retriever.k = 3

        class CustomEnsembleRetriever:
            def invoke(self, query):
                docs = chroma_retriever.invoke(query) + bm25_retriever.invoke(query)
                combined = []
                seen = set()
                for d in docs:
                    if d.page_content not in seen:
                        seen.add(d.page_content)
                        combined.append(d)
                return combined

        # 编译并挂载大脑
        workflow_app = get_langgraph_chain(CustomEnsembleRetriever())
        return {"status": "success", "message": "LangGraph 与 Hybrid RAG 组装完毕！AI 准备就绪。"}

    except Exception as e:
        return {"status": "error", "message": str(e)}

# 定义用户提问的格式
class QueryRequest(BaseModel):
    query: str


@app.post("/diagnose", summary="3. 发起智能诊断对话")
async def diagnose(request: QueryRequest):
    """前端把用户的话发到这里，我们触发图流转并返回最终报告"""
    global workflow_app
    if workflow_app is None:
        return {"status": "error", "message": "请先调用 /build_agent 组装大脑！"}

    try:
        print(f"\n🚀 [API 触发] 收到诊断请求: {request.query}")

        # 组装数据并扔给 LangGraph 执行
        inputs = {"messages": [HumanMessage(content=request.query)]}
        final_state = workflow_app.invoke(inputs)
        answer = final_state["messages"][-1].content

        return {
            "status": "success",
            "query": request.query,
            "diagnosis": answer
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)