# 1. 头部依赖与初始化 (Lines 1-45)
# import ...: 引入各类工具。FastAPI 做接口，pandas/torch 做算法，langgraph 做智能体大脑。
# app = FastAPI(...): 医院开门营业，建立核心微服务。
# GLOBAL_DF = None: 【极其危险的全局变量】。目前充当大厅中央的唯一一块黑板，所有人上传的表格都往这里写。
# real_model = TF_GDC(...): 加载你用 AutoDL 训练好的 PyTorch 权重（.pth），并开启 .eval() 推理模式，挂载到 GPU/CPU 上准备干活。

import os
import io

import json     # 新增：用于将表格转为字符串
import uuid     # 新增：用于生成唯一的 session_id
import redis    # 新增：Redis 客户端
from langchain_core.runnables import RunnableConfig # 新增：LangChain 底层传参魔法

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
# GLOBAL_DF = None
workflow_app = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"正在使用 {device} 加载 TF_GDC 模型...")


# ✅ 新增：连接 Redis 储物柜
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True, protocol=2)
    redis_client.ping()
    print("✅ Redis 分布式缓存连接成功！")
except Exception as e:
    print(f"❌ Redis 连接失败，请确保 Redis 服务已启动: {e}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    real_model = TF_GDC(win_size=512, input_c=3, patch_size=16)
    real_model.load_state_dict(torch.load('best_model.pth', map_location=device))
    real_model.eval()
    real_model.to(device)
    print("✅ TF-GDC 模型引擎点火成功！")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    real_model = None



# 2. 算法工具 Tool 1: analyze_battery_data (Lines 47-83)
# 这是 Agent 的“左手”。当用户问“当前电池正常吗”，大模型就会触发这个函数。
# 它直接从大厅黑板 GLOBAL_DF 上拿数据，取最后 512 行，做标准化，转成张量，送进你的 TF-GDC 模型。
# 根据返回的 mse_loss（均方误差）判断是否大于 0.65 阈值，最后返回一段文本报告给大模型。
# ==========================================
# 工具 1：深度学习探测仪 (完全复用你的代码)
# ==========================================
@tool
def analyze_battery_data(query: str, config: RunnableConfig) -> str:
    """当用户询问实时电池运行状态、当前数据是否有异常时，必须调用此工具。"""
    global real_model, device

    # 1. 🌟 从大模型传下来的 config 里掏出专属暗号
    session_id = config.get("configurable", {}).get("session_id")
    if not session_id:
        return "系统错误：未能获取您的专属 Session ID。"

    # 2. 🌟 拿着暗号去 Redis 取数据
    cached_data = redis_client.get(session_id)
    if not cached_data:
        return "缓存中未找到您的 CSV 数据，或数据已过期（1小时），请重新上传。"

    if real_model is None: return "后端模型未加载。"

    try:
        # 3. 🌟 将 Redis 里的 JSON 字符串还原为 Pandas 表格
        df = pd.read_json(io.StringIO(cached_data))

        if len(df) < 512: return f"数据不足512行，当前为 {len(df)} 行。"

        # --- 以下逻辑不变，依然是你精妙的算法前向推理 ---
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

# 大脑组装厂 Tool 2: get_langgraph_chain (Lines 85-116)
# 这是定义 Agent 思考逻辑的地方。请了 DeepSeek 作为主脑（LLM）。
# retriever_tool: Agent 的“右手”。遇到不懂的专业词汇，用它去向量数据库里查 PDF 手册。
# StateGraph: 画状态机流转图。规定了大脑（agent 节点）和手（tools 节点）必须互相配合，直到得出最终结论。
# ==========================================
# 工具 2：LangGraph 大脑组装厂 (复用你的代码)
# ==========================================
def get_langgraph_chain(ensemble_retriever):
    llm = ChatOpenAI(
        openai_api_key=""YOUR_API_KEY_NAME"",
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

# 4. 三大 API 接口 (Lines 118-结尾)
# /upload_csv: 接收用户的表格，赋值给那个危险的 GLOBAL_DF。
# /build_agent: 接收 PDF，切片，存入 Chroma 数据库，并把组装好的大脑挂载到全局变量 workflow_app 上。
# （注：工业界中每个用户的知识库也应该隔离，但为了循序渐进，我们今天只切除最致命的 CSV 串线问题，把知识库当作所有用户的“公共参考书”。）
# /diagnose: 接收用户的文字提问，塞进大脑 workflow_app 进行图推演，返回最终答案。
# ==========================================
# 🌟 核心 API 接口定义区 🌟
# ==========================================

@app.post("/upload_csv", summary="1. 上传电池时序数据")
async def upload_csv(file: UploadFile = File(...)):
    """接收外部传来的 CSV 文件，序列化后存入 Redis 并返回唯一标识"""
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        # 1. 生成这名用户的专属暗号
        session_id = str(uuid.uuid4())

        # 2. 将表格转化为 JSON 字符串，存入 Redis，设置过期时间 3600 秒
        redis_client.set(session_id, df.to_json(), ex=3600)

        # 3. 将暗号返回给前端
        return {
            "status": "success",
            "message": f"挂载成功！共 {len(df)} 行数据",
            "filename": file.filename,
            "session_id": session_id  # 🌟 关键：前端必须存下这个字段
        }
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
    session_id: str  # 🌟 新增：强制要求前端传暗号


@app.post("/diagnose", summary="3. 发起智能诊断对话")
async def diagnose(request: QueryRequest):
    """前端带着暗号和问题来，我们触发图流转并返回报告"""
    global workflow_app
    if workflow_app is None:
        return {"status": "error", "message": "请先调用 /build_agent 组装大脑！"}

    try:
        print(f"\n" + "🔥" * 25)
        print(f"🚀 [API 触发] 用户 {request.session_id} 正在请求诊断: {request.query}")

        # 组装问题
        inputs = {"messages": [HumanMessage(content=request.query)]}

        final_state = None
        # 🌟 核心修改：将 invoke 换成 stream，实时透视 LangGraph 的思考过程
        for event in workflow_app.stream(
                inputs,
                config={"configurable": {"session_id": request.session_id}}
        ):
            for node_name, node_state in event.items():
                print(f"📍 [节点流转] ---> 成功进入 `{node_name}` 节点")

                # 可选：打印每个节点产生的中间思考内容
                if "messages" in node_state and node_state["messages"]:
                    latest_msg = node_state["messages"][-1]
                    content = str(latest_msg.content).replace('\n', ' ')
                    print(f"📝 [节点反馈]: {content[:100]}...")

            # 记录流水线上的最后一次状态
            final_state = node_state

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