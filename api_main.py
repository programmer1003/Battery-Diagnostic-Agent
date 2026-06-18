# ==========================================
# 1. 头部依赖与初始化
# ==========================================
import os

# 🌟 新增：强制使用 Hugging Face 国内镜像站，解决 SSL 网络断开问题！
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import io
import json
import uuid
import redis
import asyncio
from dotenv import load_dotenv
# 🌟 新增：读取 .env 文件，瞬间激活 LangSmith 监控！
load_dotenv()
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from fastapi import Form, FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import uvicorn
import pandas as pd
import numpy as np
import torch
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage
from typing import TypedDict, Annotated, Sequence, List
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool, Tool
from sentence_transformers import CrossEncoder

from tf_gdc import TF_GDC  # 你的模型

os.environ["PYTHONIOENCODING"] = "utf-8"

app = FastAPI(title="新能源电池诊断核心微服务 API", version="v1.0")

user_workflows = {}

# ------------------------------------------
# 预热区：Redis、TF-GDC、BGE-Reranker (只执行一次)
# ------------------------------------------
try:
    redis_host = os.getenv("REDIS_HOST", "localhost")  # 动态寻址
    redis_client = redis.Redis(host=redis_host, port=6379, db=0, decode_responses=True, protocol=2)
    redis_client.ping()
    print("✅ Redis 分布式缓存连接成功！")
except Exception as e:
    print(f"❌ Redis 连接失败: {e}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"正在使用 {device} 加载核心引擎...")

try:
    real_model = TF_GDC(win_size=512, input_c=3, patch_size=16)
    real_model.load_state_dict(torch.load('best_model.pth', map_location=device))
    real_model.eval()
    real_model.to(device)
    print("✅ TF-GDC 模型引擎点火成功！")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    real_model = None

try:
    print("正在预热加载 BGE-Reranker 模型 (全局唯一)...")
    global_reranker = CrossEncoder('BAAI/bge-reranker-base')
    print("✅ BGE-Reranker 重排引擎就绪！")
except Exception as e:
    print(f"❌ BGE-Reranker 加载失败: {e}")
    global_reranker = None


# ==========================================
# 2. 算法工具与大模型编排
# ==========================================
@tool
def analyze_battery_data(query: str, config: RunnableConfig) -> str:
    """当用户询问实时电池运行状态、当前数据是否有异常时，必须调用此工具。"""
    global real_model, device

    session_id = config.get("configurable", {}).get("session_id")
    if not session_id:
        return "系统错误：未能获取您的专属 Session ID。"

    cached_data = redis_client.get(session_id)
    if not cached_data:
        return "缓存中未找到您的 CSV 数据，或数据已过期（1小时），请重新上传。"

    if real_model is None: return "后端模型未加载。"

    try:
        df = pd.read_json(io.StringIO(cached_data))
        if len(df) < 512: return f"数据不足512行，当前为 {len(df)} 行。"

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


def get_langgraph_chain(ensemble_retriever):
    llm = ChatOpenAI(
        openai_api_key=api_key,
        openai_api_base="https://api.deepseek.com",
        model_name="deepseek-chat",
        temperature=0.1
    )

    def advanced_search(original_query: str) -> str:
        rewrite_prompt = f"""你是一个专门提取电池搜索关键词的 AI。请提取核心故障特征生成一个极其精简的搜索引擎检索词。
            原问题：{original_query}
            要求：只输出检索词，绝对不要包含其他标点符号或客套话。"""

        search_query = llm.invoke(rewrite_prompt).content.strip()
        print(f"🔄 [前置优化] 原始问题: '{original_query}'  =>  引擎检索词: '{search_query}'")

        docs = ensemble_retriever.invoke(search_query)
        return "\n\n".join([doc.page_content for doc in docs])

    retriever_tool = Tool(
        name="search_battery_manual",
        description="遇到特定故障或需维修建议时，必须优先调用此工具搜索知识库。",
        func=advanced_search
    )

    tools = [analyze_battery_data, retriever_tool]

    # 🌟 绝杀修复：强行加物理锁！逼迫大模型必须调用工具！
    llm_with_tools = llm.bind_tools(tools, tool_choice="any")

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]

    def call_model(state: AgentState):
        messages = state["messages"]

        # 🌟 架构师的终极解法：动态嗅探历史记录
        # 遍历历史消息，看看有没有类型为 'tool' 的消息（判断是不是已经查过书了）
        has_tool_called = any(hasattr(msg, 'type') and msg.type == 'tool' for msg in messages)

        if not has_tool_called:
            # 第一次思考：还没查过书，强行铐上手铐，必须查！
            current_llm = llm.bind_tools(tools, tool_choice="any")
        else:
            # 第二次思考：工具已经把资料返回来了，立刻解开手铐，允许大模型正常总结输出！
            current_llm = llm.bind_tools(tools)

        system_prompt = HumanMessage(
            content="""系统人设：严谨的电池算法工程师。
            【绝对指令】：面对任何问题，你必须优先调用工具寻找证据，绝不允许凭空记忆回答！"""
        )
        response = current_llm.invoke([system_prompt] + messages)
        return {"messages": [response]}

    def reviewer_node(state: AgentState):
        messages = state["messages"]
        last_msg = messages[-1].content

        reviewer_prompt = HumanMessage(content=f"""
        审核以下 AI 回答：
        如果回答清晰专业且包含维修步骤，回复大写 'PASS'。
        如果回答敷衍或自相矛盾，回复 'REJECT:' 并跟上修改要求。
        待审核内容：{last_msg}
        """)

        evaluation = llm.invoke([reviewer_prompt]).content

        if "PASS" in evaluation.upper():
            print("\n   🛡️ [自我反思机制] 评委核验通过！")
            return {"messages": []}
        else:
            print(f"\n   ❌ [自我反思机制] 拦截重造！原因：{evaluation}")
            return {
                "messages": [HumanMessage(content=f"【自我反思指令】你的回答未通过审核，请重写。审核意见：{evaluation}")]}

    def review_condition(state: AgentState):
        messages = state["messages"]
        last_msg = messages[-1]
        if isinstance(last_msg, HumanMessage) and "【自我反思指令】" in last_msg.content:
            return "agent"
        return END

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("reviewer", reviewer_node)

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: "reviewer"})
    workflow.add_edge("tools", "agent")
    workflow.add_conditional_edges("reviewer", review_condition, {"agent": "agent", END: END})

    memory = MemorySaver()
    app_compiled = workflow.compile(checkpointer=memory)
    app_compiled.my_retriever = ensemble_retriever
    return app_compiled


# ==========================================
# 3. 核心 API 接口定义区
# ==========================================
# 🌟 新增一个纯同步的后台处理函数（扮演后厨角色）
def process_csv_in_background(content_bytes: bytes) -> tuple[str, int]:
    """这个函数在后台线程运行，怎么卡都不会影响主程序"""
    df = pd.read_csv(io.BytesIO(content_bytes))
    row_count = len(df)
    json_str = df.to_json()
    return json_str, row_count


@app.post("/upload_csv", summary="1. 上传电池时序数据")
async def upload_csv(file: UploadFile = File(...)):
    try:
        # await file.read() 是原生异步的，不会卡
        content = await file.read()
        session_id = str(uuid.uuid4())

        # 🌟 绝杀优化：把极其耗时的读取和转换，全部扔给后台线程去做！瞬间释放主线程！
        json_str, row_count = await asyncio.to_thread(process_csv_in_background, content)

        # 写入 Redis
        redis_client.set(session_id, json_str, ex=3600)

        return {
            "status": "success",
            "message": f"挂载成功！共 {row_count} 行数据",
            "filename": file.filename,
            "session_id": session_id
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/build_agent", summary="2. 上传 PDF 手册并组装大脑")
async def build_agent(file: UploadFile = File(...), session_id: str = Form(...)):
    global user_workflows, global_reranker
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.retrievers import BM25Retriever
    from langchain_community.vectorstores import Chroma

    try:
        text = ""
        content = await file.read()
        for page in PdfReader(io.BytesIO(content)).pages:
            extracted = page.extract_text()
            if extracted: text += extracted

        chunks = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_text(text)

        emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = Chroma.from_texts(chunks, emb, collection_name=f"db_{session_id}")
        chroma_retriever = db.as_retriever(search_kwargs={"k": 3})

        bm25_retriever = BM25Retriever.from_texts(chunks)
        bm25_retriever.k = 3

        class CustomEnsembleRetriever:
            def invoke(self, query):
                chroma_retriever.search_kwargs = {"k": 5}
                bm25_retriever.k = 5
                raw_docs = chroma_retriever.invoke(query) + bm25_retriever.invoke(query)

                unique_docs = []
                seen = set()
                for d in raw_docs:
                    if d.page_content not in seen:
                        seen.add(d.page_content)
                        unique_docs.append(d)

                if not unique_docs: return []

                if global_reranker is None: return unique_docs[:3]

                cross_inp = [[query, doc.page_content] for doc in unique_docs]
                scores = global_reranker.predict(cross_inp)

                doc_score_pairs = list(zip(unique_docs, scores))
                doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

                print("\n" + "=" * 40)
                print(f"🎯 [后置重排] 双路粗排共捞回 {len(unique_docs)} 块文本。")
                print("⚖️ [BGE-Reranker] 正在进行深度交叉打分...")

                top_docs = []
                for i, (doc, score) in enumerate(doc_score_pairs[:3]):
                    top_docs.append(doc)
                    preview_text = doc.page_content[:30].replace('\n', '')
                    print(f"   🏆 Top-{i + 1} | BGE 相似度打分: {score:.4f} | 预览: {preview_text}...")
                print("=" * 40 + "\n")

                return top_docs

        user_workflows[session_id] = get_langgraph_chain(CustomEnsembleRetriever())
        return {"status": "success", "message": "专属 LangGraph 与 Hybrid RAG 组装完毕！"}

    except Exception as e:
        return {"status": "error", "message": str(e)}


class QueryRequest(BaseModel):
    query: str
    session_id: str


@app.post("/diagnose", summary="3. 发起智能诊断对话")
async def diagnose(request: QueryRequest):
    global user_workflows

    my_workflow = user_workflows.get(request.session_id)
    if my_workflow is None:
        return {"status": "error", "message": "未找到您的专属系统，请先上传 PDF 构建大脑！"}

    try:
        print(f"\n" + "🔥" * 25)
        print(f"🚀 [API 触发] 用户 {request.session_id} 请求诊断: {request.query}")

        inputs = {"messages": [HumanMessage(content=request.query)]}

        config = {
            "configurable": {
                "session_id": request.session_id,
                "thread_id": str(uuid.uuid4())  # 🌟 改成动态生成！
            }
        }

        for event in my_workflow.stream(inputs, config=config):
            for node_name, node_state in event.items():
                print(f"📍 [节点流转] ---> 成功进入 `{node_name}` 节点")

                if "messages" in node_state and node_state["messages"]:
                    latest_msg = node_state["messages"][-1]
                    content = str(latest_msg.content).replace('\n', ' ')
                    print(f"📝 [节点反馈]: {content[:100]}...")

        current_state = my_workflow.get_state(config)
        answer = current_state.values["messages"][-1].content

        # 🌟 绝杀修复：把提取 Ragas 证据的代码补回来了！
        real_contexts = []
        if hasattr(my_workflow, 'my_retriever'):
            docs = my_workflow.my_retriever.invoke(request.query)
            real_contexts = [doc.page_content for doc in docs]

        return {
            "status": "success",
            "query": request.query,
            "diagnosis": answer,
            "contexts": real_contexts
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


# 🌟 【新增补丁】：进阶版流式接口 (Reviewer-Safe Streaming)
@app.post("/diagnose_stream", summary="3.1 发起智能诊断对话 (流式打字机效果)")
async def diagnose_stream(request: QueryRequest):
    global user_workflows
    my_workflow = user_workflows.get(request.session_id)
    if my_workflow is None:
        return {"status": "error", "message": "未找到您的专属系统，请先上传 PDF 构建大脑！"}

    async def event_generator():
        inputs = {"messages": [HumanMessage(content=request.query)]}
        config = {"configurable": {"session_id": request.session_id, "thread_id": str(uuid.uuid4()) }}

        # 1. 播报图谱流转的进度条（让前端知道医生在干嘛）
        async for event in my_workflow.astream(inputs, config=config):
            for node_name, node_state in event.items():
                # 🌟 补丁：把你最爱的终端 X光日志加回来！
                print(f"📍 [节点流转] ---> 成功进入 `{node_name}` 节点")
                if "messages" in node_state and node_state["messages"]:
                    latest_msg = node_state["messages"][-1]
                    content = str(latest_msg.content).replace('\n', ' ')
                    print(f"📝 [节点反馈]: {content[:100]}...")
                    
                if node_name == "agent":
                    yield f"data: {json.dumps({'type': 'status', 'content': '🧠 大脑正在思考决策...'})}\n\n"
                elif node_name == "tools":
                    yield f"data: {json.dumps({'type': 'status', 'content': '🛠️ 正在调度工具检索知识库与算法数据...'})}\n\n"
                elif node_name == "reviewer":
                    yield f"data: {json.dumps({'type': 'status', 'content': '👨‍⚖️ 质量评委正在审核回答...'})}\n\n"

        # 2. 评委审核通过后，拿出最终答案
        current_state = my_workflow.get_state(config)
        final_answer = current_state.values["messages"][-1].content

        # 3. 像打字机一样，把最终答案切成小块推给前端
        chunk_size = 3
        for i in range(0, len(final_answer), chunk_size):
            chunk = final_answer[i:i + chunk_size]
            yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
            await asyncio.sleep(0.02)  # 控制打字速度，看起来更自然

    # 返回流式响应，像水流一样推给护士
    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ... 下面保留你原来的 if __name__ == "__main__": ...

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)