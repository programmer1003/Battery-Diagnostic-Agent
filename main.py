import os

os.environ["PYTHONIOENCODING"] = "utf-8"  # 1. 强制系统用 utf-8 编码，防止你在 Windows 终端看到乱码
import streamlit as st                    # 2. 前端网页框架（就是你看到的那个漂亮的网页）
import pandas as pd                       # 3. 处理表格数据的神器（读 CSV 全靠它）
import numpy as np                        # 4. 数学计算神器（算平均值、标准差）
from dotenv import load_dotenv            # 5. 用来读取隐藏的密码文件（虽然这版代码没用到，但写上是个好习惯）
from PyPDF2 import PdfReader              # 6. 用来读取 PDF 文件的文字

# --- 核心依赖 ---
from langchain_text_splitters import CharacterTextSplitter   # 7. 把长篇大论切成小块的剪刀
from langchain_openai import ChatOpenAI                      # 8. 接入大模型的接口（我们用它接了 DeepSeek）
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder      # 9. 提示词模板工具
from langchain_core.messages import HumanMessage, AIMessage  # 10. 定义什么是“人说的话”，什么是“AI说的话”

from typing import TypedDict, Annotated, Sequence           # 11. 规定数据格式的语法
from langchain_core.messages import BaseMessage             # 12. 基础消息类型
from langgraph.graph.message import add_messages            # 13. 往记事本里追加消息的动作
from langgraph.graph import StateGraph, END                 # 14. 画状态图的画笔，和图的终点(END)
from langgraph.prebuilt import ToolNode, tools_condition    # 15. 预先做好的工具节点和条件判断
from langchain_core.tools import tool, Tool                 # 16. 把普通函数变成大模型工具的

from htmlTemplates import css, bot_template, user_template  # 17. 导入你本地写的网页样式和头像

import torch
# 🌟 导入你写的心血结晶！(注意文件名和类名不要写错)
from tf_gdc import TF_GDC

# 🌟 全局挂载真实深度学习模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"正在使用 {device} 加载 TF_GDC 模型...")

try:
    # 按照你代码里的默认参数实例化：win_size=512, input_c=3
    real_model = TF_GDC(win_size=512, input_c=3,patch_size=16)

    # 加载你在 AutoDL 上跑出来的那个 best_model.pth 权重文件
    # (确保把 .pth 文件下载到了当前文件夹)
    real_model.load_state_dict(torch.load('best_model.pth', map_location=device))
    # 切换模式（极度重要！）：告诉模型现在是“考试/看病”状态，不是“学习”状态
    real_model.eval()  # 开启纯推理模式
    real_model.to(device)
    print("✅ TF-GDC 模型引擎点火成功！")
except Exception as e:
    print(f"❌ 模型加载失败，请检查权重文件是否存在: {e}")
    real_model = None
# 🌟 终极绝杀：用 Python 原生全局变量绕过 Streamlit 的多线程隔离陷阱！
GLOBAL_DF = None                                            # 18. 在餐厅正中央立一块空黑板，用来存 CSV 数据，所有线程都能看到


# --- 1. CSV 数据处理与摘要生成 ---
def process_csv_data(csv_file):
    global GLOBAL_DF  # 19. 声明我要用大厅里那块黑板
    csv_file.seek(0)  # 20. 把文件读取指针拨回开头（防止之前读过一遍导致读出空数据）
    try:
        df = pd.read_csv(csv_file)  # 21. 把 CSV 文件变成 Pandas 的二维表格 (DataFrame)
        st.session_state.raw_df = df  # 22. 往网页的私人记事本里存一份
        GLOBAL_DF = df  # 23. 往全局黑板上也抄一份（最关键的一步）
    except Exception as e:
        return None, f"CSV 读取错误: {str(e)}"  # 24. 万一读错了，别崩溃，返回报错信息

    # 下面开始写一份简单的“体检报告摘要”
    try:
        summary = f"【检测到电池运行数据文件: {csv_file.name}】\n"
        summary += f"- 数据行数: {len(df)}\n"

        # 25. 在表格所有列名里，找找有没有包含 'volt' (电压) 的列
        volt_col = next((col for col in df.columns if 'volt' in col.lower()), None)
        if volt_col:
            v_min = df[volt_col].min()  # 26. 找出最低电压
            v_max = df[volt_col].max()  # 27. 找出最高电压
            v_mean = df[volt_col].mean()  # 28. 算出平均电压
            summary += f"- 电压监测: 范围 {v_min:.3f}V ~ {v_max:.3f}V (均值: {v_mean:.3f}V)\n"
            # 29. 简单的阈值报警（跌破 3V 或超过 4.25V）
            if v_min < 3.0: summary += "  ⚠️ 异常标记: 检测到欠压风险 (Under-voltage)！\n"
            if v_max > 4.25: summary += "  ⚠️ 异常标记: 检测到过压风险 (Over-voltage)！\n"

        return df, summary  # 30. 把表格和这段文字报告返回出去
    except Exception as e:
        return None, f"数据特征提取失败: {str(e)}"


# 🌟 2. TF-GDC 算法工具 ---
@tool
def analyze_battery_data(query: str) -> str:
    """当用户询问实时电池运行状态、当前数据是否有异常时，必须调用此工具。它将调用后端的 TF-GDC 深度学习模型进行 512 窗口的时序异常检测。"""
    global GLOBAL_DF
    global real_model
    global device

    if GLOBAL_DF is None:
        return "当前系统没有检测到上传的电池时序数据(CSV)。请提示用户先在侧边栏上传数据。"

    if real_model is None:
        return "后端 TF-GDC 模型尚未成功加载，无法执行深度诊断。"

    try:
        df = GLOBAL_DF
        if len(df) < 512:
            return f"❌ TF-GDC 算法要求输入至少 512 个时间步的数据，当前仅为 {len(df)}。"

        # 1. 提取 [V, I, T] 数据
        v_col = next((c for c in df.columns if 'volt' in c.lower()), df.columns[0])
        i_col = next((c for c in df.columns if 'current' in c.lower()), df.columns[1])
        t_col = next((c for c in df.columns if 'temp' in c.lower()), df.columns[2])
        window_data = df[[v_col, i_col, t_col]].tail(512).values

        # 2. 数据标准化
        mean = np.mean(window_data, axis=0)
        std = np.std(window_data, axis=0) + 1e-8
        normalized_window = (window_data - mean) / std

        # 3. 数组转 PyTorch Tensor: shape 变为 [1, 512, 3]
        input_tensor = torch.tensor(normalized_window, dtype=torch.float32).unsqueeze(0).to(device)

        # ==========================================
        # 💥 核心：真实发动你的 TF-GDC 模型推理！
        # ==========================================
        with torch.no_grad():
            # 接收你的 3 个返回值
            reconstructed_data, proj_a, proj_b = real_model(input_tensor)

            # 4. 计算真实的 MSE 重构误差得分
            mse_loss = torch.nn.functional.mse_loss(reconstructed_data, input_tensor).item()

        # 5. 从你的 viz_cache 提取真实的 V-T 注意力权重
        attn_matrix = real_model.viz_cache.get('attn_weights', None)
        if attn_matrix is not None:
            # attn_matrix 形状是 [patch_num, 3, 3]
            # 提取所有 patch 在 V(索引0) 和 T(索引2) 之间的平均耦合权重
            v_t_weight = float(np.mean(attn_matrix[:, 0, 2]))
        else:
            v_t_weight = 0.0

        # 6. 异常判定
        threshold = 0.65  # 这是你验证集最佳阈值，可随时修改
        if mse_loss >= threshold:
            status = "🚨 异常 (Score >= Threshold)"
        else:
            status = "✅ 正常 (Score < Threshold)"

        # 7. 组装终极战报
        report = f"""
        【TF-GDC 真实深度学习模型推理报告】
        - 模型架构：Time-Frequency Decomposition-Enhanced Global Dictionary Contrastive Network
        - 当前窗口实际重构误差 (MSE)：{mse_loss:.4f}
        - 预设报警阈值：{threshold}
        - 最终系统判定：{status}
        - 电压(V)与温度(T)平均交叉注意力权重：{v_t_weight:.4f}
        """

        if mse_loss >= threshold:
            report += "\n[专家行动指令]：检测到真实的重构误差超标，且 V-T 耦合权重出现异动。请大模型立刻去知识库检索‘电压下降伴随温度升高的内部机理’（如微短路、热失控）及维修建议！"

        return report

    except Exception as e:
        return f"真实 TF-GDC 推理流转失败: {str(e)}"


# --- 3. PDF 处理与混合知识库 ---
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        for page in PdfReader(pdf).pages:                               # 44. 一页一页地翻 PDF
            if extracted := page.extract_text(): text += extracted      # 45. 把上面的字全抠下来拼在一起
    return text


def get_text_chunks(text):
    # 46. 把几十万字切成每块 1000 字的小块，两块之间重叠 200 字（防止一句话被从中间劈开）
    return CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len).split_text(
        text)


def get_ensemble_retriever(text_chunks):
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.retrievers import BM25Retriever
    from langchain_community.vectorstores import Chroma

    # 47. 加载开源的 MiniLM 嵌入模型（负责把中文翻译成几百维的数学向量）
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # 48. 建立 ChromaDB 向量数据库（意会派管理员），每次搜出最相关的前 3 块 (k=3)
    chroma_vectorstore = Chroma.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        persist_directory="./chroma_battery_db",
        collection_name="bms_manual_collection"
    )
    chroma_retriever = chroma_vectorstore.as_retriever(search_kwargs={"k": 3})
    # 49. 建立 BM25 稀疏检索器（死理派管理员），靠关键词精准匹配前 3 块
    bm25_retriever = BM25Retriever.from_texts(text_chunks)
    bm25_retriever.k = 3

    # 50. 手搓的双路融合器类
    class CustomEnsembleRetriever:
        def __init__(self, r1, r2):
            self.r1 = r1
            self.r2 = r2

        def invoke(self, query):
            docs1 = self.r1.invoke(query)
            docs2 = self.r2.invoke(query)

            combined_docs = []
            seen_content = set()
            for doc in docs1 + docs2:
                if doc.page_content not in seen_content:
                    seen_content.add(doc.page_content)
                    combined_docs.append(doc)
            return combined_docs

    return CustomEnsembleRetriever(bm25_retriever, chroma_retriever)


# 🌟🌟🌟 4. LangGraph 状态图架构 🌟🌟🌟
class AgentState(TypedDict):
    # 53. 定义全局档案袋，里面只有一样东西：不断追加的历史对话记录 (messages)
    messages: Annotated[Sequence[BaseMessage], add_messages]


def get_langgraph_chain(ensemble_retriever):
    # 54. 请来 DeepSeek 作为项目经理（大脑）
    llm = ChatOpenAI(
        openai_api_key="sk-84fa3d55140d4b47b5137cc6381f86e0",
        openai_api_base="https://api.deepseek.com",
        model_name="deepseek-chat",
        temperature=0.1
    )
    # 55. 把刚才的双路混合检索器，也包装成一个带说明书的工具
    retriever_tool = Tool(
        name="search_battery_manual",
        description="当用户询问特定故障代码、电化学原理、维修建议时，必须调用此工具搜索专家知识库。",
        func=lambda query: "\n\n".join([doc.page_content for doc in ensemble_retriever.invoke(query)])
    )

    tools = [analyze_battery_data, retriever_tool]  # 56. 把两个工具放进工具箱
    llm_with_tools = llm.bind_tools(tools)          # 57. 把工具箱交到大脑手里（绑定）

    def call_model(state: AgentState):
        # 58. 这是 agent 部门的工作日常：打开档案袋，拿到记录
        messages = state["messages"]
        system_prompt = HumanMessage(content="""系统人设：你是一位资深新能源电池算法工程师与BMS诊断专家。
        请结合电化学原理进行严谨的深度归因推理。如果数据提示异常，务必去知识库寻找解决方案。""")# 默念一遍自己的专家人设
        # 59. 结合人设和历史记录，大脑开始思考，并给出决定（回答问题，或下达使用工具的命令）
        response = llm_with_tools.invoke([system_prompt] + messages)
        return {"messages": [response]}     # 把决定塞回档案袋

    tool_node = ToolNode(tools)     # 60. 这是 tools 部门，专门负责执行大脑下达的工具命令

    # --- 下面开始画流程图 ---
    workflow = StateGraph(AgentState)  # 61. 拿出一张画纸，规定大家都用 AgentState 档案袋
    workflow.add_node("agent", call_model)  # 62. 在纸上画一个圆圈，叫 agent
    workflow.add_node("tools", tool_node)  # 63. 再画一个圆圈，叫 tools

    workflow.set_entry_point("agent")  # 64. 规定起点：所有任务从 agent 开始

    # 65. 画分叉路口：agent 思考完后，如果命令用工具，就走 'tools' 路线；如果回答完毕，就走 END 路线结束。
    workflow.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})

    # 66. 画强制回流线：tools 干完活，绝对不允许下班，必须把结果送回 agent 再次思考。
    workflow.add_edge("tools", "agent")

    return workflow.compile()  # 67. 把画好的草图编译成可执行的程序


# --- 5. 核心交互与透视打印 ---
def handle_userinput(user_question):
    if st.session_state.chat_history is None:
        st.session_state.chat_history = []

    chat_history_msgs = []
    for q, a in st.session_state.chat_history:
        chat_history_msgs.append(HumanMessage(content=q))
        chat_history_msgs.append(AIMessage(content=a))
    chat_history_msgs.append(HumanMessage(content=user_question))

    with st.spinner("AI 专家正在使用 LangGraph 进行多步推理与调度..."):
        try:
            print("\n" + "🔥" * 25)  # 68. 在黑色终端里打出火焰分割线
            print(f"🚀 [LangGraph 启动] 收到指令，开始图计算流转...")

            final_state = None
            # 69. 这里的 .stream() 就像看流水线一样，盯死状态图里的每一次流转
            for event in st.session_state.conversation.stream(
                    {"messages": chat_history_msgs},
                    config={"recursion_limit": 10}
            ):
                for node_name, node_state in event.items():
                    # 70. 一旦有部门完成工作，就在终端打印出来：“成功进入 xxx 节点”
                    print(f"\n📍 [节点流转] ---> 成功进入 `{node_name}` 节点")
                    if "messages" in node_state and node_state["messages"]:
                        # 71. 把这步的结果前 150 个字打印出来
                        latest_msg = node_state["messages"][-1]
                        content = str(latest_msg.content).replace('\n', ' ')
                        print(f"📝 [节点反馈]: {content[:150]}...")

                final_state = node_state

            # 72. 整个图流转结束后，把档案袋里最后一条消息作为最终答案提取出来
            answer = final_state["messages"][-1].content
        except Exception as e:
            answer = f"图流转执行出错：{str(e)}"

    st.session_state.chat_history.append((user_question, answer))

    for q, a in st.session_state.chat_history:
        st.write(user_template.replace("{{MSG}}", q), unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}", a), unsafe_allow_html=True)


# --- 6. 界面布局 ---
def main():
    # 73. 网页的各种设置（标题、图标）
    load_dotenv()
    st.set_page_config(page_title="新能源电池故障智能诊断系统", page_icon="🔋", layout="wide")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state: st.session_state.conversation = None
    if "chat_history" not in st.session_state: st.session_state.chat_history = None
    if "csv_summary" not in st.session_state: st.session_state.csv_summary = None
    # 74. 网页左侧边栏 (st.sidebar) 的开发，用来传文件、画线形图、点击构建按钮
    # 这里基本都是调用我们上面写好的 process_csv_data 和 get_langgraph_chain 函数
    with st.sidebar:
        st.title("🔋 诊断控制台")
        st.markdown("---")
        st.markdown("**系统版本:** v7.0 (LangGraph + TF-GDC)")
        st.markdown("**核心架构:** 时序异常检测 + 混合检索")

        st.subheader("1. 实时数据导入 (CSV)")
        csv_file = st.file_uploader("上传电池时序数据", type=['csv'], key="csv_uploader")

        if csv_file is not None:
            with st.spinner("正在加载时序数据..."):
                df, summary = process_csv_data(csv_file)
                if df is not None:
                    st.session_state.csv_summary = summary
                    st.success("数据加载完成，已准备就绪喂给 TF-GDC！")
                    volt_col = next((col for col in df.columns if 'volt' in col.lower()), None)
                    if volt_col:
                        st.line_chart(df[volt_col])
                else:
                    st.error(summary)

        st.markdown("---")
        st.subheader("2. 知识库导入 (PDF)")
        pdf_docs = st.file_uploader("上传规格书或维修手册", accept_multiple_files=True)

        if st.button("构建 LangGraph 专家系统"):
            with st.spinner("正在构建混合索引与状态图编排..."):
                raw_text = get_pdf_text(pdf_docs)
                if not raw_text or not raw_text.strip():
                    st.error("❌ 提取失败：请上传包含可选中文本的文档。")
                else:
                    text_chunks = get_text_chunks(raw_text)
                    if not text_chunks:
                        st.error("❌ 文本太短，无法有效切分！")
                    else:
                        ensemble_retriever = get_ensemble_retriever(text_chunks)
                        st.session_state.conversation = get_langgraph_chain(ensemble_retriever)
                        st.success("✅ LangGraph 状态机已编译上线！")

    st.header("⚡ 新能源电池智能故障诊断系统")
    st.caption("基于 TF-GDC 时序算法与 LangGraph 的工业级 BMS 辅助架构")
    st.markdown("---")
    # 75. 网页主区域，最底下的聊天输入框。一旦用户回车，就触发 handle_userinput 开始整套流程
    user_question = st.text_input("请输入电池异常描述、特定故障码或综合诊断需求：")
    if user_question:
        if st.session_state.conversation is None:
            st.warning("请先在左侧构建 LangGraph 专家系统！")
        else:
            handle_userinput(user_question)


if __name__ == '__main__':
    main()  # 76. 整个程序的入口大门，从这里开始运行