# main.py (前端 Streamlit)： 是医院大厅的接待护士。她负责长得好看、和病人聊天、收病历本。她自己不看病。
# api_main.py (后端 FastAPI)： 是躲在帘子后面的主治医生团队。他们不露面，但掌握着核心医术（TF-GDC 模型）和专家会诊流程（LangGraph 大脑）。
# requests 库： 是连接护士和医生的对讲机。
# Redis 数据库： 是医院的带锁储物柜。病人拿到的 session_id 就是开柜子的手牌号。
import os #当你写下 import os 的时候，你就是在告诉 Python：“我接下来要写的一些代码，超出了你（Python 语言本身）的计算能力范围。
            # 我需要直接和电脑的操作系统（比如 Windows、Mac 或 Linux）打交道，请你帮我把 os 这个工具箱拿过来。
import requests  # 🌟 新增：这是前端和后端“打电话”通讯的神器
import pandas as pd # 处理表格数据
import streamlit as st # 画网页的画笔
from dotenv import load_dotenv # 加载环境变量（隐藏密码用）
import json

# 导入你本地写的网页样式和头像
from htmlTemplates import css, bot_template, user_template # 导入好看的聊天气泡样式

# ==========================================
# 🌟 全局配置：指定后端 API 的“电话号码”
# ==========================================
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
# 作用： 告诉前端，后端的 FastAPI 医生住在本地电脑的 8000 房间。以后有事就往这里发消息。


def main():
    load_dotenv()
    st.set_page_config(page_title="新能源电池故障智能诊断系统", page_icon="🔋", layout="wide")
    st.write(css, unsafe_allow_html=True) # 把 CSS 样式刷到网页上

    # ==========================================
    # 🌟 状态管理 (Session State)：用来记住暗号和聊天记录
    # ==========================================
    # 知识点 st.session_state： 网页每次点击按钮都会刷新重置。用这个东西，可以让网页像人的记忆一样，记住之前发生过的事（比如存下后端发来的暗号）
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [] # 拿个小本本记录聊天记录
    if "session_id" not in st.session_state:
        st.session_state.session_id = None  # 极其重要：用来装后端的暗号
    if "agent_ready" not in st.session_state:
        st.session_state.agent_ready = False  # 记录大脑是否组装完毕
    # ==========================================
    # 侧边栏：上传区
    # ==========================================
    with st.sidebar:
        st.title("🔋 诊断控制台")
        st.markdown("---")
        st.markdown("**前端版本:** v8.0 (前后端分离微服务架构)")

        # -----------------------------------
        # 1. 上传 CSV 并获取暗号
        # -----------------------------------
        st.subheader("1. 实时数据导入 (CSV)")
        csv_file = st.file_uploader("上传电池时序数据", type=['csv'], key="csv_uploader")
        # 用户上传了文件，且还没拿手牌
        if csv_file is not None and st.session_state.session_id is None:
            with st.spinner("正在将数据传输至后端云存储(Redis)..."):
                try:
                    # 🌟 核心动作：前端用 requests 把文件发给后端接口
                    files = {"file": (csv_file.name, csv_file.getvalue(), "text/csv")}
                    response = requests.post(f"{API_BASE_URL}/upload_csv", files=files)
                    res_json = response.json()

                    if res_json.get("status") == "success":
                        # 🌟 绝杀：拿到暗号并存进网页兜里
                        st.session_state.session_id = res_json.get("session_id")
                        st.success(f"✅ {res_json.get('message')}")

                        # 顺手在前端画个图给用户看
                        csv_file.seek(0) # seek(0) 就是强制把阅读指针“拨回第 0 字节（文件的最开头）”。就像读完一本书后，重新翻回第一页，准备再读一遍。
                        df = pd.read_csv(csv_file)

                        # 1. 找出所有带有 'volt' 的列（不再只拿第一个了）
                        volt_cols = [col for col in df.columns if 'volt' in col.lower()]

                        if volt_cols:
                            # 2. 尝试寻找时间列，如果有，就把它设置为 X 轴
                            time_col = next(
                                (col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()), None)
                            if time_col:
                                df.set_index(time_col, inplace=True)

                            # 3. 防卡死机制：如果数据太长，只取最后 1000 行展示近期趋势
                            plot_data = df[volt_cols].tail(1000) if len(df) > 1000 else df[volt_cols]

                            # 4. 渲染多重折线图
                            st.line_chart(plot_data)
                    else:
                        st.error(f"后端报错: {res_json.get('message')}")
                except Exception as e:
                    st.error("无法连接到后端服务器，请确认 API 已启动！")

        # -----------------------------------
        # 2. 上传 PDF 并编译图谱大脑
        # -----------------------------------
        st.markdown("---")
        st.subheader("2. 知识库导入 (PDF)")
        pdf_file = st.file_uploader("上传单本规格书或维修手册", type=['pdf'])

        if st.button("构建 LangGraph 专家系统"):
            if pdf_file is None:
                st.warning("请先上传一本 PDF 手册！")
            else:
                with st.spinner("通知后端编译混合索引与状态图..."):
                    try:
                        files = {"file": (pdf_file.name, pdf_file.getvalue(), "application/pdf")}
                        data = {"session_id": st.session_state.session_id}  # <- 新增这一行
                        response = requests.post(f"{API_BASE_URL}/build_agent", files=files,data=data)
                        res_json = response.json()

                        if res_json.get("status") == "success":
                            st.session_state.agent_ready = True
                            st.success("✅ LangGraph 状态机已在后端编译上线！")
                        else:
                            st.error(f"后端报错: {res_json.get('message')}")
                    except Exception as e:
                        st.error("无法连接到后端服务器！")

    # ==========================================
    # 主界面：智能对话区
    # ==========================================
    st.header("⚡ 新能源电池智能故障诊断系统")
    st.caption("基于前后端分离架构：前端轻量展示，后端 TF-GDC + LangGraph 联合驱动")
    st.markdown("---")

    # 渲染历史聊天记录
    # 作用： 网页每次点击都会刷新，为了不让客人觉得“我刚说的话你怎么忘了”，护士每次都会先把小本本（chat_history）里的历史对话重新画在屏幕上。
    for q, a in st.session_state.chat_history:
        st.write(user_template.replace("{{MSG}}", q), unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}", a), unsafe_allow_html=True)

    # 用户输入框
    user_question = st.chat_input("请输入电池异常描述、特定故障码或综合诊断需求...")

    # 涉及知识点： 异常阻断。
    # 作用： 如果客人什么都没传（没有手牌号，医生也没读过书），上来就问“我的电池怎么了”，后端一定会崩溃报错。护士在这里直接把这种不合理的要求拦下来，根本不麻烦后端。这就是所谓的“前端防呆”。
    if user_question:
        # 防呆检查：必须有暗号，且大脑必须装好
        if not st.session_state.session_id:
            st.warning("⚠️ 请先在侧边栏上传 CSV 数据以获取会话暗号！")
        elif not st.session_state.agent_ready:
            st.warning("⚠️ 请先在侧边栏上传 PDF 并构建系统！")
        else:
            # 1. 立即把用户的问题显示在屏幕上
            st.write(user_template.replace("{{MSG}}", user_question), unsafe_allow_html=True)

            with st.spinner("后端大模型多步推理中，请稍候..."):
                try:
                    # 🌟 核心动作：带上问题和暗号，向后端发起 POST 请求
                    payload = {
                        "query": user_question,
                        "session_id": st.session_state.session_id
                    }
                    with requests.post(f"{API_BASE_URL}/diagnose_stream", json=payload, stream=True) as response:
                        # 给护士准备两个小黑板：一个写进度，一个写字
                        status_container = st.empty()
                        text_container = st.empty()
                        full_answer = ""

                        # 竖起耳朵听后端像流水一样传来的数据
                        for line in response.iter_lines():
                            if line:
                                decoded_line = line.decode('utf-8')
                                if decoded_line.startswith("data: "):
                                    data_str = decoded_line[6:]
                                    data_json = json.loads(data_str)

                                    # 如果是状态汇报
                                    if data_json["type"] == "status":
                                        status_container.info(data_json["content"])

                                    # 如果是正文打字
                                    elif data_json["type"] == "token":
                                        status_container.empty()  # 擦掉状态条
                                        full_answer += data_json["content"]
                                        # 结合你的 bot_template，动态刷新内容
                                        text_container.write(bot_template.replace("{{MSG}}", full_answer + "▌"),
                                                             unsafe_allow_html=True)

                        # 彻底接收完毕，去掉闪烁的光标 ▌
                        text_container.write(bot_template.replace("{{MSG}}", full_answer), unsafe_allow_html=True)

                        # 把最终对话存入小本本历史记录
                        st.session_state.chat_history.append((user_question, full_answer))

                except Exception as e:
                    st.error(f"网络请求失败，后端可能已宕机。详细错误: {e}")


if __name__ == '__main__':
    main()