import os
import requests  # 🌟 新增：这是前端和后端“打电话”通讯的神器
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# 导入你本地写的网页样式和头像
from htmlTemplates import css, bot_template, user_template

# ==========================================
# 🌟 全局配置：指定后端 API 的“电话号码”
# ==========================================
API_BASE_URL = "http://127.0.0.1:8000"


def main():
    load_dotenv()
    st.set_page_config(page_title="新能源电池故障智能诊断系统", page_icon="🔋", layout="wide")
    st.write(css, unsafe_allow_html=True)

    # ==========================================
    # 🌟 状态管理 (Session State)：用来记住暗号和聊天记录
    # ==========================================
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
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

                        # 顺手在前端画个图给用户看（读取本地文件内容画图，不占用后端算力）
                        csv_file.seek(0)
                        df = pd.read_csv(csv_file)
                        volt_col = next((col for col in df.columns if 'volt' in col.lower()), None)
                        if volt_col:
                            st.line_chart(df[volt_col])
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
                        response = requests.post(f"{API_BASE_URL}/build_agent", files=files)
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
    for q, a in st.session_state.chat_history:
        st.write(user_template.replace("{{MSG}}", q), unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}", a), unsafe_allow_html=True)

    # 用户输入框
    user_question = st.chat_input("请输入电池异常描述、特定故障码或综合诊断需求...")

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
                    response = requests.post(f"{API_BASE_URL}/diagnose", json=payload)
                    res_json = response.json()

                    if res_json.get("status") == "success":
                        answer = res_json.get("diagnosis")
                        # 2. 把后端的回答显示在屏幕上，并存入历史记录
                        st.write(bot_template.replace("{{MSG}}", answer), unsafe_allow_html=True)
                        st.session_state.chat_history.append((user_question, answer))
                    else:
                        st.error(f"诊断失败: {res_json.get('message')}")
                except Exception as e:
                    st.error("网络请求失败，后端可能已宕机。")


if __name__ == '__main__':
    main()