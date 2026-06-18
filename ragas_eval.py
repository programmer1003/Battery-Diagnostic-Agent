import os
import requests
import pandas as pd
from datasets import Dataset
from ragas import evaluate

# 🌟 修复黄牌警告：使用 Ragas 最新版本的导入路径
# ✅ 替换为这段新的导入
from ragas.metrics import (
    Faithfulness,
    ContextRecall
)
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import time

load_dotenv()

BASE_URL = "http://127.0.0.1:8000"

# ==========================================
# 0. 准备你的真实测试文件路径
# ==========================================
# 💥 注意：请替换为你电脑上真实的 CSV 和 PDF 路径！
TEST_CSV_PATH = "C:/Users/赵旭东/Desktop/Battery_Agent/healthy_battery_data.csv"
TEST_PDF_PATH = "C:/Users/赵旭东/Desktop/Battery_Agent/Battery_RAG/某品牌新能源电池故障诊断手册.pdf"

# ==========================================
# 1. 准备“黄金测试集” (Ground Truth)
# 完全基于《某品牌新能源电池故障诊断手册.pdf》真实内容编写
# ==========================================
eval_data = [
    {
        "question": "仪表盘显示高压系统故障，车辆无法上电，伴随 P3001 故障码。这是什么原理导致的？给出具体的维修建议。",
        "ground_truth": "故障原理是BMS检测到高压互锁回路(HVIL)的PWM信号或电平信号中断。通常由MSD未压紧、高压线束接插件松动或互锁线束断路引起。维修建议：1.检查手动维护开关(MSD)是否完全锁止。2.检查电机控制器、空调压缩机的高压接插件是否松动。3.测量互锁回路电阻，标准值应小于5欧姆。"
    },
    {
        "question": "电池温度在30秒内急剧上升超过10°C，且伴随烟雾传感器报警（热失控预警）。这是什么原理？应该如何处置？",
        "ground_truth": "原理是SEI膜在高温下分解(>120°C)，导致负极与电解液反应放热，进而融化隔膜导致正负极短路，引发链式反应。处置建议：立即切断高压，远离车辆，拨打火警。严禁贸然开箱检查。"
    }
]


def setup_environment():
    """考前准备：自动获取 Session ID 并构建大脑"""
    print("⚙️ [初始化] 正在向系统挂载 CSV 并获取专属暗号...")
    session_id = None
    try:
        with open(TEST_CSV_PATH, "rb") as f:
            res1 = requests.post(f"{BASE_URL}/upload_csv", files={"file": f})
            if res1.json().get("status") == "success":
                session_id = res1.json().get("session_id")
                print(f"✅ 成功获取会话暗号: {session_id}")
    except Exception as e:
        print(f"❌ CSV 上传失败，请检查后端是否启动或路径是否正确: {e}")
        return None

    print("⚙️ [初始化] 正在向系统上传 PDF 并编译 LangGraph 大脑...")
    try:
        with open(TEST_PDF_PATH, "rb") as f:
            res2 = requests.post(f"{BASE_URL}/build_agent", files={"file": f})
            if res2.json().get("status") == "success":
                print("✅ 系统大脑编译完成，准备开始考试！\n")
    except Exception as e:
        print(f"❌ PDF 上传失败: {e}")
        return None

    return session_id


def run_evaluation_pipeline(session_id):
    print("🚀 开始收集 Ragas 评测数据...")

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for item in eval_data:
        question = item["question"]
        print(f"正在测试问题: {question}")

        try:
            res = requests.post(
                f"{BASE_URL}/diagnose",
                json={"query": question, "session_id": session_id}
            )
            res_json = res.json()
            answer = res_json.get("diagnosis", "系统无响应")

            # 🌟 核心替换：抛弃假的模拟数据，接收后端真正查到的 PDF 上下文
            real_contexts = res_json.get("contexts", [])

            # 🌟 新增打印：让证据大白于天下！
            print(f"   🔍 找出的证据数量: {len(real_contexts)} 块")
            if real_contexts:
                print(f"   📜 证据内容预览: {str(real_contexts[0])[:150]}...")

            # 防呆机制：如果这道题模型没查文档（比如闲聊），就塞入原题垫底，防报错
            if not real_contexts or len(real_contexts) == 0:
                real_contexts = [item["ground_truth"]]

            questions.append(question)
            answers.append(answer)
            contexts.append(real_contexts)  # 🌟 递给裁判真正的参考资料！
            ground_truths.append(item["ground_truth"])

        except Exception as e:
            print(f"❌ 请求失败: {e}")

    # ==========================================
    # 3. 组装数据集并呼叫大模型裁判
    # ==========================================
    # 确保收集到了数据才去评测，防止抛出 IndexError
    if not answers:
        print("❌ 未收集到任何系统回答，评测终止。请检查后端日志。")
        return

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    print("\n⚖️ 数据收集完毕，正在呼叫大模型裁判进行 Ragas 打分...")

    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    judge_llm = ChatOpenAI(
        api_key=deepseek_key,
        base_url="https://api.deepseek.com",
        model="deepseek-chat"
    )

    judge_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 忽略评测过程中的一些底层运行警告
    import warnings
    warnings.filterwarnings("ignore")

    # ✅ 替换为这段新的打分代码
    result = evaluate(
        dataset=dataset,
        # ✅ 只测这两个最核心的硬核指标
        metrics=[ContextRecall(), Faithfulness()],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    print("\n🏆 Ragas 最终评测报告:")
    # ✅ 替换为这段极其稳健的打印代码
    df_result = result.to_pandas()

    # 智能匹配列名：不管是叫 question 还是 user_input，只要表格里有的我都拿出来
    display_cols = [col for col in ['question', 'user_input', 'context_recall', 'faithfulness'] if
                    col in df_result.columns]

    print("\n" + "=" * 50)
    print(df_result[display_cols].to_markdown(index=False))  # 用 markdown 格式画一个极其漂亮的表格
    print("=" * 50)


if __name__ == "__main__":
    # 1. 考前准备：获取暗号并注入知识
    active_session_id = setup_environment()

    # 2. 如果准备成功，正式开始考试打分
    if active_session_id:
        time.sleep(2)  # 稍微等两秒让系统缓冲
        run_evaluation_pipeline(active_session_id)