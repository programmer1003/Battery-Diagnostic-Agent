import requests
import json
import time

# 服务器的基础地址
BASE_URL = "http://127.0.0.1:8000"

print("=========================================")
print("🚀 开始执行全链路自动化测试 (API Eval)")
print("=========================================\n")

# 🌟 新增：准备一个空变量，用来接住后端发来的暗号
current_session_id = None
# ---------------------------------------------------------
# 1. 测试 CSV 挂载接口
# ---------------------------------------------------------
print("▶️ [步骤 1] 正在上传电池时序数据...")
csv_path = "C:/Users/赵旭东/Desktop/Battery_Agent/healthy_battery_data.csv"  # 💥 请修改为你真实的 CSV 路径

try:
    with open(csv_path, "rb") as f:
        res1 = requests.post(f"{BASE_URL}/upload_csv", files={"file": f})
        # ✅ 正确做法：先解析并赋值给一个变量
        res1_json = res1.json()
        print("✅ 返回结果:", res1_json)
    # 🌟 关键动作：从返回结果中提取暗号并保存
    if res1_json.get("status") == "success":
        current_session_id = res1_json.get("session_id")
        print(f"🔑 成功获取会话暗号: {current_session_id}")
except FileNotFoundError:
    print("❌ 找不到 CSV 文件，请检查路径是否正确！")

time.sleep(1)

# ---------------------------------------------------------
# 2. 测试多文档 RAG 构建接口 (完全体核心)
# ---------------------------------------------------------
print("\n▶️ [步骤 2] 正在批量上传 PDF 手册并编译 LangGraph...")
# 💥 请修改为你真实的 PDF 路径，可以放一本，也可以放多本
pdf_path_1 = "C:/Users/赵旭东/Desktop/Battery_Agent/Battery_RAG/某品牌新能源电池故障诊断手册.pdf"
# pdf_path_2 = "C:/Users/赵旭东/Desktop/你的维修手册2.pdf"

try:
    # 🌟 修改：直接用简单的 files={"file": f} 格式，与 api_main.py 严格对齐
    with open(pdf_path_1, "rb") as f:
        res2 = requests.post(f"{BASE_URL}/build_agent", files={"file": f})
    print("✅ 返回结果:", res2.json())
except FileNotFoundError:
    print("❌ 找不到 PDF 文件，请检查路径是否正确！")

time.sleep(1)

# ---------------------------------------------------------
# 3. 测试智能诊断接口
# ---------------------------------------------------------
print("\n▶️ [步骤 3] 正在发起智能诊断请求...")

# 🌟 防呆设计：如果没有拿到暗号，就不要往下走了
if not current_session_id:
    print("❌ 缺少 session_id 暗号，无法发起诊断，测试终止！")
else:
    # 🌟 关键动作：把步骤 1 拿到的暗号，放在请求体里交还给后端
    query_data = {
        "query": "请用深度学习模型检查当前电池数据，如果发现重构误差异常，请从知识库中检索‘微短路’的原因和解决办法。",
        "session_id": current_session_id
    }

    res3 = requests.post(f"{BASE_URL}/diagnose", json=query_data)

    print("\n🎯 最终诊断报告:")
    print(json.dumps(res3.json(), indent=4, ensure_ascii=False))

print("\n🎉 全链路自动化测试执行完毕！")