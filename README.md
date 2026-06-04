# 🔋 新能源电池智能诊断 Agent (Battery Diagnostic Agent)

本项目是一个结合了 **深度学习时序预测 (TF-GDC)** 与 **大模型智能体 (Agent)** 的企业级辅助诊断原型系统。

## 🌟 核心架构设计

1. **网关层 (FastAPI)**：基于异步机制重构，提供标准 RESTful API，完美隔离 I/O 阻塞与 CPU 密集型计算。
2. **大脑层 (LangGraph)**：打破传统 AgentExecutor 的黑盒限制，采用状态机 (State Machine) 架构，实现“读取异常-> 评估阈值 -> 条件检索”的强制逻辑闭环。
3. **知识底座 (Hybrid RAG)**：集成 ChromaDB 稠密向量与 BM25 稀疏检索，通过哈希去重，彻底解决电池垂直领域专有名词的“语义塌陷”问题。
4. **物理引擎 (PyTorch)**：底层挂载自研时序预测模型，提取真实 MSE 误差，为 Agent 提供硬核的物理数值先验。

## 📸 系统演示 (Demo)

**1. 前端智能诊断交互界面**
![System Demo](docs/demo.png)
- **支持一键上传**电池时序数据与维修手册，实时渲染 Markdown 格式的高管级诊断报告。

**2. Agent 底层物理预测引擎 (TF-GDC)**
![Model Architecture](docs/model.png)
- **精准监控** Agent 截取滑动窗口、调用深度学习提取 MSE 误差的多步推理全过程。

---

## 🚀 快速启动 (Quick Start)

本系统采用彻底的前后端分离架构，并强依赖 Redis 进行多用户并发状态隔离。提供以下两种部署方式：

### 方案 A：企业级 Docker 一键部署（推荐 🌟）
无需配置复杂的 Python 与 Redis 环境，实现开箱即用：

**1. 克隆仓库**
```bash
git clone https://github.com/programmer1003/Battery-Diagnostic-Agent.git
cd Battery-Diagnostic-Agent
```
```bash
**2. 一键拉起微服务集群（包含 Redis 缓存、FastAPI 后端)**

docker-compose up -d --build
```


### 方案 B：本地双端化运行 (适合开发调试)
```bash
 **1. 环境准备**

pip install -r requirements.txt
```

 **2. 启动底层缓存基座 (Redis)**
请确保本地已安装并启动 Redis 服务器（默认端口 6379）。
注：系统已配置降级兼容 RESP2 协议，支持 Windows 移植版 Redis 5.0


** 3. 启动大脑微服务后端 (Terminal 1)**
```bash
# 热重载模式启动 FastAPI 引擎

uvicorn api_main:app --reload
```
**当看到 ✅ TF-GDC 模型引擎点火成功！ 时，说明底层张量计算与 Agent 架构已就绪**

** 4. 启动可视化接待前厅 (Terminal 2)**

```bash
# 启动 Streamlit UI

streamlit run main.py
```
**浏览器会自动打开 http://localhost:8501，即可开始体验。**


**📂 核心目录结构**
```bash
├── api_main.py          # FastAPI 后端核心微服务与 LangGraph 编排逻辑
├── main.py              # Streamlit 前端交互 UI
├── test_api.py          # 基于 requests 的自动化全链路测试脚本
├── docker-compose.yml   # 容器化集群编排文件
├── Dockerfile           # 后端环境构建脚本
├── requirements.txt     # 核心依赖清单
├── docs/                # 存放系统架构图与模型结构图
└── Battery_RAG/         # 存放 ChromaDB 向量库与生僻故障码的维修手册 PDF
```
