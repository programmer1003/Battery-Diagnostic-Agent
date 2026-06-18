# 🔋 新能源电池智能诊断 Agent (Battery Diagnostic Agent)

本项目是一个结合了深度学习时序异常检测 (TF-GDC) 与多智能体协作 (Multi-Agent) 的工业级故障诊断系统。系统打通了从底层传感器数据预警到结构化维修方案生成的全链路，并针对大模型落地中的幻觉控制、并发死锁与状态隔离问题进行了深度工程治理。

## 🌟 核心架构设计

1. **大脑编排 (LangGraph)**:基于 Actor-Critic 架构构建带有 Reviewer 节点的自我纠错图谱，设计动态状态嗅探与 thread_id 隔离机制，彻底攻克大模型工具调用的“递归死锁”与高并发重连导致的记忆错乱 (HTTP 400) 难题。
2. **知识底座 (Advanced RAG)**：集成 ChromaDB 稠密向量与 BM25 稀疏检索，前置 LLM 意图重写，后置 BGE-Reranker 交叉编码器进行二次打分，解决电池垂直领域长尾生僻故障码召回精度低的问题。
3. **物理引擎 (PyTorch)**：底层挂载自研 TF-GDC 时序异常检测模型，通过大模型工具调用 (Tool Calling) 动态提取滑动窗口 MSE 误差，为 Agent 提供硬核的跨模态物理数值先验。
4. **网关与流式 (FastAPI & SSE)**：结合 Redis 缓存与 asyncio.to_thread 异步线程池剥离文件 I/O 阻塞；设计“状态播报+结果生成”的双段式 SSE 协议，化解大模型反复自省与前端文字闪烁撤回的体验冲突。
5.**监控评测 (LangSmith & Ragas)**：全链路无感追踪 Agent 节点流转、工具延迟与 Token 消耗；底层挂载 Ragas 自动化盲测流水线，量化系统 Context Recall 与抗幻觉核心指标。

## 📸 系统监控与交互演示 (Demo)

**1. 前端智能诊断交互界面**
![System Demo](docs/demo.png)
- **支持一键上传**电池时序数据与维修手册，实时渲染 Markdown 格式的高管级诊断报告。

**2. 核心链路追踪图 (LangSmith Observability)**
![Model Architecture](docs/model.png)
- **精准监控**结构化监控 Agent 思考路径、Hybrid RAG 双路耗时及重排分数。

---

## 🚀 快速启动 (Quick Start)

本系统采用彻底的前后端分离架构，并强依赖 Redis 进行多用户并发状态隔离。提供以下两种部署方式：

**0. 环境配置准备**
```bash
# 大模型驱动 (本系统使用 DeepSeek 兼容 OpenAI 格式)
DEEPSEEK_API_KEY=your_api_key_here

# LangSmith 链路追踪配置 (必填)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=Battery_Agent_Prod
LANGCHAIN_API_KEY=your_langsmith_key_here
```

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
├── api_main.py          # FastAPI 后端核心：LangGraph 图谱流转、SSE 接口、工具挂载
├── main.py              # Streamlit 前端交互：防呆逻辑、动态图表、流式对讲解析
├── ragas_eval.py        # Ragas 自动化评测流水线脚本
├── tf_gdc.py            # 物理引擎：时序异常检测神经网络结构
├── docker-compose.yml   # 容器化集群微服务编排文件
├── Dockerfile           # 统一基础镜像构建脚本 (内置 PyTorch 瘦身逻辑)
├── requirements.txt     # 全局核心依赖清单
├── .env.example         # 环境变量配置模板
└── docs/                # 系统架构图、LangSmith 监控截图等资料
```
