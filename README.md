# 🔋 新能源电池智能诊断 Agent (Battery Diagnostic Agent)

本项目是一个结合了 **深度学习时序预测 (TF-GDC)** 与 **大模型智能体 (Agent)** 的企业级辅助诊断原型系统。

## 🌟 核心架构设计

1. **网关层 (FastAPI)**：基于异步机制重构，提供标准 RESTful API，完美隔离 I/O 阻塞与 CPU 密集型计算。
2. **大脑层 (LangGraph)**：打破传统 AgentExecutor 的黑盒限制，采用状态机 (State Machine) 架构，实现“读取异常-> 评估阈值 -> 条件检索”的强制逻辑闭环。
3. **知识底座 (Hybrid RAG)**：集成 ChromaDB 稠密向量与 BM25 稀疏检索，通过哈希去重，彻底解决电池垂直领域专有名词的“语义塌陷”问题。
4. **物理引擎 (PyTorch)**：底层挂载自研时序预测模型，提取真实 MSE 误差，为 Agent 提供硬核的物理数值先验。

## 🚀 快速启动 (Quick Start)

### 1. 环境安装
```bash
pip install -r requirements.txt