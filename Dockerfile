# 1. 地基：轻量级 Python
FROM python:3.9-slim

# 2. 设定工作目录
WORKDIR /app

# 3. 安装 C++ 编译环境 (ChromaDB 和一些底层计算库需要它)
RUN apt-get update && apt-get install -y build-essential

# 🌟 4. 绝杀防坑：必须先强制安装 CPU 版 PyTorch！
# 提前占位，防止后续依赖自动下载庞大的 GPU 驱动版，把镜像体积从 2GB 压缩到几百MB
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 5. 拷贝依赖清单并安装业务库 (使用清华源提速)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 6. 把项目所有代码拷进集装箱
COPY . .

# 7. 暴露后端和前端的端口
EXPOSE 8000 8501