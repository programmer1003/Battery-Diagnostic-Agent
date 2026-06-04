# 1. 选择一个包含 Python 环境的基础镜像 (瘦身版)
FROM python:3.9-slim

# 2. 告诉容器，我们的工作目录叫 /app
WORKDIR /app

# 3. 把你电脑当前文件夹里的所有代码，全抄到容器的 /app 里面
COPY . /app

# 4. 安装你的各种库 (需要在同目录建一个 requirements.txt)
# 针对 PyTorch，通常会指定专门的源来减小体积
RUN pip install fastapi uvicorn pandas numpy redis langchain_openai langchain_community PyPDF2
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu 

# 5. 暴露 8000 端口
EXPOSE 8000

# 6. 容器启动时，执行你熟悉的那个命令！
CMD ["uvicorn", "api_main:app", "--host", "0.0.0.0", "--port", "8000"]