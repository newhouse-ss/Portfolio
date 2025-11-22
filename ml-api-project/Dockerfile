# --- 1. 基础镜像 ---
# 相当于给电脑装一个纯净的 Python 3.9 系统
# "slim" 版本比较小，下载快
FROM python:3.9-slim

# --- 2. 设置工作目录 ---
# 相当于在容器里新建一个文件夹叫 /app，并且 cd 进去
# 之后所有的操作都在这个文件夹里进行
WORKDIR /app

# --- 3. 复制依赖文件 ---
# 把你电脑当前目录下的 requirements.txt 复制到容器里的 /app 目录下
COPY requirements.txt .

# --- 4. 安装依赖 ---
# 在容器里运行 pip install，安装 FastAPI, Pandas 等库
# --no-cache-dir 是为了不缓存安装包，让镜像体积更小
RUN pip install --no-cache-dir -r requirements.txt

# --- 5. 复制项目代码 ---
# 把你电脑当前目录下的所有文件（main.py, titanic_model.pkl 等）
# 全部复制到容器里的 /app 目录下
COPY . .

# --- 6. 声明端口 ---
# 告诉 Docker，这个程序默认使用 8000 端口
# (这行主要是给人看的注释，实际起作用的是后面的 CMD)
EXPOSE 8000

# --- 7. 启动命令 ---
# 容器启动时自动执行的命令
# uvicorn: 启动服务器
# main:app : 运行 main.py 里的 app 对象
# --host 0.0.0.0 : 允许外部访问 (必填，否则宿主机连不上)
# --port 8000 : 在 8000 端口运行
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]