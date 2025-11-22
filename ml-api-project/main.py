from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import sqlite3
from datetime import datetime

# 初始化 API
app = FastAPI(title="Titanic ML API with DB")

# --- 1. 数据库设置 ---
DB_NAME = "titanic_logs.db"

def init_db():
    """初始化数据库：如果表不存在，就建一张表"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    # 建表 SQL 语句 (标准的 SQL 语法)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pclass INTEGER,
            age REAL,
            fare REAL,
            prediction INTEGER,
            probability REAL,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

# 服务启动时，先运行一次建表
init_db()

# --- 2. 加载模型 ---
if not os.path.exists("titanic_model.pkl"):
    raise RuntimeError("Model not found! Run train_model.py first.")
model = joblib.load("titanic_model.pkl")

# --- 3. 定义数据格式 ---
class Passenger(BaseModel):
    pclass: int
    age: float
    fare: float

@app.get("/")
def home():
    return {"status": "running", "message": "Titanic API with Database is ready!"}

@app.post("/predict")
def predict(passenger: Passenger):
    # --- A. 数据处理与预测 ---
    data = pd.DataFrame([passenger.model_dump()])
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]
    
    # --- B. 写入数据库 (Log) ---
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # 插入数据的 SQL 语句
    insert_query = """
        INSERT INTO prediction_logs (pclass, age, fare, prediction, probability, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    """
    cursor.execute(insert_query, (
        passenger.pclass,
        passenger.age,
        passenger.fare,
        int(prediction),
        float(probability),
        datetime.now().isoformat()
    ))
    
    conn.commit() # 提交保存
    conn.close()  # 关闭连接
    
    # --- C. 返回结果 ---
    return {
        "survived_prediction": int(prediction),
        "survival_probability": round(float(probability), 4),
        "log_status": "Saved to Database"
    }

# 新增一个接口：查看所有历史记录 (方便你验证)
@app.get("/logs")
def get_logs():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM prediction_logs ORDER BY id DESC LIMIT 5")
    logs = cursor.fetchall()
    conn.close()
    return {"recent_logs": logs}