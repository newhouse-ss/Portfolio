from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import sqlite3
from datetime import datetime

# API initialization
app = FastAPI(title="Titanic ML API with DB")

# ---Database Settings---
DB_NAME = "titanic_logs.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
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

init_db()

# ---Load Model---
if not os.path.exists("titanic_model.pkl"):
    raise RuntimeError("Model not found! Run train_model.py first.")
model = joblib.load("titanic_model.pkl")

# ---Define Input Template---
class Passenger(BaseModel):
    pclass: int
    age: float
    fare: float

@app.get("/")
def home():
    return {"status": "running", "message": "Titanic API with Database is ready!"}

@app.post("/predict")
def predict(passenger: Passenger):
    data = pd.DataFrame([passenger.model_dump()])
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
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
    
    conn.commit()
    conn.close()
    
    return {
        "survived_prediction": int(prediction),
        "survival_probability": round(float(probability), 4),
        "log_status": "Saved to Database"
    }

@app.get("/logs")
def get_logs():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM prediction_logs ORDER BY id DESC LIMIT 5")
    logs = cursor.fetchall()
    conn.close()
    return {"recent_logs": logs}