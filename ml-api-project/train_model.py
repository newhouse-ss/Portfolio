import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

#Data Preparation
data = {
    'pclass': [1, 3, 3, 1, 2, 2, 3, 1],
    'age': [22, 35, 20, 50, 28, 45, 10, 60],
    'fare': [7.25, 8.05, 8.05, 50.0, 13.0, 26.0, 7.0, 80.0],
    'survived': [0, 0, 0, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)
X = df[['pclass', 'age', 'fare']]
y = df['survived']

#Train Model
print("Training model...")
model = RandomForestClassifier(n_estimators = 10, random_state = 42)
model.fit(X, y)

#Save Model
joblib.dump(model, "titanic_model.pkl")#joblib 是一个 Python 库，专门用来把内存里的 Python 对象（比如训练好的模型）保存成文件，或者从文件加载回内存。它比 Python 自带的 pickle 处理大数据（如 NumPy 数组）更快。
print("✅ Model saved as 'titanic_model.pkl'")