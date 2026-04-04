import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# SIMPLE DATA
data = {
    "area": [1000, 1500, 2000, 2500, 3000],
    "bedrooms": [2, 3, 3, 4, 4],
    "price": [20, 30, 40, 50, 60]
}

df = pd.DataFrame(data)

X = df[["area", "bedrooms"]]
y = df["price"]

# TRAIN MODEL
model = LinearRegression()
model.fit(X, y)

# SAVE MODEL
joblib.dump(model, "model.pkl")

print("Model trained and saved!")