import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

# sample dataset (simple)
data = {
    "Pregnancies":[6,1,8,1],
    "Glucose":[148,85,183,89],
    "BloodPressure":[72,66,64,66],
    "SkinThickness":[35,29,0,23],
    "Insulin":[0,0,0,94],
    "BMI":[33.6,26.6,23.3,28.1],
    "DPF":[0.627,0.351,0.672,0.167],
    "Age":[50,31,32,21],
    "Outcome":[1,0,1,0]
}

df = pd.DataFrame(data)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

model = LogisticRegression()
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))

print("Model created!")