from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

# Load model correctly
model_path = os.path.join(os.path.dirname(__file__), "model", "model.pkl")
model = pickle.load(open(model_path, "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form.to_dict()   # 👈 store input values

    data = [float(x) for x in form_data.values()]
    final = [data]

    prediction = model.predict(final)[0]
    prob = model.predict_proba(final)[0][1]

    if prediction == 1:
        result = f"Diabetic (Risk: {round(prob*100,2)}%)"
    else:
        result = f"Not Diabetic (Risk: {round(prob*100,2)}%)"

    return render_template('index.html',
                           prediction_text=result,
                           form_data=form_data)   # 👈 send back

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)