from flask import Flask, request, jsonify, render_template_string
from pytorch_tabnet.tab_model import TabNetClassifier
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# ======= Load dataset and train TabNet =======

df = pd.read_csv("crop_recommendation.csv")

X = df.drop("label", axis=1).values
le = LabelEncoder()
y = le.fit_transform(df["label"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

tabnet_model = TabNetClassifier()
tabnet_model.fit(X_train, y_train, max_epochs=100, patience=20)

# ======= HTML Template =======

with open("template.html", "r") as f:
    html_template = f.read()

@app.route("/")
def home():
    return render_template_string(html_template)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [
            float(request.form.get(k)) for k in 
            ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        ]
        prediction = tabnet_model.predict(np.array([features]))[0]
        crop_name = le.inverse_transform([prediction])[0]
        return jsonify({"prediction": crop_name})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form.get('message', '').strip().lower()

    responses = {
        "what crops do i grow?": "Based on your inputs, rice may be suitable.",
        "how to improve soil?": "Use compost, green manure, crop rotation, and organic matter to improve soil quality.",
        "tell me about tabnet": "TabNet is a deep learning architecture designed for tabular data. It uses attention mechanisms for feature selection and offers interpretability.",
        "what is the best fertilizer?": "The best fertilizer depends on your soil test. Generally, NPK-based fertilizers like urea, DAP, and potash are common.",
        "how much rainfall is good for rice?": "Rice typically requires 1000-2000 mm of water depending on the variety and soil.",
        "suggest crops for dry area": "For dry regions, consider crops like millet, sorghum, chickpeas, and groundnut.",
        "suggest crops for wet area": "In wet areas, rice, jute, and sugarcane perform well.",
        "suggest crops for acidic soil": "Tea, potatoes, and pineapples grow well in acidic soil (pH < 6.5).",
        "suggest crops for alkaline soil": "Barley, cotton, and beets tolerate alkaline soil (pH > 7.5).",
        "how to reduce soil acidity?": "Apply agricultural lime or dolomite to neutralize acidic soils.",
        "how to reduce soil alkalinity?": "Use gypsum, organic compost, and sulfur to improve alkaline soils.",
        "how to improve crop yield?": "Use quality seeds, balanced fertilization, timely irrigation, and pest management.",
        "what is npk?": "NPK stands for Nitrogen (N), Phosphorus (P), and Potassium (K), essential macronutrients for plant growth.",
        "how does rainfall affect crops?": "Rainfall affects soil moisture, crop water needs, disease risk, and nutrient leaching.",
        "how to save water in farming?": "Drip irrigation, mulching, and rainwater harvesting help conserve water."
    }

    reply = responses.get(user_message, "Sorry, I don't understand. Try asking about crop types, soil, or farming tips.")
    return jsonify({'reply': reply})


if __name__ == "__main__":
    app.run(debug=True)
