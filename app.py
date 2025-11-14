# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load saved model
model = joblib.load('house_price_model.pkl')

@app.route('/')
def home():
    return "🏠 House Price Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    size = data['size']
    bedrooms = data['bedrooms']
    
    prediction = model.predict([[size, bedrooms]])[0]
    return jsonify({'predicted_price': prediction})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)

