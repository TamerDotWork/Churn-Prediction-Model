from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load model and transformer
model = joblib.load('log_reg.pkl')
transformer = joblib.load('transformer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        logging.info(f"Received data: {data}")
        df = pd.DataFrame(data, index=[0])
        
        # Validate input data
        if df.isnull().values.any():
            raise ValueError("Input data contains null values")
        
        transformed_data = transformer.transform(df)
        logging.info(f"Transformed data: {transformed_data}")
        prediction = model.predict(transformed_data)[0]
        probability = model.predict_proba(transformed_data)[0][1]
        
        result = {
            'prediction': 'Churn' if prediction == 1 else 'Not Churn',
            'probability': round(probability, 2)
        }
        
        return jsonify(result)
    except ValueError as ve:
        logging.error(f"Validation error: {ve}")
        return jsonify({'error': 'Invalid input data'}), 400
    except Exception as e:
        logging.error(f"Error during prediction: {e}", exc_info=True)
        return jsonify({'error': 'Internal Server Error'}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)