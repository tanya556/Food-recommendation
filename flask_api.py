# flask_api.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import logging

# Setup logging
logging.basicConfig(level=logging.ERROR)

# Load model and data
MODEL_FILE = 'nutrition_recommendation_model.pkl'
DATA_FILE = 'labeled_nutrition.csv'

model = joblib.load(MODEL_FILE)
df = pd.read_csv(DATA_FILE)

# Features used in the model
FEATURES = ['Caloric Value', 'Fat', 'Carbohydrates', 'Sugars', 'Protein', 'Sodium']

# Flask app
app = Flask(__name__)
CORS(app)  # Allow requests from other origins like Laravel/React

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()

        # Validate JSON structure
        if not data or 'health_condition' not in data:
            return jsonify({'error': 'Missing health_condition field.'}), 400

        # Clean input
        health_condition = data.get('health_condition', '').strip().lower()

        # Validate health condition
        if health_condition not in ['diabetes', 'hypertension', 'obesity']:
            return jsonify({'error': 'Invalid health condition. Please choose diabetes, hypertension, or obesity.'}), 400

        # Prepare input data
        X = df[FEATURES]

        # Predict
        y_pred = model.predict(X)

        # Validate prediction shape
        if len(y_pred.shape) != 2 or y_pred.shape[1] != 3:
            raise ValueError("Model output shape is incorrect. Expected 3 columns.")

        # Map predictions
        prediction_df = pd.DataFrame(y_pred, columns=['good_for_diabetes', 'good_for_hypertension', 'good_for_obesity'])

        # Combine food names with predictions
        result_df = pd.concat([df['food'], prediction_df], axis=1)

        # Filter recommendations
        if health_condition == 'diabetes':
            recommended = result_df[result_df['good_for_diabetes'] == True]['food']
        elif health_condition == 'hypertension':
            recommended = result_df[result_df['good_for_hypertension'] == True]['food']
        else:  # obesity
            recommended = result_df[result_df['good_for_obesity'] == True]['food']

        # Return top 5
        return jsonify({
            'recommendations': recommended.head(5).tolist()
        })

    except Exception as e:
        logging.error(f"Error in /recommend: {str(e)}")
        return jsonify({'error': 'Internal server error.'}), 500

@app.route('/')
def index():
    return jsonify({'message': 'AI Food Recommendation API is running 🚀'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
