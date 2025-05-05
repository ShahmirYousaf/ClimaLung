from flask import Flask, request, jsonify
import requests
from dotenv import load_dotenv
import os
import onnxruntime 
import pandas as pd
import numpy as np
from models.airquality import pm25_prediction
from flask_cors import CORS

from functools import lru_cache

app = Flask(__name__)

## NEEDED IF USING LOCALLY

# CORS(app, resources={
#     r"/*": {
#         "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
#         "methods": ["GET", "POST", "OPTIONS"],
#         "allow_headers": ["Content-Type"]
#     }
# })

load_dotenv() 

API_KEY = os.getenv("OPEN_WEATHER_API_KEY")

MODEL_URL = "https://drive.google.com/uc?export=download&id=1mz1GmedHm4dPDjFV_eYV5gsim2737nGb" 
MODEL_PATH = "/tmp/lung_health_model.pkl"

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Air Quality API is running",
        "endpoints": {
            "predict_pm25": "POST /predict_pm25",
            "download_models": "GET /download_models" 
        }
    })

@app.route('/download_models', methods=['GET'])
def download_models():
    """Endpoint to manually trigger model downloads"""
    try:
        # Create properly formatted SINGLE-ROW DataFrame without list values
        test_data = pd.DataFrame({
            "Max Temperature (F)": 72.0,
            "Avg Temperature (F)": 68.5,
            "Min Temperature (F)": 65.0,
            "Max Humidity (%age)": 40,
            "Avg Humidity (%age)": 30,
            "Min Humidity (%age)": 20,
            "AQI": 12,
            "PM10 (ug/m^3)": 10.5,
            "O3 (ppb)": 40.0,
            "SO2 (ppb)": 0.5,
            "NO2 (ppb)": 5.0,
            "CO (ppb)": 0.2,
            "Day": "Wednesday"
        }, index=[0])  # index=[0] ensures single-row DataFrame
        
        # This will trigger model downloads
        prediction = pm25_prediction(test_data)
        
        return jsonify({
            'status': 'success',
            'message': 'Models are ready for use',
            'note': 'Models were downloaded automatically on first prediction'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/predict_pm25', methods=['POST'])
def predict_pm25():
    try:
        input_data = request.get_json()
        new_data = pd.DataFrame([input_data])  # Single row DataFrame
        
        prediction = pm25_prediction(new_data)
        
        if prediction is not None:
            return jsonify({
                'status': 'success',
                'prediction': prediction,
                'units': 'μg/m³'
            }), 200
        else:
            return jsonify({
                'status': 'error',
                'message': 'Prediction failed'
            }), 500
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400



@lru_cache(maxsize=1)
def get_model():
    """Download and cache the ONNX model"""
    if not os.path.exists(MODEL_PATH):
        print("Downloading ONNX model...")
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, 'wb') as f:
            f.write(response.content)
    
    # Create inference session
    return onnxruntime.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])

def preprocess_input(data):
    """Convert input data to numpy array (assuming pre-encoded values)"""
    features = [
        'Age',
        'Gender',                       
        'Exposure_to_Occupational_Hazards', 
        'History_Of_Chronic_Respiratory_Diseases',
        'Lived_In_Highly_Polluted_Area',
        'Shortness_Of_breath',
        'Coughed_Blood',
        'Persistent_Cough',
        'Ever_Smoked',
        'PM2.5 (ug/m^3)',
        'PM10 (ug/m^3)',
        'AQI',
        'NO2 (ppb)',
        'SO2 (ppb)',
        'CO (ppb)'
    ]
    
    # Create array with default fallback values
    input_array = np.array([
        float(data.get(feature, 0)) if feature in ['Age', 'PM2.5 (ug/m^3)', 'PM10 (ug/m^3)', 
                                                 'AQI', 'NO2 (ppb)', 'SO2 (ppb)', 'CO (ppb)']
        else int(data.get(feature, 0))  # For categorical features (0/1/2 encoded)
        for feature in features
    ], dtype=np.float32).reshape(1, -1)
    
    return input_array

@app.route('/predict', methods=['POST'])
def predict():
   try:
        # Get and validate input
        app.logger.info("Received predict_pm25 request")
        input_data = request.get_json()
        app.logger.info(f"Input data: {input_data}")
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        # Preprocess input
        input_array = preprocess_input(data)
        
        # Get model and predict
        model = get_model()
        prediction = model.run(None, {'float_input': input_array})[0]
        
        # Return result
        result = "Poor Lung Health" if prediction[0] == 1 else "Good Lung Health"
        return jsonify({'prediction': result})

   except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_aqi(lat, lon):
    url = f'http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}'
    response = requests.get(url)
    if response.status_code == 200:
        air_data = response.json()

        # Extract AQI from the API response (the key is 'list' -> 'main' -> 'aqi')
        aqi = air_data['list'][0]['main']['aqi']

        # Convert AQI value to a readable string
        if aqi == 1:
            aqi_text = "Good"
        elif aqi == 2:
            aqi_text = "Fair"
        elif aqi == 3:
            aqi_text = "Moderate"
        elif aqi == 4:
            aqi_text = "Poor"
        elif aqi == 5:
            aqi_text = "Very Poor"
        
        return aqi_text
    else:
        return None

@app.route('/webhook', methods=['POST'])
def webhook():
    print("Webhook endpoint hit!")
    req = request.get_json()
    parameters = req.get('queryResult').get('parameters')
    lat = parameters.get('latitude')
    lon = parameters.get('longitude')

    aqi = get_aqi(lat, lon)
    if aqi:
        fulfillment_text = f'The current Air Quality Index (AQI) at the specified location is {aqi}.'
    else:
        fulfillment_text = 'I am unable to retrieve the AQI for the specified location at this time.'

    return jsonify({'fulfillmentText': fulfillment_text})

## NEEDED IF USING LOCALLY
# if __name__ == '__main__':
#     app.run(port=5000)


if __name__ == '__main__':
    app.run()
    
    ## (NOT NEEDED FOR NOW)
    
    # if not os.path.exists('air_quality_models'):
    #     os.makedirs('air_quality_models')
    
    # app.run(host='0.0.0.0', port=5000, debug=True)
    # application = app  # For Vercel deployment

# Expose the Flask app for Vercel deployment
application = app