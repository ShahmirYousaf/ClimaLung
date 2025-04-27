from flask import Flask, request, jsonify
import requests
from dotenv import load_dotenv
import os
import onnxruntime 
import numpy as np
from functools import lru_cache

app = Flask(__name__)

load_dotenv() 

API_KEY = os.getenv("OPEN_WEATHER_API_KEY")

MODEL_URL = "https://drive.google.com/uc?export=download&id=1mz1GmedHm4dPDjFV_eYV5gsim2737nGb" 
MODEL_PATH = "/tmp/lung_health_model.pkl"

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

# if __name__ == '__main__':
#     app.run(port=5000)

    
if __name__ == '__main__':
    app.run()

# Expose the Flask app for Vercel deployment
# application = app