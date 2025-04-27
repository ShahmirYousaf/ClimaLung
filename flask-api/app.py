from flask import Flask, request, jsonify
import requests
from dotenv import load_dotenv
import os
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from functools import lru_cache

app = Flask(__name__)

load_dotenv() 

API_KEY = os.getenv("OPEN_WEATHER_API_KEY")

MODEL_URL = "https://drive.google.com/uc?export=download&id=1mz1GmedHm4dPDjFV_eYV5gsim2737nGb" 
MODEL_PATH = "/tmp/lung_health_model.pkl"

@lru_cache(maxsize=1)
def get_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, 'wb') as f:
            f.write(response.content)
    
    return joblib.load(MODEL_PATH)

# Load the label encoder (if used during training)
label_encoder = LabelEncoder()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model = get_model()  # Model loads only when needed
        # Rest of your prediction code
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    # Get the input data from the frontend
    data = request.get_json()

    # Preprocess the input data as done during training
    input_data = pd.DataFrame([data])

    input_data['Exposure_to_Occupational_Hazards'] = label_encoder.fit_transform(input_data['Exposure_to_Occupational_Hazards'].fillna('No'))
    input_data['History_Of_Chronic_Respiratory_Diseases'] = label_encoder.fit_transform(input_data['History_Of_Chronic_Respiratory_Diseases'].fillna('No'))
    input_data['Diagnosed_with_Cancer'] = label_encoder.fit_transform(input_data['Diagnosed_with_Cancer'].fillna('No'))
    input_data['Lived_In_Highly_Polluted_Area'] = label_encoder.fit_transform(input_data['Lived_In_Highly_Polluted_Area'].fillna('No'))
    input_data['Shortness_Of_breath'] = label_encoder.fit_transform(input_data['Shortness_Of_breath'].fillna('No'))
    input_data['Coughed_Blood'] = label_encoder.fit_transform(input_data['Coughed_Blood'].fillna('No'))
    input_data['Ever_Smoked'] = label_encoder.fit_transform(input_data['Ever_Smoked'].fillna('No'))
    input_data['Persistent_Cough'] = label_encoder.fit_transform(input_data['Persistent_Cough'].fillna('No'))
    input_data['Age'] = pd.to_numeric(input_data['Age'], errors='coerce')
    input_data['Gender'] = label_encoder.fit_transform(input_data['Gender'].fillna('Male'))

    feature_columns = ['Age', 'Gender', 'Exposure_to_Occupational_Hazards', 'History_Of_Chronic_Respiratory_Diseases',
                       'Lived_In_Highly_Polluted_Area', 'Shortness_Of_breath', 'Coughed_Blood', 'Persistent_Cough','Ever_Smoked',
                       'PM2.5 (ug/m^3)', 'PM10 (ug/m^3)', 'AQI', 'NO2 (ppb)', 'SO2 (ppb)', 'CO (ppb)']
    
    input_data = input_data[feature_columns]

    # Make prediction using the Random Forest model
    prediction = model.predict(input_data)

    # Return the result as a JSON response
    if prediction[0] == 1:
        result = "Poor Lung Health"
    else:
        result = "Good Lung Health"
    
    return jsonify({'prediction': result})

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