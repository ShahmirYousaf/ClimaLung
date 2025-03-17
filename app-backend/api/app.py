from flask import Flask, request, jsonify
import requests
from dotenv import load_dotenv
import os
from flask_vercel import Vercel

app = Flask(__name__)

load_dotenv() 

API_KEY = os.getenv("OPEN_WEATHER_API_KEY")

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

# Expose the Flask app for Vercel deployment
application = app