import React, { useState, useEffect } from 'react';
import './AirQualityAnalysis.css';
import Sidebar from '../../Components/Sidebar/Sidebar';
import { Line } from 'react-chartjs-2';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import L from 'leaflet';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

//// (USED FOR VERCEL)
const API_BASE_URL = 'https://clima-lung-bot-api.vercel.app';

//// USED FOR LOCAL
//const API_BASE_URL = 'http://127.0.0.1:5000';

const DefaultIcon = new L.Icon({
  iconUrl: require('leaflet/dist/images/marker-icon.png'),
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41],
});

L.Marker.prototype.options.icon = DefaultIcon;

const AirQualityAnalysis = () => {
  const [location, setLocation] = useState(null);
  const [error, setError] = useState(null);
  const [airQualityData, setAirQualityData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [cityName, setCityName] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [predictionLoading, setPredictionLoading] = useState(false);
  const [predictionError, setPredictionError] = useState(null);
  const [progress, setProgress] = useState(0);
  const [showPredictionButton, setShowPredictionButton] = useState(true);

  const getLocation = () => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const { latitude, longitude } = position.coords;
          setLocation({ latitude, longitude });
          fetchAirQualityData(latitude, longitude);
          fetchCityName(latitude, longitude);
        },
        (err) => setError('Unable to retrieve your location')
      );
    } else {
      setError('Geolocation is not supported by this browser.');
    }
  };

  const handleGetPrediction = async () => {
    setShowPredictionButton(false);
    setPredictionLoading(true);
    setPredictionError(null);

    try {
      // Directly call the prediction API without downloading models separately
      await fetchPrediction(location.latitude, location.longitude);
    } catch (error) {
      setPredictionError(error.message);
      setShowPredictionButton(true);
    } finally {
      setPredictionLoading(false);
    }
  };

  const fetchPrediction = async (latitude, longitude) => {
    try {
      const weatherResponse = await fetch(
        `https://api.openweathermap.org/data/2.5/weather?lat=${latitude}&lon=${longitude}&appid=${process.env.REACT_APP_OPEN_WEATHER_API_KEY}`
      );
      
      const weatherData = await weatherResponse.json();

      // Get air quality data from API-Ninjas instead
      const airQualityResponse = await fetch(
        `https://api.api-ninjas.com/v1/airquality?lat=${latitude}&lon=${longitude}`,
        {
          headers: { 'X-Api-Key': process.env.REACT_APP_API_NINJAS_KEY }
        }
      );
      const airQualityData = await airQualityResponse.json();

      const inputData = {
        "Max Temperature (F)": [(weatherData.main.temp_max - 273.15) * 9 / 5 + 32],
        "Avg Temperature (F)": [(weatherData.main.temp - 273.15) * 9 / 5 + 32],
        "Min Temperature (F)": [(weatherData.main.temp_min - 273.15) * 9 / 5 + 32],
        "Max Humidity (%age)": [weatherData.main.humidity + 10],
        "Avg Humidity (%age)": [weatherData.main.humidity],
        "Min Humidity (%age)": [weatherData.main.humidity - 10],
        "AQI": [airQualityData.overall_aqi], // Using API-Ninjas AQI
        "PM10 (ug/m^3)": [airQualityData.PM10?.concentration || 0],
        "O3 (ppb)": [airQualityData.O3?.concentration || 0],
        "SO2 (ppb)": [airQualityData.SO2?.concentration || 0],
        "NO2 (ppb)": [airQualityData.NO2?.concentration || 0],
        "CO (ppb)": [airQualityData.CO?.concentration || 0],
        "Day": [new Date().toLocaleString('en-us', { weekday: 'long' })]
      };

      const response = await fetch(`${API_BASE_URL}/predict_pm25`, {
        method: 'POST',
        
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(inputData)
      });

      const data = await response.json();
      setPrediction(data);
    } catch (error) {
      throw error;
    }
  };
  // Function to fetch air quality data from an API using the user's coordinates
  const fetchAirQualityData = async (latitude, longitude) => {
    setLoading(true);
    try {
      // Using API-Ninjas instead of OpenWeatherMap
      const response = await fetch(
        `https://api.api-ninjas.com/v1/airquality?lat=${latitude}&lon=${longitude}`,
        {
          headers: { 'X-Api-Key': process.env.REACT_APP_API_NINJAS_KEY }
        }
      );
      
      const data = await response.json();
      

      if (data.overall_aqi) {
        setAirQualityData({
          aqi: data.overall_aqi,
          components: {
            pm2_5: data['PM2.5']?.concentration,
            pm10: data.PM10?.concentration,
            no2: data.NO2?.concentration,
            so2: data.SO2?.concentration,
            o3: data.O3?.concentration,
            co: data.CO?.concentration
          },
          labels: ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'],
          datasets: [
            {
              label: 'PM2.5',
              data: [data.PM2_5?.concentration, 40, 45, 38, 50, 55, 60],
              borderColor: 'rgb(36, 130, 166)',
              fill: false,
            },
            {
              label: 'PM10',
              data: [data.PM10?.concentration, 30, 35, 28, 45, 50, 60],
              borderColor: 'rgb(255, 99, 132)',
              fill: false,
            },
          ],
        });
      } else {
        setError('No air quality data available');
      }
    } catch (error) {
      setError('Failed to fetch air quality data');
      console.error('API-Ninjas error:', error);
    }
    setLoading(false);
  };
  // Function to fetch city name using reverse geocoding API
  const fetchCityName = async (latitude, longitude) => {
    const API_KEY = process.env.REACT_APP_OPEN_WEATHER_API_KEY;
    const url = `https://api.openweathermap.org/data/2.5/weather?lat=${latitude}&lon=${longitude}&appid=${API_KEY}`;
    const response = await fetch(url);
    const data = await response.json();
    const city = data.name;
    setCityName(city);
  };

  // Add this inside your AirQualityAnalysis component
  const calculateAQI = (pm25) => {
    // Indian PM2.5 AQI breakpoints (µg/m³)
    const breakpoints = [
      [0, 30, 0, 50],      // Good
      [31, 60, 51, 100],    // Satisfactory
      [61, 90, 101, 200],   // Moderate
      [91, 120, 201, 300],  // Poor
      [121, 250, 301, 400], // Very Poor
      [251, 500, 401, 500]  // Severe
    ];

    for (const [bpLow, bpHigh, aqiLow, aqiHigh] of breakpoints) {
      if (pm25 >= bpLow && pm25 <= bpHigh) {
        return Math.round(
          ((aqiHigh - aqiLow) / (bpHigh - bpLow)) * (pm25 - bpLow) + aqiLow
        );
      }
    }
    return null;
  };

  // Inside your AirQualityAnalysis component, before the return statement
  const renderPredictionDetails = () => {
    if (!prediction || !prediction.prediction) return null;

    const pm25 = prediction.prediction;
    const aqi = calculateAQI(pm25);

    // Calculate margin and ranges
    let margin;
    if (pm25 < 12) margin = 0.10;
    else if (pm25 < 35) margin = 0.20;
    else margin = 0.30;

    const lowerBound = pm25 * (1 - margin);
    const upperBound = pm25 * (1 + margin);
    const aqiLower = calculateAQI(lowerBound);
    const aqiUpper = calculateAQI(upperBound);

    // Determine status
    const status = pm25 < 12 ? 'Good' :
      pm25 < 35 ? 'Moderate' :
        pm25 < 55 ? 'Unhealthy for Sensitive Groups' :
          pm25 < 150 ? 'Unhealthy' : 'Hazardous';

    // Status tip
    const tip = pm25 < 12 ? 'Great day for outdoor activities!' :
      pm25 < 35 ? 'Generally acceptable conditions' :
        pm25 < 55 ? 'Limit prolonged outdoor exertion' :
          pm25 < 150 ? 'Avoid outdoor activities' :
            'Stay indoors with air purifiers if possible';

    return (
      <div className="aqi-prediction-card">
        <div className="aqi-prediction-value">
          <span className="aqi-prediction-number">PM2.5({pm25.toFixed(1)})</span>
          <span className="aqi-prediction-unit">μg/m³</span>
        </div>

        <div className="aqi-prediction-range">
          <p> <strong>Expected PM2.5 Range:</strong></p>
          <p className="aqi-range-values">
            {lowerBound.toFixed(1)} μg/m³ to {upperBound.toFixed(1)} μg/m³
          </p>
          <p><strong>Possible AQI Range: </strong></p>
          
          <p className="aqi-range-values">
            {aqiLower} to {aqiUpper}
          </p>
          <p>(Note: This AQI is calculated on the basis of the PM2.5 predictions, other factors may also contribute in AQI's accuracy):</p>
        </div>
        <div className="aqi-prediction-details">
          <p >Status: <strong>{status}</strong></p>
          <p className="aqi-prediction-tip">{tip}</p>
        </div>

    
      </div>
    );
  };
  // Trigger location fetch when component loads
  useEffect(() => {
    getLocation();
  }, []);

  return (
    <div className="dashboard-container">
      <Sidebar />
      <div className="main-content">
        <div className="header">
          <img src={`${process.env.PUBLIC_URL}/ClimaLung-logo.png`} alt="ClimaLung Logo" className="logo" />
          <h1>Air Quality Analysis</h1>
        </div>

        {!location ? (
          <div>
            <p>Please enable location access to view air quality data.</p>
            <button onClick={getLocation}>Allow Location</button>
          </div>
        ) : (
          <>
            <div className="cards">
              <div className="card">
                <h2>Current Air Quality in {cityName || 'Loading...'}</h2>
                {loading ? (
                  <p>Loading...</p>
                ) : (
                  <>
                    <p>Air Quality Index: {airQualityData?.aqi || 'N/A'}</p>
                    <p>Status: {
                      airQualityData?.aqi ?
                        (airQualityData.aqi <= 50 ? 'Good' :
                          airQualityData.aqi <= 100 ? 'Moderate' :
                            airQualityData.aqi <= 150 ? 'Unhealthy for Sensitive Groups' :
                              airQualityData.aqi <= 200 ? 'Unhealthy' :
                                airQualityData.aqi <= 300 ? 'Very Unhealthy' :
                                  'Hazardous')
                        : 'N/A'
                    }</p>

                    <button className="card-btn" onClick={() => fetchAirQualityData(location.latitude, location.longitude)}>
                      Refresh Data
                    </button>
                  </>
                )}
                {error && <p className="error">{error}</p>}
              </div>
            </div>

            <div className="concentration-container">
              <h3>Pollutant Concentrations</h3>
              <table className="concentration-table">
                <thead>
                  <tr>
                    <th>Pollutant</th>
                    <th>Concentration (µg/m³)</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>PM2.5</td>
                    <td>{airQualityData?.components?.pm2_5 || 'N/A'}</td>
                  </tr>
                  <tr>
                    <td>PM10</td>
                    <td>{airQualityData?.components?.pm10 || 'N/A'}</td>
                  </tr>
                  <tr>
                    <td>NO2</td>
                    <td>{airQualityData?.components?.no2 || 'N/A'}</td>
                  </tr>
                  <tr>
                    <td>SO2</td>
                    <td>{airQualityData?.components?.so2 || 'N/A'}</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div className="charts">
              <div className="chart-container">
                <h3>Air Quality Trends (Last 7 Days)</h3>
                {airQualityData ? (
                  <Line
                    data={{
                      labels: airQualityData.labels,
                      datasets: airQualityData.datasets,
                    }}
                  />
                ) : (
                  <p>No data available for the chart</p>
                )}
              </div>
            </div>

            <div className="aqi-prediction-container">
              <h3>Tomorrow's AQI Prediction</h3>

              {showPredictionButton ? (
                <button
                  className="aqi-get-prediction-btn"
                  onClick={handleGetPrediction}
                  disabled={!location}
                >
                  Get Prediction
                </button>
              ) : predictionLoading ? (
                <div>
                  <p>Downloading models and calculating prediction...</p>
                  <progress value={progress} max="100" />
                </div>
              ) : predictionError ? (
                <div className="prediction-error-alert">
                  <p>Error: {predictionError}</p>
                  <button onClick={handleGetPrediction}>
                    Try Again
                  </button>
                </div>
              ) : (
                renderPredictionDetails()
              )}
            </div>

            <div className="map-container">
              {location && (
                <MapContainer
                  center={[location.latitude, location.longitude]}
                  zoom={12}
                  style={{ height: '400px', width: '100%' }}
                >
                  <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
                  <Marker position={[location.latitude, location.longitude]}>
                    <Popup>
                      User's Location <br />
                      Latitude: {location.latitude} <br />
                      Longitude: {location.longitude}
                    </Popup>
                  </Marker>
                </MapContainer>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default AirQualityAnalysis;
