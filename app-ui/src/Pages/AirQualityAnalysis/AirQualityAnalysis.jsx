import React, { useState, useEffect } from 'react';
import './AirQualityAnalysis.css';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS } from 'chart.js/auto';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import { useMap } from 'react-leaflet';
import Sidebar from '../../Components/Sidebar/Sidebar';
import L from 'leaflet';

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

  // This function is called to get the user's location
  const getLocation = () => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const { latitude, longitude } = position.coords;
          console.log("Location updated:", { latitude, longitude }); // Debugging location
          setLocation({ latitude, longitude });
          fetchAirQualityData(latitude, longitude); // Fetch air quality data once we have coordinates
        },
        (err) => setError('Unable to retrieve your location')
      );
    } else {
      setError('Geolocation is not supported by this browser.');
    }
  };

  // Function to fetch air quality data from an API using the user's coordinates
  const fetchAirQualityData = async (latitude, longitude) => {
    setLoading(true);
    try {
      console.log("env ariables:", process.env);
      const API_KEY = process.env.REACT_APP_OPEN_WEATHER_API_KEY; // Access the API key from .env
      console.log("API Key:", process.env.REACT_APP_OPEN_WEATHER_API_KEY);
      const url = `https://api.openweathermap.org/data/2.5/air_pollution?lat=${latitude}&lon=${longitude}&appid=${API_KEY}`;
      const response = await fetch(url);
      const data = await response.json();

      // Example of extracting data you might use
      const aqi = data.list[0].main.aqi;
      const airQualityLevels = ['Good', 'Fair', 'Moderate', 'Poor', 'Very Poor'];

      // Initialize the air quality data for the chart
      setAirQualityData({
        labels: ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'], // Example labels
        datasets: [
          {
            label: 'Air Quality Index',
            data: [40, 55, 60, 50, 48, 45, 38], // Example data
            borderColor: 'rgb(36, 130, 166)',
            fill: false,
          },
        ],
        aqi,
        level: airQualityLevels[aqi - 1],
      });
    } catch (error) {
      setError('Failed to fetch air quality data');
    }
    setLoading(false);
  };

  // Trigger location fetch when component loads
  useEffect(() => {
    getLocation();
  }, []);

  // Debugging location object
  console.log("Location:", location);

  return (
    <div className="dashboard-container">
        <Sidebar />

      <div className="main-content">
        <div className="header">
          <img src={`${process.env.PUBLIC_URL}/ClimaLung-logo.png`} alt="ClimaLung Logo" className="logo" />
          <h1>Air Quality Analysis</h1>
        </div>

        {/* Display location and air quality data */}
        {!location ? (
          <div>
            <p>Please enable location access to view air quality data.</p>
            <button onClick={getLocation}>Allow Location</button>
          </div>
        ) : (
          <>
            <div className="cards">
              {/* Air Quality Card */}
              <div className="card">
                <h2>Current Air Quality</h2>
                {loading ? (
                  <p>Loading...</p>
                ) : (
                  <>
                    <p>Air Quality Index: {airQualityData?.aqi || 'N/A'}</p>
                    <p>Status: {airQualityData?.level || 'N/A'}</p>
                    <button className="card-btn" onClick={() => fetchAirQualityData(location.latitude, location.longitude)}>
                      Refresh Data
                    </button>
                  </>
                )}
                {error && <p className="error">{error}</p>}
              </div>
            </div>

            {/* Air Quality Graphs */}
            <div className="charts">
              <div className="chart-container">
                <h3>Air Quality Trends (Last 7 Days)</h3>

                {/* Conditionally render the chart only when air quality data is available */}
                {airQualityData ? (
                  <Line data={airQualityData} />
                ) : (
                  <p>No data available for the chart</p>
                )}
              </div>
            </div>

            {/* Leaflet Map to show the location */}
            <div className="map-container">
              {/* Ensure that the coordinates are valid before rendering the map */}
              {location && (
                <MapContainer center={[location.latitude, location.longitude]} zoom={20} style={{ height: '400px', width: '100%' }}>
                  <TileLayer
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                  />
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
