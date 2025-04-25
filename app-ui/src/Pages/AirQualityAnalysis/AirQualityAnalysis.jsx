import React, { useState, useEffect } from 'react';
import './AirQualityAnalysis.css';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS } from 'chart.js/auto';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import L from 'leaflet';
import Sidebar from '../../Components/Sidebar/Sidebar';

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

  // This function is called to get the user's location
  const getLocation = () => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const { latitude, longitude } = position.coords;
          console.log("Location updated:", { latitude, longitude });
          setLocation({ latitude, longitude });
          fetchAirQualityData(latitude, longitude); // Fetch air quality data once we have coordinates
          fetchCityName(latitude, longitude); // Fetch city name
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
      const API_KEY = process.env.REACT_APP_OPEN_WEATHER_API_KEY;
      const url = `https://api.openweathermap.org/data/2.5/air_pollution?lat=${latitude}&lon=${longitude}&appid=${API_KEY}`;
      
      const response = await fetch(url);
      const data = await response.json();
  
      // Log the data to check the structure
      console.log("Air Quality Data:", data);
  
      const aqi = data.list[0]?.main?.aqi;
      const components = data.list[0]?.components;
  
      if (components) {
        setAirQualityData({
          aqi,
          components,
          labels: ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'], // Example labels (can be dynamic based on actual data)
          datasets: [
            {
              label: 'PM2.5',
              data: [components.pm2_5, 40, 45, 38, 50, 55, 60], // Replace with actual data over 7 days
              borderColor: 'rgb(36, 130, 166)',
              fill: false,
            },
            {
              label: 'PM10',
              data: [components.pm10, 30, 35, 28, 45, 50, 60], // Example data
              borderColor: 'rgb(255, 99, 132)',
              fill: false,
            },
            // Add more datasets for other components like NO2, SO2, etc.
          ],
        });
      } else {
        setError('No air quality data available');
      }
    } catch (error) {
      setError('Failed to fetch air quality data');
    }
    setLoading(false);
  };

  // Function to fetch city name using reverse geocoding API
  const fetchCityName = async (latitude, longitude) => {
    const API_KEY = process.env.REACT_APP_OPEN_WEATHER_API_KEY;
    const url = `https://api.openweathermap.org/data/2.5/weather?lat=${latitude}&lon=${longitude}&appid=${API_KEY}`;
    const response = await fetch(url);
    const data = await response.json();
    const city = data.name; // Extracting city name from the response
    setCityName(city);
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
                <h2>Current Air Quality in {cityName || 'Loading...'}</h2>
                {loading ? (
                  <p>Loading...</p>
                ) : (
                  <>
                    <p>Air Quality Index: {airQualityData?.aqi || 'N/A'}</p>
                    <p>Status: {airQualityData?.aqi ? ['Good', 'Fair', 'Moderate', 'Poor', 'Very Poor'][airQualityData.aqi - 1] : 'N/A'}</p>
                    <button className="card-btn" onClick={() => fetchAirQualityData(location.latitude, location.longitude)}>
                      Refresh Data
                    </button>
                  </>
                )}
                {error && <p className="error">{error}</p>}
              </div>
            </div>

            {/* Air Quality Concentrations Table */}
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

            {/* Air Quality Graphs */}
            <div className="charts">
              <div className="chart-container">
                <h3>Air Quality Trends (Last 7 Days)</h3>

                {/* Conditionally render the chart only when air quality data is available */}
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

            {/* Leaflet Map to show the location */}
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
