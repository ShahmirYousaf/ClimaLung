import React, { useState } from 'react';
import axios from 'axios';
import Papa from 'papaparse';
import './PatientDataAnalysis.css'
import Sidebar from '../../Components/Sidebar/Sidebar';

const PatientDataAnalysis = () => {
  const [age, setAge] = useState('');
  const [exposure, setExposure] = useState('No');
  const [symptoms, setSymptoms] = useState('No');
  const [shortnessOfBreath, setShortnessOfBreath] = useState('No');
  const [coughedBlood, setCoughedBlood] = useState('No');
  const [pollutedArea, setPollutedArea] = useState('No');
  const [gender, setGender] = useState('')
  const [hasSmoked, setHasSmoked] = useState("no")
  const [PersistentCough, setPersistentCough] = useState("no");
  const [airQualityData, setAirQualityData] = useState(null);
  const [prediction, setPrediction] = useState(null);

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      Papa.parse(file, {
        complete: (result) => {
          const lastRow = result.data[result.data.length - 1]; // Get the last row
          // Set the last row of air quality data
          setAirQualityData({
            PM2_5: lastRow['PM2.5 (ug/m^3)'],
            PM10: lastRow['PM10 (ug/m^3)'],
            AQI: lastRow['AQI'],
            NO2: lastRow['NO2 (ppb)'],
            SO2: lastRow['SO2 (ppb)'],
            CO: lastRow['CO (ppb)'],
          });
        },
        header: true, // Assuming the CSV has headers
      });
    }
  };

  const handleSubmit = async () => {
    if (!airQualityData) {
      alert('Please upload the air quality data CSV file.');
      return;
    }

    const patientData = {
      Age: age,
      Gender: gender,
      Exposure_to_Occupational_Hazards: exposure,
      History_Of_Chronic_Respiratory_Diseases: symptoms,
      Lived_In_Highly_Polluted_Area: pollutedArea,
      Shortness_Of_breath: shortnessOfBreath,
      Coughed_Blood: coughedBlood,
      Persistent_Cough: PersistentCough,
      Ever_Smoked: hasSmoked,
      'PM2.5 (ug/m^3)': airQualityData.PM2_5,
      'PM10 (ug/m^3)': airQualityData.PM10,
      AQI: airQualityData.AQI,
      'NO2 (ppb)': airQualityData.NO2,
      'SO2 (ppb)': airQualityData.SO2,
      'CO (ppb)': airQualityData.CO,
    };

    try {
      const response = await axios.post('https://clima-lung-bot-api.vercel.app/predict', patientData);
      setPrediction(response.data.prediction);
    } catch (error) {
      console.error('Error predicting:', error);
    }
  };

  return (
    <div className="predict-container">
        <Sidebar/>
      <h1>Lung Health Prediction</h1>

      <div className="input-group">
        <label>Age</label>
        <input
          type="number"
          placeholder="Enter Age"
          value={age}
          onChange={(e) => setAge(e.target.value)}
        />
      </div>

      <div className="input-group">
        <label>Gender</label>
        <select value={gender} onChange={(e) => setGender(e.target.value)}>
          <option value="Male">Male</option>
          <option value="Female">Female</option>
        </select>
      </div>
      

      <div className="input-group">
        <label>Exposure to Occupational Hazards</label>
        <select value={exposure} onChange={(e) => setExposure(e.target.value)}>
          <option value="Yes">Yes</option>
          <option value="No">No</option>
          <option value="Maybe">Maybe</option>
        </select>
      </div>

      <div className="input-group">
        <label>History of Chronic Respiratory Diseases</label>
        <select value={symptoms} onChange={(e) => setSymptoms(e.target.value)}>
          <option value="Yes">Yes</option>
          <option value="No">No</option>
        </select>
      </div>

      <div className="input-group">
        <label>Lived in a Highly Polluted Area</label>
        <select value={pollutedArea} onChange={(e) => setPollutedArea(e.target.value)}>
          <option value="Yes">Yes</option>
          <option value="No">No</option>
        </select>
      </div>

      <div className="input-group">
        <label>Shortness of Breath</label>
        <select value={shortnessOfBreath} onChange={(e) => setShortnessOfBreath(e.target.value)}>
          <option value="Yes">Yes</option>
          <option value="No">No</option>
        </select>
      </div>

      <div className="input-group">
        <label>Persistent Cough</label>
        <select value={PersistentCough} onChange={(e) => setPersistentCough(e.target.value)}>
          <option value="Yes">Yes</option>
          <option value="No">No</option>
        </select>
      </div>

      <div className="input-group">
        <label>Coughed Blood</label>
        <select value={coughedBlood} onChange={(e) => setCoughedBlood(e.target.value)}>
          <option value="Yes">Yes</option>
          <option value="No">No</option>
        </select>
      </div>

      <div className="input-group">
        <label>Ever Smoked</label>
        <select value={hasSmoked} onChange={(e) => setHasSmoked(e.target.value)}>
          <option value="Yes">Yes</option>
          <option value="No">No</option>
        </select>
      </div>

      <div className="input-group">
        <label>Upload Air Quality Data (CSV)</label>
        <input type="file" accept=".csv" onChange={handleFileUpload} />
      </div>

      <button className="predict-btn" onClick={handleSubmit}>
        Predict
      </button>

      {prediction && (
        <div className="prediction-result">
          <h2>Prediction Result:</h2>
          <p>{prediction}</p>
        </div>
      )}
    </div>
  );
};

export default PatientDataAnalysis ;
