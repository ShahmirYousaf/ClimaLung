import React, { useState } from 'react';
import axios from 'axios';
import Papa from 'papaparse';
import './PatientDataAnalysis.css'
import Sidebar from '../../Components/Sidebar/Sidebar';

const PatientDataAnalysis = () => {
  const [age, setAge] = useState('');
  const [exposure, setExposure] = useState(0);
  const [symptoms, setSymptoms] = useState(0);
  const [shortnessOfBreath, setShortnessOfBreath] = useState(0);
  const [coughedBlood, setCoughedBlood] = useState(0);
  const [pollutedArea, setPollutedArea] = useState(0);
  const [diagnosedWithCancer, setDiagnosedWithCancer] = useState(0);
  const [gender, setGender] = useState(0)
  const [hasSmoked, setHasSmoked] = useState(0)
  const [PersistentCough, setPersistentCough] = useState(0);
  const [airQualityData, setAirQualityData] = useState(null);
  const [prediction, setPrediction] = useState(null);

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      Papa.parse(file, {
        complete: (result) => {
          

          const lastRow = result.data[result.data.length - 2]; // Get the last row
          
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
      Age: Number(age),
      Gender: gender,
      Exposure_to_Occupational_Hazards: exposure,
      Diagnosed_with_Cancer: diagnosedWithCancer,
      History_Of_Chronic_Respiratory_Diseases: symptoms,
      Lived_In_Highly_Polluted_Area: pollutedArea,
      Shortness_Of_breath: shortnessOfBreath,
      Coughed_Blood: coughedBlood,
      Persistent_Cough: PersistentCough,
      Ever_Smoked: hasSmoked,
      'PM2.5 (ug/m^3)': parseFloat(airQualityData.PM2_5),
      'PM10 (ug/m^3)': parseFloat(airQualityData.PM10),
      AQI: parseFloat(airQualityData.AQI),
      'NO2 (ppb)': parseFloat(airQualityData.NO2),
      'SO2 (ppb)': parseFloat(airQualityData.SO2),
      'CO (ppb)': parseFloat(airQualityData.CO),
    };    

    try {
      //// FOR VERCEL
      //const response = await axios.post('https://clima-lung-bot-api.vercel.app/predict', patientData);

      console.log(patientData);
      // FOR LOCAL
      const response = await axios.post('http://127.0.0.1:5000/predict', patientData);

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
        <select value={gender} onChange={(e) => setGender(Number(e.target.value))}>
          <option value={1}>Male</option>
          <option value={0}>Female</option>
        </select>
      </div>
      

      <div className="input-group">
        <label>Exposure to Occupational Hazards</label>
        <select value={exposure} onChange={(e) => setExposure(Number(e.target.value))}>
          <option value={2}>Yes</option>
          <option value={1}>No</option>
          <option value={0}>Maybe</option>
        </select>
      </div>

      <div className="input-group">
        <label>History of Chronic Respiratory Diseases</label>
        <select value={symptoms} onChange={(e) => setSymptoms(Number(e.target.value))}>
          <option value={2}>Yes</option>
          <option value={1}>No</option>
          <option value={0}>Maybe</option>
        </select>
      </div>

      <div className="input-group">
        <label>Diagnosed with Cancer</label>
        <select value={gender} onChange={(e) => setDiagnosedWithCancer(Number(e.target.value))}>
          <option value={1}>Yes</option>
          <option value={0}>No</option>
        </select>
      </div>

      <div className="input-group">
        <label>Lived in a Highly Polluted Area</label>
        <select value={pollutedArea} onChange={(e) => setPollutedArea(Number(e.target.value))}>
          <option value={2}>Yes</option>
          <option value={1}>No</option>
          <option value={0}>Maybe</option>
        </select>
      </div>

      <div className="input-group">
        <label>Shortness of Breath</label>
        <select value={shortnessOfBreath} onChange={(e) => setShortnessOfBreath(Number(e.target.value))}>
          <option value={2}>Yes</option>
          <option value={1}>No</option>
          <option value={0}>Maybe</option>
        </select>
      </div>

      <div className="input-group">
        <label>Persistent Cough</label>
        <select value={PersistentCough} onChange={(e) => setPersistentCough(Number(e.target.value))}>
          <option value={2}>Yes</option>
          <option value={1}>No</option>
          <option value={0}>Maybe</option>
        </select>
      </div>

      <div className="input-group">
        <label>Coughed Blood</label>
        <select value={coughedBlood} onChange={(e) => setCoughedBlood(Number(e.target.value))}>
          <option value={2}>Yes</option>
          <option value={1}>No</option>
          <option value={0}>Maybe</option>
        </select>
      </div>

      <div className="input-group">
        <label>Ever Smoked</label>
        <select value={hasSmoked} onChange={(e) => setHasSmoked(Number(e.target.value))}>
          <option value={1}>Yes</option>
          <option value={0}>No</option>
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
