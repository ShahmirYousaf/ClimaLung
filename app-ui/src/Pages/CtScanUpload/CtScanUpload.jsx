import React, { useState } from 'react';
import './CtScanUpload.css';
import Sidebar from '../../Components/Sidebar/Sidebar';
import axios from 'axios';

function CTScanUpload() {
  const [ctImage, setCtImage] = useState(null);
  const [medicalInsights, setMedicalInsights] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [prediction, setPrediction] = useState(null);

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setLoading(true);
    setError(null);
    setMedicalInsights(null);
    setCtImage(null);
    setPrediction(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await axios.post('http://localhost:5000/analyze', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const { prediction, scan_image, medical_insights } = response.data;
      setCtImage(`data:image/png;base64,${scan_image}`);
      setMedicalInsights(medical_insights);

      if (prediction === 1) {
        setPrediction('Cancerous');
      } else {
        setPrediction('Non-Cancerous');
      }
    } catch (err) {
      console.error('Error:', err);
      setError("Further verification needed to confirm results.Please consult a medical professional");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="ctscan-upload-container-main">
      <Sidebar />
      <main className="main-content-ctscan">
        <div className="header-ctscan">
          <img
            src={`${process.env.PUBLIC_URL}/ClimaLung-logo.png`}
            alt="ClimaLung Logo"
            className="logo-ctscan"
          />
          <h1>Lung Cancer Detection</h1>
        </div>

        {/* CT Scan Upload Card */}
        <div className="cards-ctscan">
          <div className="card-ctscan">
            <p className="card-ctscan-para">Upload your CT scan and get the results of whether the scan is cancerous or not.</p>

            <label className="upload-label">Choose a CT scan image:</label>
            <input
              type="file"
              accept=".npy,.jpg,.jpeg,.png,.jfif"
              onChange={handleFileUpload}
              className="upload-input"
              disabled={loading}
            />

            {loading && <p className="loading-message">Analyzing your CT scan...</p>}
            {error && <p className="error-message">{error}</p>}
          </div>
        </div>

        {/* Results Section */}
        {ctImage && (
          <div className="card-ctscan result-card">
            <div className="result-content">
              {/* Image Display */}
              <div className="image-container">
                <h3>Original CT Scan</h3>
                <img src={ctImage} alt="CT Scan" className="image-display" />
              </div>

              {/* Medical Analysis */}
              <div className="medical-analysis">
                <h3>Medical Analysis of the highlighted nodule:</h3>
                <div className="analysis-grid">
                  {medicalInsights &&
                    medicalInsights.slice(0, 4).map((section, index) => (
                      <div key={index} className="insight-section">
                        <h4>{section.title}</h4>
                        <ul>
                          {section.findings.map((finding, i) => (
                            <li key={i}>{finding}</li>
                          ))}
                        </ul>
                      </div>
                    ))}
                </div>
                     {/* Prediction Result */}
            {prediction && (
              <div className="result-section">
                <h3>Prediction Result:</h3>
                <p>The lung CT scan is classified as <strong>{prediction}</strong>.</p>
              </div>
            )}
              </div>
            </div>

           
          </div>
        )}
      </main>
    </div>
  );
}

export default CTScanUpload;
