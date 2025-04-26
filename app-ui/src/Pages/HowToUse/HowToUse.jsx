import React from "react";
import "./HowToUse.css";
import { Link } from "react-router-dom"
import Sidebar from "../../Components/Sidebar/Sidebar";

const HowToUse = () => {
  return (
    <div className="how-to-use-container">
        <Sidebar/>
      <h1 className="page-title">How to Use ClimaLung</h1>
      <p className="page-description">
        ClimaLung is a powerful platform that leverages AI to analyze the
        impact of air quality and health data to detect lung cancer and assess
        lung health. Below are the main features of the ClimaLung platform and
        how to use them.
      </p>

      <div className="module-section">
        <h2 className="module-title">1. CT Scan Upload</h2>
        <p className="module-description">
          The CT Scan Upload feature allows users to upload their CT scan images
          for AI-powered lung cancer detection. Simply upload your CT scan, and
          the system will analyze it and provide you with a detailed result on
          the likelihood of lung cancer.
        </p>
        <div className="module-button">
          <Link to="/ct-scan-upload">
            <button className="module-btn">Go to CT Scan Upload</button>
          </Link>
        </div>
      </div>

      <div className="module-section">
        <h2 className="module-title">2. Air Quality Analysis</h2>
        <p className="module-description">
          This module provides real-time air quality predictions and analysis.
          By entering your location, ClimaLung will fetch environmental data
          (e.g., PM2.5 levels, temperature, humidity) and predict potential
          risks to lung health. It helps users understand the air quality of
          their environment and the potential long-term effects.
        </p>
        <div className="module-button">
          <Link to="/air-quality-analysis">
            <button className="module-btn">Go to Air Quality Analysis</button>
          </Link>
        </div>
      </div>

      <div className="module-section">
        <h2 className="module-title">3. Patient Data Analysis</h2>
        <p className="module-description">
          The Patient Data Analysis page allows users to input medical
          information through a series of questions. Based on this data, the
          system predicts lung health and offers insights into potential risks
          for lung diseases or cancer.
        </p>
        <div className="module-button">
          <Link to="/patient-data-analysis">
            <button className="module-btn">Go to Patient Data Analysis</button>
          </Link>
        </div>
      </div>

      <div className="module-section">
        <h2 className="module-title">4. ChatBot Assistance</h2>
        <p className="module-description">
          ClimaLung comes with an AI-powered chatbot that can answer your
          questions related to lung health, air quality, and lung cancer. Whether
          you're looking for general information or specific guidance, the
          chatbot is available to provide instant responses.
        </p>
        <div className="module-button">
          <Link to="/chatbot">
            <button className="module-btn">Go to ChatBot</button>
          </Link>
        </div>
      </div>

      <div className="footer">
        <p>ClimaLung - Your AI-powered Lung Health Assistant</p>
      </div>
    </div>
  );
};

export default HowToUse;
