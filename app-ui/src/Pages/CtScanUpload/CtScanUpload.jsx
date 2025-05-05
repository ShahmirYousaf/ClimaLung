import React from 'react';
import './CtScanUpload.css';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faUpload } from '@fortawesome/free-solid-svg-icons';
import Sidebar from '../../Components/Sidebar/Sidebar';

function CTScanUpload() {
  return (
    <div className="ctscan-upload-container-main">
      <Sidebar />
      <main className="main-content-ctscan">
        <div className="header-ctscan">
          <img src={`${process.env.PUBLIC_URL}/ClimaLung-logo.png`} alt="ClimaLung Logo" className="logo-ctscan" />
          <h1>Lung Cancer Detection</h1>
        </div>
        <div className="cards-ctscan">
          <div className="card-ctscan">
            <h2>Upload CT Scan for Analysis</h2>
            <p>Upload your CT scan and get the results of whether the scan is cancerous or not.</p>
            <button className="card-btn-ct">
              <FontAwesomeIcon icon={faUpload} /> Upload
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}

export default CTScanUpload;
