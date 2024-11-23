import React from 'react'
import './Dashboard.css'
import logo from '../../Assets/ClimaLung-logo.png'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faUpload } from '@fortawesome/free-solid-svg-icons';

function Dashboard() {
  return (
    <main className="main-content">
    <div className="header">
      <img src={logo} alt="ClimaLung Logo" className="logo" />
      <h1>Good day! How may I assist you today?</h1>
    </div>
    <div className="cards">
      {/* <div className="card">
        <h2>Consult an Expert</h2>
        <p>Let us know your symptoms in order to identify your disease.</p>
        <button className="card-btn">Go!</button>
      </div> */}
      <div className="card">
        <h2>Upload CT Scans</h2>
        <p>Detailed insights from your lung CT scans.</p>
        <button className="card-btn">
            <FontAwesomeIcon icon={faUpload} /> Upload
          </button>
      </div>
    </div>
  </main>
  )
}

export default Dashboard