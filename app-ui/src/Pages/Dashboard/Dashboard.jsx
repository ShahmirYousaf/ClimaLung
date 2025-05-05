import React from 'react';
import './Dashboard.css';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faPoll, faFileMedical, faHeartbeat } from '@fortawesome/free-solid-svg-icons'; // Icons for the new boxes
import Sidebar from '../../Components/Sidebar/Sidebar';
import ChatBar from '../../Components/ChatBar/ChatBar';
import { Link } from 'react-router-dom'; // Import for routing

function Dashboard() {
  return (
    <div className="dashboard-container-main">
      <Sidebar />
      <main className="main-content-dashboard">
        <div className="header-dashboard">
          <img src={`${process.env.PUBLIC_URL}/ClimaLung-logo.png`} alt="ClimaLung Logo" className="logo-dashboard" />
          <h1>Good day! How may I assist you today?</h1>
        </div>
        <div className="cards-dashboard">
          {/* Air Quality Analysis Box */}
          <Link to="/air-quality-analysis" className="card-link">
            <div className="card-dashboard">
              <FontAwesomeIcon icon={faPoll} size="3x" className="card-icon" />
              <h2>Air Quality Analysis</h2>
              <p>Analyze air quality data for different regions.</p>
            </div>
          </Link>

          {/* CT Scan Analysis Box */}
          <Link to="/ct-scan-analysis" className="card-link">
            <div className="card-dashboard">
              <FontAwesomeIcon icon={faFileMedical} size="3x" className="card-icon" />
              <h2>CT Scan Analysis</h2>
              <p>Upload and analyze CT scans for lung conditions.</p>
            </div>
          </Link>

          {/* Patient Data Analysis Box */}
          <Link to="/patient-data-analysis" className="card-link">
            <div className="card-dashboard">
              <FontAwesomeIcon icon={faHeartbeat} size="3x" className="card-icon" />
              <h2>Patient Data Analysis</h2>
              <p>Analyze patient data for insights into lung health.</p>
            </div>
          </Link>
        </div>
      </main>
      <ChatBar />
    </div>
  );
}

export default Dashboard;
