import React from 'react'
import './Dashboard.css'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faUpload } from '@fortawesome/free-solid-svg-icons';
import Sidebar from '../../Components/Sidebar/Sidebar';
import ChatBar from '../../Components/ChatBar/ChatBar';

function Dashboard() {
  return (
    <div className="dashboard-container-main">
      <Sidebar/>
    <main className="main-content-dashboard">
    <div className="header-dashboard">
      <img src={`${process.env.PUBLIC_URL}/ClimaLung-logo.png`} alt="ClimaLung Logo" className="logo-dashboard" />
      <h1>Good day! How may I assist you today?</h1>
    </div>
    <div className="cards-dashboard">
      <div className="card-dashboard">
        <h2>Upload CT Scans</h2>
        <p>Detailed insights from your lung CT scans.</p>
        <button className="card-btn">
            <FontAwesomeIcon icon={faUpload} /> Upload
          </button>
      </div>
    </div>
  </main>
  <ChatBar/>
  </div>
  )
}

export default Dashboard