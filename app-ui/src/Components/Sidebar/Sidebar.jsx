import React from 'react'
import './Sidebar.css'
import { useNavigate } from 'react-router-dom'
import doctor from '../../Assets/Doctor.png'
import { Link } from "react-router";

function Sidebar() {
  const navigate = useNavigate();

  const handleNewChat = () => {
    navigate("/ChatPage");
  }

  return (
    <aside className="sidebar">
    <div className="sidebar-header">
        <Link to = "/" className='sidebar-heading'>ClimaLung</Link>
        <button className="new-chat-btn" onClick={handleNewChat}>
          + New Chat
          </button>
        <div className="search-box">
          <input
            type="text"
            placeholder="Search..."
            className="search-input"
          />
          <span className="search-icon">🔍</span>
        </div>
        
      </div>
      <nav className="sidebar-nav">
        <h2>Analysis Tools</h2>
        <ul>
          <li>🩻 CT Scan Analysis</li>
          <li>🌬️ Air Quality Analysis</li>
        </ul>
        <h2>User Resources</h2>
        <ul>
          <li>⬆️ Upload CT-Scans</li>
          <li>❓ How to use ClimaLung?</li>
        </ul>
        <h2>Accounts</h2>
        <ul>
          <li>🔓 Log Out</li>
        </ul>
      </nav>
      <div className="sidebar-footer">
        <div className="settings">⚙️ Settings</div>
        <div className="profile">
          <img
            src= {doctor}
            alt="Profile"
          />
          <span>Dr Shahmir Yousaf</span>
        </div>
      </div>
    </aside>
  )
}

export default Sidebar