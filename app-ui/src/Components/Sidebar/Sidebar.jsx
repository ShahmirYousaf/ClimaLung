import React from 'react'
import './Sidebar.css'

function Sidebar() {
  return (
    <aside className="sidebar">
    <div className="sidebar-header">
        <h1 className='sidebar-heading'>ClimaLung</h1>
        <button className="new-chat-btn">+ New Chat</button>
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
        <h2>Our Products</h2>
        <ul>
          <li>🩻 CT Scan Analysis</li>
          <li>🌬️ Air Quality Analysis</li>
        </ul>
        <h2>Previous Chats</h2>
        <ul>
          <li>⬆️ Upload CT-Scans</li>
          {/* <li>🩺 Get a monthly check-up</li> */}
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
            src="https://via.placeholder.com/150"
            alt="Profile"
          />
          <span>Andrew Neilson</span>
        </div>
      </div>
    </aside>
  )
}

export default Sidebar