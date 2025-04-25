import React, {useState, useEffect} from 'react'
import './Sidebar.css'
import { useNavigate } from 'react-router-dom'
import doctor from '../../Assets/Doctor.png'
import { Link } from "react-router";
import { signOut } from 'firebase/auth';
import { auth } from '../../Firebase';
import Swal from "sweetalert2";

function Sidebar() {
  const navigate = useNavigate();
  const [isSidebarOpen, setIsSidebarOpen] = useState(false); 
  const [userEmail, setUserEmail] = useState(null);

  useEffect(() => {
    const user = auth.currentUser; // Get the logged-in user from Firebase
    if (user) {
      setUserEmail(user.email); 
    }
  }, []);

  const handleNewChat = () => {
    navigate("/ChatPage");
  }

  const handleLogout = () => {
     Swal.fire(
      {title: 'Are you sure?',
      text: "You won't be able to revert this action!",
      icon: 'warning',
      showCancelButton: true,
      confirmButtonColor: '#2482A6',
      cancelButtonColor: '#ff0000',
      confirmButtonText: 'Yes, log out!'
    }).then(() => {
      signOut(auth)
      .then(() => {
        Swal.fire(
          'Logged Out!',
          'You have been logged out.',
          'success'
        );
        navigate('/login');
      })
      .catch((error) => {
        console.error('Error logging out:', error);
      });
      }
  );
}

  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen); // Toggle sidebar open state
  };

  return (
    <>
     {/* Toggler Button */}
     <button className="sidebar-toggler" onClick={toggleSidebar}>
        ☰
      </button>
    <aside className={`sidebar ${isSidebarOpen ? 'open' : ''}`}>
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
          <li><Link to="/air-quality-analysis" className='sidebar-nav-list'>🌬️ Air Quality Analysis</Link>
          </li>
        </ul>
        <h2>User Resources</h2>
        <ul>
          <li>⬆️ Upload CT-Scans</li>
          <li>❓ How to use ClimaLung?</li>
        </ul>
        <h2>Accounts</h2>
        <ul>
          <li onClick={handleLogout}>🔓 Log Out</li>
        </ul>
      </nav>
      <div className="sidebar-footer">
        <div className="settings">⚙️ Settings</div>
        <div className="profile">
          <img
            src= {doctor}
            alt="Profile"
          />
          <span >{userEmail ? userEmail : 'Loading...'}</span>
        </div>
      </div>
    </aside>
    </>
  )
}

export default Sidebar