import './App.css';
import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import Dashboard from './Pages/Dashboard/Dashboard';
import ChatPage from './Pages/ChatPage/ChatPage';
import Signup from './Pages/Authentication/Singup';
import Login from './Pages/Authentication/Login';
import AirQualityAnalysis from './Pages/AirQualityAnalysis/AirQualityAnalysis';
import { useContext } from 'react';
import AuthContext from './AuthContext';
import PatientDataAnalysis from './Pages/PatientDataAnalysis/PatientDataAnalysis';
import About from './Pages/About/About';
import HowToUse from './Pages/HowToUse/HowToUse';
import CTScanUpload from './Pages/CtScanUpload/CtScanUpload';

function App() {
  const { user, loading } = useContext(AuthContext);

  if (loading) {
    return <div>Loading...</div>;
  }

  return (
    <Router>
      <Routes>
        {/* Public Routes */}
        <Route path="/signup" element={<Signup />} />
        <Route path="/login" element={<Login />} />

        {/* Protected Routes */}
        <Route path="/" element={user ? <Dashboard /> : <Navigate to="/login" />} />
        <Route path="/chatpage" element={user ? <ChatPage /> : <Navigate to="/login" />} />
        <Route path="/air-quality-analysis" element={user ? <AirQualityAnalysis /> : <Navigate to="/login" />} />
        <Route path="/ct-scan-analysis" element={user ? <CTScanUpload/> : <Navigate to="/login" />} />
        <Route path="/patient-data-analysis" element={user ? <PatientDataAnalysis /> : <Navigate to="/login" />} />
        <Route path="/about" element={user ? <About /> : <Navigate to="/login" />} />
        <Route path="/how-to-use" element={user ? <HowToUse/> : <Navigate to="/login" />} />
      </Routes>
    </Router>
  );
}

export default App;
