import './App.css';
import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import Dashboard from './Pages/Dashboard/Dashboard';
import ChatPage from './Pages/ChatPage/ChatPage';
import Signup from './Pages/Authentication/Singup';
import Login from './Pages/Authentication/Login';
import { useContext } from 'react';
import AuthContext from './AuthContext';

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
      </Routes>
    </Router>
  );
}

export default App;
