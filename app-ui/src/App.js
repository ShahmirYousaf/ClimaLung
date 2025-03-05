import './App.css';
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Dashboard from './Pages/Dashboard/Dashboard';
import ChatPage from './Pages/ChatPage/ChatPage';
import Signup from './Pages/Authentication/Singup';
import Login from './Pages/Authentication/Login';

function App() {
  return (
    // <Router>
    //   <Routes>
    //     <Route path="/" element={<Dashboard />} />
    //     <Route path="/ChatPage" element={<ChatPage />} />
    //   </Routes>
    // </Router>
   
    <Router>
      <Routes>
        <Route path="/signup" element={<Signup />} />
        <Route path="/login" element={<Login />} />
      </Routes>
    </Router>
  );
}

export default App;
