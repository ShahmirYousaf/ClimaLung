import './App.css';
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Dashboard from './Pages/Dashboard/Dashboard';
import ChatPage from './Pages/ChatPage/ChatPage';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/ChatPage" element={<ChatPage />} />
      </Routes>
    </Router>
  );
}

export default App;
