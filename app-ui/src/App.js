import './App.css';
import Sidebar from "./Components/Sidebar/Sidebar";
import ChatBar from './Components/ChatBar/ChatBar';
import Dashboard from './Components/Dashboard/Dashboard';

function App() {
  return (
    <div className="app-container">
    <Sidebar />
    <Dashboard/>
    <ChatBar/>
  </div>
  );
}

export default App;
