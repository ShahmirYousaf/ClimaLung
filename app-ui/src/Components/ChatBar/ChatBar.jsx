import React, {useState} from 'react'
import './ChatBar.css'
import { useNavigate } from 'react-router-dom';


function ChatBar() {
  const navigate = useNavigate()
  const [input, setInput] = useState("");

  const handleSendChat = () => {
    if (input.trim() !== "") { 
      navigate("/ChatPage", { state: { message: input } }); 
      setInput(""); 
    }
  }

  return (
    <div className="chatBar">
    <input
      type="text"
      value={input}
      onChange={(e) => setInput(e.target.value)}
      placeholder="🧠 What's in your mind?..."
      className="chat-input"
    />
    <button onClick ={handleSendChat} className="send-btn">➤</button>
  </div>
  )
}

export default ChatBar