import React from 'react'
import './ChatBar.css'

function ChatBar() {
  return (
    <div className="chatBar">
    <input
      type="text"
      placeholder="🧠 What's in your mind?..."
      className="chat-input"
    />
    <button className="send-btn">➤</button>
  </div>
  )
}

export default ChatBar