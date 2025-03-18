import React, {useState, useEffect} from 'react'
import Sidebar from '../../Components/Sidebar/Sidebar'
import ChatBar from '../../Components/ChatBar/ChatBar'
import './ChatPage.css'
import axios from "axios";
import { useLocation } from "react-router-dom";
import { v4 as uuidv4 } from "uuid"; // Import UUID for unique session ID


function ChatPage() {
  const sessionId = uuidv4();
  const location = useLocation();
  const [messages, setMessages] = useState([]);
  const [receivedInput, setReceivedInput] = useState("");

  useEffect(() => {
    if (location.state?.message) {
      setReceivedInput(location.state.message);
    } else {
      setReceivedInput("No message sent.");
    }
  }, [location.state]);

  useEffect(() => {
    if (receivedInput && receivedInput !== "No message sent.") {
      sendMessage();
    }
  }, [receivedInput]); 

  const sendMessage = async () => {
    const userMessage = { sender: "user", text: receivedInput };
    setMessages([...messages, userMessage]);
    const backendURL = process.env.REACT_APP_BACKEND_URL || "https://clima-lung-backend.vercel.app";

    try {
        // Fetch the access token from the backend
        const tokenResponse = await axios.get(`${backendURL}/get-token`);
        const accessToken = tokenResponse.data.token;

        if (!accessToken) {
            throw new Error("Failed to retrieve access token");
        }

        // Send message to Dialogflow
        const response = await axios.post(
            `https://dialogflow.googleapis.com/v2/projects/${process.env.REACT_APP_PROJECT_ID}/agent/sessions/${sessionId}:detectIntent`,
            {
                queryInput: {
                    text: { text: receivedInput, languageCode: "en" },
                },
            },
            {
                headers: {
                    Authorization: `Bearer ${accessToken}`, // Use the token from backend
                    "Content-Type": "application/json",
                },
            }
        );

        // Handle bot's response
        const botMessage = { sender: "bot", text: response.data.queryResult.fulfillmentText };
        setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
        console.error("Error sending message:", error);
    }
};


  return (
    <div className='ChatPage-container'>
        <Sidebar/>
        <div className="chat-messages">
        <div className="chat-box">
          {messages.map((msg, index) => (
            <div key={index} className={msg.sender === "user" ? "user-message" : "bot-message"}>
              {msg.text}
            </div>
          ))}
        </div>
      </div>
        <ChatBar/>
    </div>
  )
}

export default ChatPage