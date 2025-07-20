import { useState } from 'react'
import './App.css'

function App() {

  const [userInput, setUserInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleSend = async () => {}

  
  const clearMemory = async () => {
    const response = await fetch("http://localhost:8000/clear-memory", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });
    if (response.status === 200) {
      console.log('cleared memory!');
    }
  }

  return (
      <div className="container">
        <h1 className="title">LangGraph Agent</h1>
          <div className="main-container">

            <div className="messages">
              <div className="card-content" id="chat-box">
                {messages.map((msg, index) => (
                  <div key={index} className="message">
                    <strong>{msg.role === "user" ? "User:" : "Agent:"}</strong>
                  </div>
                ))}
                </div>
              </div>
                  <div className="input-group">
                      <textarea className="input-textarea" rows="4" cols="50" onChange={(e) => setUserInput(e.target.value)}
                        onKeyDown={(e) => e.key === "Enter" && handleSend()} placeholder="Type your prompt..." value={userInput}
                        />
                      <button className="send-button" onClick={handleSend} disabled={loading}>
                        {loading ? "Sending..." : "Send"}
                      </button>
                      <button className="clear-memory" onClick={clearMemory}>
                        Clear memory
                      </button>
                </div>
                
                </div>
                </div>
  )
}

export default App
