import React from "react";
import { ThemeProvider } from "./context/ThemeContext";
import ChatBot from "./components/ChatBot";

function App() {
  return (
    <ThemeProvider>
      <div style={{ minHeight: "100vh" }}>
        <ChatBot />
      </div>
    </ThemeProvider>
  );
}

export default App;

