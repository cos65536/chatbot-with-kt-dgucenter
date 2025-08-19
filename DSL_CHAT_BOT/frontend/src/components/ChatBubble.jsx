import React from "react";

export default function ChatBubble({ text, isUser, loading }) {
  return (
    <div
      className={`chat-bubble ${isUser ? "user" : "bot"}${loading ? " loading" : ""}`}
    >
      {text}
    </div>
  );
}