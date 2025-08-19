import React, { useRef, useEffect } from "react";
import ChatBubble from "./ChatBubble";

export default function ChatMessages({
  messages,
  loading,
  showCategorySelector,
  selectedCategory,
  inputNotice,
  onCategorySelect
}) {
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, loading, showCategorySelector, selectedCategory, inputNotice]);

  return (
    <div className="chat-messages" ref={scrollRef}>
      {messages.map((msg, i) => (
        <ChatBubble key={i} text={msg.text} isUser={msg.isUser} />
      ))}
      {loading && <ChatBubble text="..." isUser={false} loading />}
      {showCategorySelector && (
        <>
          {inputNotice && (
            <ChatBubble text={inputNotice} isUser={false} />
          )}
          <div className="category-btn-group">
            <button className="category-btn-strong" onClick={() => onCategorySelect("창업")}>창업</button>
            <button className="category-btn-strong" onClick={() => onCategorySelect("정책")}>정책</button>
            <button className="category-btn-strong" onClick={() => onCategorySelect("트렌드")}>트렌드</button>
          </div>
        </>
      )}
    </div>
  );
}