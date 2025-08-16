import React, { useState, useRef, useEffect } from "react";

export default function ChatInput({
  onSend,
  disabled,
  loading
}) {
  const [value, setValue] = useState("");
  const inputRef = useRef();

  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = "32px"; // CSS와 맞춤 (min-height)
      const scrollHeight = inputRef.current.scrollHeight;
      const maxHeight = 116; // CSS max-height와 맞춤
      
      if (scrollHeight <= maxHeight) {
        inputRef.current.style.height = scrollHeight + "px";
      } else {
        inputRef.current.style.height = maxHeight + "px";
      }
    }
  }, [value]);

  useEffect(() => {
    if (!disabled && inputRef.current) {
      inputRef.current.focus();
    }
  }, [disabled]);

  const handleSend = () => {
    if (value.trim()) {
      onSend(value.trim());
      setValue("");
      inputRef.current && inputRef.current.focus();
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="chat-input">
      <textarea
        ref={inputRef}
        rows={1}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="메시지를 입력하세요…"
        disabled={disabled}
        style={{
          minHeight: "32px",
          maxHeight: "116px",
          resize: "none"
        }}
      />
      <button onClick={handleSend} disabled={disabled || !value.trim()}>
        전송
      </button>
    </div>
  );
}