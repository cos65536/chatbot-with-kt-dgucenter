import React from "react";
import ThemeToggle from "./ThemeToggle";

export default function ChatHeader({ 
  showCategoryChangeButton, 
  onCategoryChange, 
  loading 
}) {
  return (
    <div className="chat-header">
      <ThemeToggle />
      <span className="chat-title">동성로 창업자를 위한 AI 챗봇</span>
      {showCategoryChangeButton && (
        <button
          className="category-change-btn-header"
          onClick={onCategoryChange}
          disabled={loading}
          type="button"
          aria-label="카테고리 변경"
        >
          <span>카테고리 변경</span>
        </button>
      )}
    </div>
  );
}