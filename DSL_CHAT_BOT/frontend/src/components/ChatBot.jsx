import React, { useEffect, useState } from "react";
import ChatHeader from "./ChatHeader";
import ChatMessages from "./ChatMessages";
import ChatInput from "./ChatInput";
import { useTheme } from "../context/ThemeContext";

const CATEGORY_DATA = {
  창업: {
    description: `📋 창업 상담을 선택하셨습니다!

동성로 창업 전문가가 여러분의 궁금증을 해결해드립니다.
업종별 생존률, 폐업률, 창업률 등 다양한 정보를 제공합니다.

 이런 질문들을 해보세요:
 `,
    examples: [
      `🔍 업종별 창업 문의
• 동성로에서 카페 창업하려는데 어때?
• 치킨집 폐업률은 어느정도야??
• 중식당 가게의 3년 생존률을 어느정도야?

궁금한 점을 자연스럽게 대화하듯 말씀해 주세요! `
    ],
    value: "startup",
  },
  정책: {
    description: `📋 창업 지원 정책을 선택하셨습니다!

최신 창업 지원 정책부터 내가 받을 수 있는 창업 정책까지 
상세하게 안내해드리겠습니다.

 이런 질문들을 해보세요:
 `,
    examples: [
      `✅ 대구 특화 지원
• 대구시 창업 지원 프로그램은?
• 동성로 상권 활성화 지원이 있나요?

⚙️ 대상별 맞춤 지원
• 청년 창업 지원 정책 알려주세요
• 여성 창업자 특별 지원이 있나요?
• 청년(39세 이하) 대상 정책은?
• 시니어 창업 지원도 있나요?

정확한 최신 정보로 도움드리겠습니다! `
    ],
    value: "policy",
  },
  트렌드: {
    description: ` 📋창업 트렌드 분석을 선택하셨습니다!

네이버 데이터랩 검색 트렌드를 분석하여
시장 상황과 창업 전망을 알려드립니다.

이런 질문들을 해보세요:
`,
    examples: [
      `🔍 업종별 트렌드
• 요즘 카페 창업 트렌드는 어떤가요?
• 치킨 업계 시장 상황이 궁금해요
• 디저트 카페 인기는 어떤가요?

📊 시기별 분석
• 지금이 창업하기 좋은 시기인가요?
• 요즘 뜨는 창업 아이템은 뭔가요?
• 연말 매출이 좋은 사업은?

‼️ 지역 특화 트렌드
• 동성로에서 뜨고 있는 업종은?
• 대구 지역 소비 트렌드는?
• 젊은층이 선호하는 업종은?

📈 미래 전망
• 앞으로 유망한 창업 아이템은?
• 포화상태인 업종을 알려주세요
• 계절별 트렌드 변화는?

실시간 데이터로 정확한 트렌드를 분석해드립니다!`
    ],
    value: "trend",
  },
};


export default function ChatBot() {
  const { theme } = useTheme();
  const [messages, setMessages] = useState([]);
  const [selectedCategory, setSelectedCategory] = useState(null);
  const [showCategorySelector, setShowCategorySelector] = useState(true);
  const [loading, setLoading] = useState(false);
  const [inputNotice, setInputNotice] = useState("");
  const [isFirstTime, setIsFirstTime] = useState(true);
  useEffect(() => {
    document.body.setAttribute("data-theme", theme);
    document.getElementById("root")?.setAttribute("data-theme", theme);
  }, [theme]);

  useEffect(() => {
    if (loading) {
      setInputNotice("답변을 기다리는 중입니다.");
    } else if ((!selectedCategory || showCategorySelector) && isFirstTime) {
      setInputNotice(`  안녕하세요! 동성로 창업 지원 AI 챗봇입니다.

  창업 준비부터 정책 정보, 시장 트렌드까지 
  무엇이든 궁금한 점을 물어보세요!

    ✅원하는 분야를 선택해 주세요:
  • 창업 - 업종별 창업률, 생존률, 폐업률
  • 정책 - 창업 지원 정책, 대상별 맞춤 지원
  • 트렌드 - 시장 트렌드, 인기 업종, 소비자 관심도

  💵자본과 관련된 질문(예: 임대료, 매출, 수익성 등)은
  저희가 제공하는 정보로는 정확한 답변이 어렵습니다.

  🔍상세 점포 이름에(예: 스타벅스) 대해 물어볼때는 업종명을 붙여서
  (예: 스타벅스 카페 창업 어때?) 질문해주시면 더 정확한 답변을 얻을수 있습니다.

  우측 상단에서 언제든 다른 카테고리로 변경 가능합니다!
`);
    } else if ((!selectedCategory || showCategorySelector) && !isFirstTime) {
      setInputNotice("카테고리를 다시 선택해 주세요.");
    } else {
      setInputNotice("");
    }
  }, [loading, selectedCategory, showCategorySelector, isFirstTime]);

  const handleCategorySelect = (catKey) => {
    setSelectedCategory(catKey);
    setShowCategorySelector(false);
    setIsFirstTime(false);
    const introText =
      CATEGORY_DATA[catKey].description +
      "\n" +
      CATEGORY_DATA[catKey].examples.map((ex) => ex).join("\n");
    setMessages((prev) => [
      ...prev,
      { text: introText, isUser: false }
    ]);
  };

  const handleCategoryChange = () => {
    setShowCategorySelector(true);
    setSelectedCategory(null);
  };

  const sendMessage = async (msg) => {
    setMessages((msgs) => [...msgs, { text: msg, isUser: true }]);
    setLoading(true);
    try {
      // ===================================================================
      // [Netlify Functions 사용 시] - 아래 1줄 주석 해제하고 나머지 주석 처리
      // const endpoint = '/.netlify/functions/chat';
      // ===================================================================
      // [백엔드 직접 호출 시] - 아래 2줄 주석 해제 (도커, 소스코드 배포 기본값)
      const API_URL = process.env.REACT_APP_API_URL;
      const endpoint = `${API_URL}/api/chat`;
      // ===================================================================

      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: msg,
          category: selectedCategory ? CATEGORY_DATA[selectedCategory].value : null,
        }),
      });
      
      const data = await res.json();
      setMessages((msgs) => [
        ...msgs,
        { text: data.reply || "알 수 없는 오류가 발생했습니다.", isUser: false },
      ]);
    } catch (e) {
      setMessages((msgs) => [
        ...msgs,
        { text: "서버와 연결할 수 없습니다.", isUser: false },
      ]);
    }
    setLoading(false);
  };

  return (
    <div className={`chatbot-container ${theme}`}>
      <ChatHeader
        showCategoryChangeButton={!showCategorySelector}
        onCategoryChange={handleCategoryChange}
        loading={loading}
      />
      <ChatMessages
        messages={messages}
        loading={loading}
        showCategorySelector={showCategorySelector}
        selectedCategory={selectedCategory}
        inputNotice={inputNotice}
        onCategorySelect={handleCategorySelect}
      />
      <ChatInput
        onSend={sendMessage}
        disabled={loading || !selectedCategory || showCategorySelector}
        loading={loading}
      />
    </div>
  );
}
