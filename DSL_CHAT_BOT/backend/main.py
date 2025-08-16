from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from rag_llm import (
    label_category_with_mini,
    llm_answer_with_rag,
    llm_answer_with_policy,
    llm_answer_with_trend,
    CATEGORY_STARTUP,
    CATEGORY_POLICY,
    CATEGORY_TREND,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/chat")
async def chat(request: Request):
    data = await request.json()
    question = data.get("message", "")
    selected_category = data.get("category", None)  # 프론트에서 선택한 카테고리 value (startup, policy, trend 등)

    if not question or not selected_category:
        return {"reply": "질문과 카테고리를 모두 입력해 주세요."} #디버깅용, 실제로는 UI상에서 선택해야 입력이 가능함

    # 믿음 mini로 질문의 실제 카테고리 분류
    predicted_category = label_category_with_mini(question, selected_category)

# 선택과 분류가 다르면 안내
    if predicted_category == "unknown":
        return {"reply": "더 공부하는 챗봇이 될게요!"}

    # 실제 답변은 BASE 모델 등 카테고리별 LLM에 위임
    # A와 C가 헷갈리는 경우 사용자 카테고리 우선
    if selected_category == CATEGORY_STARTUP and predicted_category in [CATEGORY_STARTUP, CATEGORY_TREND]:
        answer = llm_answer_with_rag(question)
    elif selected_category == CATEGORY_POLICY and predicted_category == CATEGORY_POLICY:
        answer = llm_answer_with_policy(question)
    elif selected_category == CATEGORY_TREND and predicted_category in [CATEGORY_STARTUP, CATEGORY_TREND]:
        answer = llm_answer_with_trend(question)
    else:
        answer = "질문이 현재 선택된 카테고리와 맞지 않아요. 카테고리를 변경해 주세요."

    return {"reply": answer}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)