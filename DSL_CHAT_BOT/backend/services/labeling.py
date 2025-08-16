from config.constants import CATEGORY_STARTUP, CATEGORY_POLICY, CATEGORY_TREND
from models.llm_model import llm_instance

class QuestionLabeling:
    def __init__(self):
        self.llm = llm_instance

    def label_category_with_mini(self, question, category):
        #라벨링모델 사용
        prompt = (
            "질문: \"{q}\"\n"
            "사용자가 선택한 카테고리: {c}\n"
            "카테고리 후보:\n"
            "[A] 창업 (아이템, 점포명, 업종별 창업률, 생존율, 폐업률, 통계,점포정보,서비스 ,창업, 창업전망,업종 등)\n"
            "[B] 정책 (지원정책, 정부/지자체/기관의 사업 및 공고 등)\n"
            "[C] 트렌드 (업종, 아이템, 키워드의 인기·변화·검색량 등)\n"
            "[D] 해당 없음(날씨,정치인,인사,전세계트렌드,기타,서울 등)\n"
            "이 질문이 사용자가 선택한 카테고리에 적절하다면 해당카테고리의 알파벳만 출력해. 적절하지 않다면 다른 카테고리의 알파벳만 출력해."
            "경북, 대구, 중구, 동성로에 해당하지 않는 지역이 언급된 질문과, 날씨, 인물 등에 관련된 질문은 해당없음으로 분류해"
        ).format(q=question.strip(), c=category)
        messages = [
            {"role": "system", "content": "너는 질문을 카테고리별로 라벨링하는 전문가야."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm.generate_response(messages, max_new_tokens=4, do_sample=False)  #차피 알파벳 한글자라서 토큰 작게해서 빠르게 출력
        
        #응답 바탕으로 카테고리에 맞는 함수 호출
        resp = response.strip().upper()
        if "A" in resp:
            return CATEGORY_STARTUP
        elif "B" in resp:
            return CATEGORY_POLICY
        elif "C" in resp:
            return CATEGORY_TREND
        elif "D" in resp:
            return "unknown"
        return "unknown"

# 전역 인스턴스
labeling = QuestionLabeling()