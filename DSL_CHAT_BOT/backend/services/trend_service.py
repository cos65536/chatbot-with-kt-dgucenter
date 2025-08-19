import requests
import json
import numpy as np
from datetime import datetime, timedelta
from models.embedding_model import embedding_instance
from models.llm_model import llm_instance
from config.settings import NAVER_DATALAB_CONFIG

class TrendService:
    def __init__(self):
        self.embedder = embedding_instance
        self.llm = llm_instance
        
        # 네이버 데이터랩 API 설정
        self.client_id = NAVER_DATALAB_CONFIG.get('client_id', '')
        self.client_secret = NAVER_DATALAB_CONFIG.get('client_secret', '')
        self.api_url = "https://openapi.naver.com/v1/datalab/search"

    def _extract_keywords(self, question):
        """질문에서 키워드 추출"""
        extract_prompt = f"다음 질문에서 트렌드 분석할 키워드들을 쉼표로 구분해서 최대 3개 추출해줘: {question}"
        messages = [
            {"role": "system", "content": "키워드만 간단히 추출해줘."},
            {"role": "user", "content": extract_prompt}
        ]
        response = self.llm.generate_response(messages, max_new_tokens=50, do_sample=False)
        keywords = [k.strip() for k in response.split(',') if k.strip()]
        return keywords[:3]

    def _fetch_trend_data(self, keywords):
        """네이버 데이터랩 API 호출"""
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        
        headers = {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret,
            "Content-Type": "application/json"
        }
        
        keyword_groups = [{"groupName": keyword, "keywords": [keyword]} for keyword in keywords]
        body = {
            "startDate": start_date,
            "endDate": end_date,
            "timeUnit": "month",
            "keywordGroups": keyword_groups
        }
        
        response = requests.post(self.api_url, headers=headers, data=json.dumps(body))
        return response.json()

    def _convert_to_text(self, keywords, trend_data):
        """트렌드 데이터를 텍스트로 변환"""
        texts = []
        for i, result in enumerate(trend_data['results']):
            keyword = keywords[i]
            data_points = result.get('data', [])
            data_str = [f"{item['period']}:{item['ratio']}" for item in data_points]
            texts.append(f"{keyword} 검색량: {', '.join(data_str)}")
        return texts

    def llm_answer_with_trend(self, question):
        """네이버 데이터랩 트렌드 데이터 기반 창업 답변 생성 (간결 버전)"""
        
        # 키워드 추출 및 트렌드 데이터 조회
        keywords = self._extract_keywords(question)
        trend_data = self._fetch_trend_data(keywords)
        trend_texts = self._convert_to_text(keywords, trend_data)
        
        # 임베딩 검색
        q_emb = self.embedder.encode(question, convert_to_numpy=True)
        trend_embeds = self.embedder.encode(trend_texts, convert_to_numpy=True)
        sims = np.dot(trend_embeds, q_emb)
        top_ids = sims.argsort()[-3:][::-1]
        contexts = [trend_texts[i] for i in top_ids]
        
        if not contexts:
            return "트렌드 데이터를 찾을 수 없어 정확한 분석이 어렵습니다. 다른 키워드로 다시 질문해 주세요!"
        
        # 간결한 프롬프트
        prompt = (
            f"현재: 2025년 8월, 대구 동성로 창업 트렌드 전문가\n\n"
            
            "[필수 준수사항]\n"
            "- 인물, 정치적인 질문에는 더 공부하는 챗봇이 될께요 라고만 출력할것"
            "- 네이버 ratio 값은 상대수치(절대값 아님)\n"
            "- 검색량 ≠ 실제 매출 (반드시 명시)\n"
            "- URL  (https, http, www 포함), 전화번호 (010, 1588, 1357 등 포함), 구체적 금액 생성 금지\n"
            "- 추측성 수치 제공 금지\n\n"
            "- 날씨, 인사 같은 일상질문에는 더 공부하는 챗봇이 될께요 라고만 출력할것\n"
            
            "[답변 구조]\n"
            "📊 트렌드 분석\n"
            "- 검색량 변화 패턴\n"
            "- 상승/하락 요인\n\n"
            
            "🤔 창업 관점\n"
            "- 시장 진입 타이밍\n"
            "- 경쟁 강도 예측\n"
            "- 동성로 적합성\n\n"

            "✅ 실행 제안\n"
            "- 구체적 사업 아이디어\n"
            "- 차별화 전략\n\n"

            "‼️ 주의사항\n"
            "- 데이터 한계 (검색량≠수익성)\n"
            "- 추가 검토 필요사항\n\n"
            
            "참고 데이터:\n" + "\n".join(contexts)
            + f"\n\n질문: {question}\n"
            + "답변:"
        )

        messages = [
            {
                "role": "system", 
                "content": "네이버 데이터랩 전문가 & 대구 동성로 창업 컨설턴트. "
                        "검색 트렌드 데이터 활용해 창업 분석하되, 데이터 한계와 위험 요소 반드시 제시. "
                        "추측하지 않고 데이터 기반 인사이트만 제공."
                        "날씨,인삿말,인물명 등의 질문에는 더 공부하는 챗봇이 될께요 라고만 출력할것"
            },
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm.generate_response(
            messages, 
            max_new_tokens=500,
            do_sample=False,
        )
        
        return response



# 전역 인스턴스
trend_service = TrendService()
