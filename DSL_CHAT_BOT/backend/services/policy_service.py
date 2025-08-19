import requests
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from models.embedding_model import embedding_instance
from models.llm_model import llm_instance
from config.settings import SERVICE_KEY, POLICY_API_URLS

class PolicyService:
    def __init__(self):
        self.embedder = embedding_instance
        self.llm = llm_instance
        self.policy_corpus = []
        self.policy_embeds = np.array([])
        self._load_policy_data()
    
    def _load_policy_data(self):
        # 정책 데이터 임베딩으로 처리
        try:
            # API 1
            url1 = POLICY_API_URLS['url1']
            params1 = {'page': 1, 'perPage': 30, 'serviceKey': SERVICE_KEY}
            response = requests.get(url1, params=params1, timeout=10)
            data = response.json()
            df_pol1 = pd.DataFrame(data['data'])
            for _, row in df_pol1.iterrows():
                text = f"{row.get('기관명', '')} {row.get('사업명', '')} {row.get('연령', '')} 지원정책"
                self.policy_corpus.append(text)
            
            # API 2  
            url2 = POLICY_API_URLS['url2']
            params2 = {'serviceKey': SERVICE_KEY, 'pageNo': 1, 'numOfRows': 30}
            response = requests.get(url2, params=params2, timeout=10)
            root = ET.fromstring(response.text)
            for item in root.findall('.//item'):
                row_data = {col.attrib['name']: col.text for col in item.findall('col')}
                text = f"{row_data.get('pbanc_ntrp_nm', '')} {row_data.get('intg_pbanc_biz_nm', '')} {row_data.get('biz_trgt_age', '')}창업지원"
                self.policy_corpus.append(text)
                
            print(f"정책 데이터 {len(self.policy_corpus)}건 로드")
            self.policy_embeds = self.embedder.encode(self.policy_corpus, convert_to_numpy=True, show_progress_bar=True)
        except Exception as e:
            print(f"정책 데이터 로드 실패: {e}")
            self.policy_corpus = []
            self.policy_embeds = np.array([])
    
    def llm_answer_with_policy(self, question):
        """URL 포함 정책 질의응답 (데이터 내 URL 정확 출력)"""
        if len(self.policy_corpus) == 0:
            return "죄송하지만 적절한 데이터를 찾지 못했어요. 다른 질문을 해보시는건 어떨까요?"
        q_emb = self.embedder.encode(question, convert_to_numpy=True)
        sims = np.dot(self.policy_embeds, q_emb)
        top_ids = sims.argsort()[-5:][::-1]
        contexts = [self.policy_corpus[i] for i in top_ids if sims[i] > 0.25]
        
        if not contexts:
            return "죄송하지만 적절한 데이터를 찾지 못했어요. 다른 질문을 해보시는건 어떨까요?"

        prompt = (
            f"현재: 2025년 8월, 대구 창업 정책 상담사\\n\\n"
            
            "[URL 처리 규칙 - 중요]\\n"
            "제공된 정책 데이터에 포함된 URL은 정확히 그대로 출력\\n"
            "데이터에 없는 URL은 절대 생성하지 마세요\\n"
            "전화번호(010, 1588, 1357 등 포함), 이메일은 생성 금지\\n\\n"
            "사용자 질문에 맞는 공감 멘트 먼저 생성하고 개요 출력 (예: '창업을 준비중이시군요!')"
            
            "[답변 구조]-> ()안에있는 내용은 참고자료로만 사용하고, 답변에는 포함시키지 않을것\\n"
            
            "🔍상세 정책 설명: (각 정책을 자연스럽게 연결하여 설명할것)\\n"
            "   - '1. [기관명]에서 주관하는 '[사업명]'이 있습니다. [3-4문장 상세설명]'\\n"
            "   - '2. '[사업명]'도 있습니다. [상세설명]'\\n"
            "   - '3. 그 외에도 [다른 정책들] 등이 있습니다.'\\n\\n"

            "(정책 목록 정리: 깔끔한 목록 형태로 재정리할것)\\n"
            " 📋지원 목록:\\n"
            "   • [사업명] - [기관명]\\n"
            "   • [사업명] - [기관명]\\n\\n"
            
            "🔗관련 링크\\n"
            "- 위 정보들은 아래 링크에서 더 자세히 확인할 수 있습니다:\\n"
            "- 창업진흥원: https://www.kised.or.kr/ \\n"
            "- 대구창업허브: https://startup.daegu.go.kr/ \\n"
            "(데이터에 URL이 없으면 이 섹션 생략)\\n\\n"
            
            f"정책 데이터:\\n{chr(10).join(contexts)}\\n\\n"
            f"질문: {question}\\n\\n"
            f"답변: 정책 정보와 데이터에 포함된 정확한 URL을 함께 제공하세요."
        )
         
        messages = [
            {
                "role": "system",
                "content": "대구 창업 정책 전문가. 데이터에 포함된 정확한 URL은 그대로 출력하되, "
                        "데이터에 없는 URL은 절대 생성하지 않음. 정확한 정보만 제공."
                        "- 정책, 지원이 아닌 정치적 질문에는 더 공부하는 챗봇이 될께요 라고만 출력할것"
            },
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm.generate_response(
            messages,
            max_new_tokens=512,
            do_sample=False
        )
        
        return response


# 전역 인스턴스
policy_service = PolicyService()