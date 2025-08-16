import pandas as pd
import numpy as np
from models.embedding_model import embedding_instance
from models.llm_model import llm_instance
from utils.text_processor import text_processor
from config.settings import DATA_PATHS

class StartupService:
    def __init__(self):
        self.embedder = embedding_instance
        self.llm = llm_instance
        self.text_processor = text_processor
        self.all_corpus = []
        self.corpus_embeds = None
        self._load_data()
    
    def _load_data(self):
        # 통합 창업 데이터 준비 (기존 + 새로운 사업장 데이터)
        try:
            # 기존 창업률 통계 데이터
            df = pd.read_csv(DATA_PATHS['startup_data'], encoding="utf-8")
            corpus = df.apply(self.text_processor.row_to_text, axis=1).tolist()
            self.all_corpus.extend(corpus)
            print(f"기존 창업률 데이터 {len(corpus)}건 로드")
            
            # 새로운 사업장 데이터
            df_biz = pd.read_csv(DATA_PATHS['business_data'], encoding="utf-8", header=None)
            biz_corpus = [self.text_processor.business_row_to_text(row) for idx, row in df_biz.iterrows() if idx > 0]
            self.all_corpus.extend(biz_corpus)
            print(f"새로운 사업장 데이터 {len(biz_corpus)}건 로드")
            
            print(f"총 {len(self.all_corpus)}건 데이터로 통합 임베딩 생성")
        except Exception as e:
            print(f"데이터 로드 오류: {e}")
            # 기존 데이터라도 사용
            df = pd.read_csv(DATA_PATHS['startup_data'], encoding="utf-8")
            self.all_corpus = df.apply(self.text_processor.row_to_text, axis=1).tolist()
        
        # 통합 임베딩 생성
        self.corpus_embeds = self.embedder.encode(self.all_corpus, convert_to_numpy=True, show_progress_bar=True)
    
    def search_context(self, query, topk=3):
        # 질문과 유사한 창업률 컨텍스트 검색 (통합 데이터에서 검색)
        q_emb = self.embedder.encode(query, convert_to_numpy=True)
        sims = np.dot(self.corpus_embeds, q_emb)
        top_ids = sims.argsort()[-topk:][::-1]
        return [self.all_corpus[i] for i in top_ids]
    
    def llm_answer_with_rag(self, question, chat_history=None):
        """창업 상담 (실제 데이터 수치 정확히 제공)"""
        contexts = self.search_context(question, topk=3)
        if not contexts:
            return """안녕하세요! 대구 동성로 창업 지원 챗봇입니다.
                    죄송하지만 질문과 관련된 자료를 찾지 못했습니다.
                    더 구체적인 키워드로 다시 질문해주세요.
                    예시: "동성로 카페 창업률은?", "치킨집 폐업률 통계는?"
                    """

        prompt = (
            "현재 시점: 2025년 8월\n"
            "당신은 대구 동성로 창업 통계 전문가입니다.\n\n"

            "[핵심 원칙]\n"
            "✅ 데이터 수치 정확히 제시\n"
            "서울등, 동성로외 지역은 답변하지 않음\n"
            "❌ 데이터에 없는 정보 추측 금지, 찾을수없는 데이터는 데이터가 없다고 솔직하게 말할것\n\n"
            "통계 데이터는 2023년 부터 2025년까지 모두 반영되어야 합니다.\n\n"
            "절대 임의로 데이터를 생성하지 않을것\n"
            "[답변 구조]\n" 
            "1. 핵심 통계: 창업률, 폐업률, 생존율 데이터에 기반한 요약정보 제공\n" 
            "2. 창업 실용 조언\n"
            "   - 통계 데이터 기반 객관적 분석\n"
            "   - 수치를 바탕으로 한 현실적 조언\n"
            "   - 업종별 주의사항 (데이터 근거)\n\n"

            f"데이터:\n{chr(10).join(contexts)}\n"
            f"질문: {question}\n"
            f"답변:"
        )

        messages = [
            {
                "role": "system", 
                "content": "창업 통계 전문가. 데이터에 있는 정확한 수치는 그대로 제시. "
                        "없는 데이터는 절대 생성하지 않을것. "
                        "창업률/폐업률/생존율/사업장 수 등 실제 통계 정확히 제공."
            },
            {"role": "user", "content": prompt}
        ]

        response = self.llm.generate_response(
            messages, 
            max_new_tokens=712,
            do_sample=False,
        )

        return response

# 전역 인스턴스
startup_service = StartupService()