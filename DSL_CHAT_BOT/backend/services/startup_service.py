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
        self.stats_corpus = []  # 통계 데이터 전용
        self.biz_corpus = []    # 사업장 데이터 전용
        self.stats_embeds = None
        self.biz_embeds = None
        self._load_data()
    
    def _load_data(self):
        try:
            # 1. 통계 데이터 로드
            df_stats = pd.read_csv(DATA_PATHS['startup_data'], encoding="utf-8")
            self.stats_corpus = [self.text_processor.row_to_text(row) for _, row in df_stats.iterrows()]
            
            # 2. 사업장 데이터 로드 (헤더 스킵)
            df_biz = pd.read_csv(DATA_PATHS['business_data'], encoding="utf-8", header=None)
            self.biz_corpus = [self.text_processor.business_row_to_text(row) for idx, row in df_biz.iterrows() if idx > 0]
            
            # 3. 분리 임베딩 생성
            print("통계 데이터 임베딩 생성 중...")
            self.stats_embeds = self.embedder.encode(
                self.stats_corpus, 
                convert_to_numpy=True,
                show_progress_bar=True
            )
            
            print("사업장 데이터 임베딩 생성 중...")
            self.biz_embeds = self.embedder.encode(
                self.biz_corpus,
                convert_to_numpy=True,
                show_progress_bar=True
            )
            
            print("✔️ 임베딩 생성 완료")
            
        except Exception as e:
            print(f"❌ 데이터 로드 오류: {e}")
            # 폴백: 통계 데이터만 로드 + 안전한 임베딩 생성
            try:
                df_stats = pd.read_csv(DATA_PATHS['startup_data'], encoding="utf-8")
                self.stats_corpus = [self.text_processor.row_to_text(row) for _, row in df_stats.iterrows()]
                self.biz_corpus = []
                
                # 🔥 중요: 폴백에서도 임베딩 생성
                print("폴백: 통계 데이터 임베딩 생성 중...")
                self.stats_embeds = self.embedder.encode(
                    self.stats_corpus, 
                    convert_to_numpy=True, 
                    show_progress_bar=True
                )
                self.biz_embeds = np.array([])  # 빈 배열로 초기화
                print("✔️ 폴백 임베딩 완료")
                
            except Exception as fallback_e:
                print(f"❌ 폴백 로드도 실패: {fallback_e}")
                # 최후의 수단: 빈 데이터로 초기화
                self.stats_corpus = []
                self.biz_corpus = []
                self.stats_embeds = np.array([])
                self.biz_embeds = np.array([])

    def search_context(self, query, topk_stats=5, topk_biz=3):
        """통계 데이터와 사업장 데이터를 별도로 검색"""
        q_emb = self.embedder.encode(query, convert_to_numpy=True)
        
        # 1. 통계 데이터 검색 (상위 5개)
        stats_results = []
        if len(self.stats_embeds) > 0:  # 🔥 안전 검사 추가
            stats_sims = np.dot(self.stats_embeds, q_emb)
            stats_ids = stats_sims.argsort()[-topk_stats:][::-1]
            stats_results = [self.stats_corpus[i] for i in stats_ids]
        
        # 2. 사업장 데이터 검색 (데이터 존재시)
        biz_results = []
        if len(self.biz_embeds) > 0:  # 🔥 임베딩 존재 여부 확인
            biz_sims = np.dot(self.biz_embeds, q_emb)
            biz_ids = biz_sims.argsort()[-topk_biz:][::-1]
            biz_results = [self.biz_corpus[i] for i in biz_ids]
        
        return stats_results + biz_results

    def llm_answer_with_rag(self, question, chat_history=None):
        """창업 상담 (실제 데이터 수치 정확히 제공)"""
        contexts = self.search_context(question)
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