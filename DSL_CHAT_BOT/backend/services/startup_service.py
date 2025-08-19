import pandas as pd
import numpy as np
import re
from collections import Counter
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

    def detect_main_sector(self, question):
        """질문에서 가장 관련 높은 업종 1개만 추출"""
        text_key = question.lower()
        matched_sectors = []
        
        # 1. 동의어 기반 매칭
        for sector, keywords in self.text_processor.SYNONYMS.items():
            for keyword in keywords:
                if keyword.lower() in text_key:
                    matched_sectors.append(sector)
        
        # 2. 업종명 직접 매칭
        for sector in self.text_processor.SYNONYMS.keys():
            if sector.lower() in text_key:
                matched_sectors.append(sector)
        
        # 3. 매칭된 업종이 있으면 가장 빈도 높은 것 선택
        if matched_sectors:
            sector_counter = Counter(matched_sectors)
            return sector_counter.most_common(1)[0][0]
        
        return "기타"

    def get_sector_statistics(self, sector):
        """특정 업종의 모든 통계 데이터 추출 및 정렬"""
        all_sector_stats = [
            stat for stat in self.stats_corpus 
            if f"[통계]" in stat and sector.lower() in stat.lower()
        ]
        
        # 연도별 정렬 (2020 → 2025)
        all_sector_stats_sorted = sorted(
            all_sector_stats,
            key=lambda x: int(re.search(r'(\d{4})년', x).group(1)) if re.search(r'(\d{4})년', x) else 0,
            reverse=True   # 이 부분이 핵심!
        )
        return all_sector_stats_sorted

    def get_sector_businesses(self, sector, user_keywords):
        """특정 업종의 사업장 데이터를 우선순위에 따라 선별"""
        # 해당 업종의 사업장 데이터
        biz_examples = [
            biz for biz in self.biz_corpus 
            if f"[사업장]" in biz and sector.lower() in biz.lower()
        ]
        
        # 사용자 키워드가 포함된 사업장 우선 선택
        keyword_matches = []
        other_matches = []
        
        for biz in biz_examples:
            # 사업장 이름 추출
            name_match = re.search(r'\[사업장\] ([^\(]+)', biz)
            name = name_match.group(1).strip() if name_match else ""
            
            # 상태 정보 추출
            status = ""
            if "영업" in biz:
                status = "영업"
            elif "폐업" in biz:
                status = "폐업"
            elif "취소" in biz:
                status = "취소"
            else:
                status_match = re.search(r'\([^,]+,\s*([^\)]+)\)', biz)
                if status_match:
                    status = status_match.group(1).strip()
            
            # 키워드 매칭
            matched = False
            for kw in user_keywords:
                if kw.lower() in name.lower():
                    matched = True
                    break
            
            if matched:
                keyword_matches.append((name, status))
            else:
                other_matches.append((name, status))
        
        # 영업/폐업 상태별 필터링
        open_keyword = [b for b in keyword_matches if "영업" in b[1]]
        open_others = [b for b in other_matches if "영업" in b[1]]
        closed_keyword = [b for b in keyword_matches if "폐업" in b[1]]
        closed_others = [b for b in other_matches if "폐업" in b[1]]
        
        # 3개 우선 선택: 키워드 영업 > 일반 영업 > 키워드 폐업 > 일반 폐업
        selected = []
        
        selected.extend(open_keyword[:min(3, len(open_keyword))])
        
        if len(selected) < 3:
            needed = 3 - len(selected)
            selected.extend(open_others[:min(needed, len(open_others))])
        
        if len(selected) < 3:
            needed = 3 - len(selected)
            selected.extend(closed_keyword[:min(needed, len(closed_keyword))])
        
        if len(selected) < 3:
            needed = 3 - len(selected)
            selected.extend(closed_others[:min(needed, len(closed_others))])
        
        return selected, len(biz_examples)

    def analyze_question(self, question):
        """질문 종합 분석 (코랩의 inspect_question 로직)"""
        # 주 업종 감지
        main_sector = self.detect_main_sector(question)
        
        # 키워드 정보
        sector_keywords = self.text_processor.SYNONYMS.get(main_sector, [])
        
        # 사용자 질문에서 직접 언급된 키워드 추출
        user_keywords = []
        question_lower = question.lower()
        for keyword in sector_keywords + [main_sector]:
            if re.search(rf'\b{re.escape(keyword.lower())}\b', question_lower):
                user_keywords.append(keyword)
        
        # 해당 업종 통계 데이터
        statistics = self.get_sector_statistics(main_sector)
        
        # 해당 업종 사업장 사례
        business_examples, total_businesses = self.get_sector_businesses(main_sector, user_keywords)
        
        return {
            "sector": main_sector,
            "keywords": sector_keywords,
            "user_keywords": user_keywords,
            "statistics": statistics,
            "business_examples": business_examples,
            "total_businesses": total_businesses
        }

    def enhanced_search_context(self, question):
        """업종 분석 기반 향상된 컨텍스트 검색"""
        # 기본 임베딩 검색
        basic_contexts = self.search_context(question, topk_stats=5, topk_biz=3)
        
        # 질문 분석
        analysis = self.analyze_question(question)
        
        # 분석된 업종의 모든 통계 데이터 추가
        sector_stats = analysis["statistics"]
        
        # 중복 제거하면서 통합
        all_contexts = []
        
        # 1. 업종별 통계 데이터 우선 추가
        for stat in sector_stats[:3]:  # 최근 3년 데이터만
            if stat not in all_contexts:
                all_contexts.append(stat)
        
        # 2. 기본 임베딩 검색 결과 추가 (중복 제거)
        for context in basic_contexts:
            if context not in all_contexts:
                all_contexts.append(context)
        
        return all_contexts[:8]  # 최대 8개로 제한

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
        # 1. 컨텍스트 검색
        contexts = self.enhanced_search_context(question)
        if not contexts:
            return """안녕하세요! 대구 동성로 창업 지원 챗봇입니다.
#                 죄송하지만 질문과 관련된 자료를 찾지 못했습니다.
#                 더 구체적인 키워드로 다시 질문해주세요.
#                 예시: "동성로 카페 창업률은?", "치킨집 폐업률 통계는?"
#                 """

        # 2. 질문 분석
        analysis = self.analyze_question(question)

        # 3. 핵심 통계 부분 직접 포맷팅 (연도 오름차순 정렬 포함)
        stats_with_year = []
        for stat in analysis['statistics'][:6]:  # 2020~2025 6년치
            match = re.match(r'\[통계\]\s*(\d{4})년.*?:\s*(.+)', stat)
            if match:
                year, rest = match.groups()
                stats_with_year.append((int(year), f"**{year}년**: {rest.strip()}"))

        # 연도 오름차순 정렬
        stats_with_year.sort(key=lambda x: x[0])

        # 정렬 후 stats_lines 만듦
        stats_lines = [line for _, line in stats_with_year]

        # 4. 대표 사업장 정형화 출력 (최대 3개)
        biz_lines = []
        for name, status in analysis.get('business_examples', [])[:3]:
            biz_lines.append(f"{len(biz_lines)+1}. {name} ({status})")
        if not biz_lines:
            biz_lines.append("데이터 없음")

        # 5. LLM에 보낼 간단 요약 문자열 생성
        stats_summary = " / ".join([line.split(": ",1)[1] for line in stats_lines]) or "통계 데이터가 부족합니다."

        prompt = (
            "현재 시점: 2025년 8월\n"
            "당신은 대구 동성로 창업 전문가입니다.\n"
            f"다음은 최근 6년간 주요 통계 수치입니다:\n{stats_summary}\n"
            "[핵심 원칙]\n"
            "✅ 데이터 수치 정확히 제시\n"
            "서울등, 동성로외 지역은 답변하지 않음\n"
            "❌ 데이터에 없는 정보 추측 및 임의 생성 금지, 찾을수없는 데이터는 데이터가 없다고 솔직하게 말할것\n\n"
            "통계 데이터는 2020년 부터 2025년까지 모두 반영되어야 합니다.\n\n"
            "아래 1번 2번,정보 출력 금지\n"

            "1. 통계 해석: 핵심 통계를 기반해 인사이트 도출 및 시사점 제시\n"
            "2. 창업 실용 조언: 창업자에게 도움이 될만한 실용적인 조언 제공(데이터를 기반하되 수치를 기반하지 않아야함), 총 3줄 생성. 각 줄은 각각 다른 종류의 조언 \n"
            "3. 요약: 위 내용을 요약해서 한 줄로 정리해 주세요.\n\n"
            f"질문: {question}\n"
            "답변:"
        )
        messages = [
            {"role": "system", "content": "창업 통계 전문가. 데이터를 기반으로 정확하고 간결한 조언 제공."},
            {"role": "user", "content": prompt}
        ]
        llm_advice = self.llm.generate_response(messages, max_new_tokens=512, do_sample=False)

        # 6. 최종 정형화 출력 합치기
        output = "✅ 동성로 카페 창업 통계 분석 (2020-2025)\n\n"
        output += "📊 핵심 통계\n\n" + "\n".join(f"- {line}" for line in stats_lines) + "\n\n"
        output += "🏢 현재 영업중인 대표사업장\n"
        output += "\n".join(biz_lines) + "\n\n"
        output += llm_advice

        return output

# 전역 인스턴스
startup_service = StartupService()