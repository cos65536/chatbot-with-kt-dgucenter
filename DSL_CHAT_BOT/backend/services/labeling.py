from config.constants import CATEGORY_STARTUP, CATEGORY_POLICY, CATEGORY_TREND
from models.llm_model import llm_instance

# 전처리 관련 상수 및 함수 (테스트 코드에서 가져옴)
TOK_STARTUP = ["창업", "업종", "점포", "폐업률", "생존율", "창업률", "통계", "비율", "기간", "점포수", "폐점", "폐업", "유지"]
TOK_POLICY = ["지원", "공고", "신청", "서류", "자격", "대상", "절차", "대출", "보증", "바우처",
              "4대 보험", "세금", "인허가", "사업자등록", "임대차", "계약", "원상복구","정부사업", "정부지원", "정부정책"]
TOK_TREND = ["인기", "유행", "검색량", "수요", "유동인구", "트렌드", "동향", "요즘", "탕후루", "요아정"]
TOK_REGION = ["동성로", "대구 중구"]
TOK_D_GATE = ["서울", "강남", "부산", "서면", "달서구", "수성구", "전역", "전체", "전국", "세계", "글로벌", "해외",
              "날씨", "기상", "기온", "미세먼지", "정치", "정당", "정치인", "이재명", "윤석열", "우주", "위성", "발사체",
              "아파트", "주택", "부동산", "50년", "10년", "정부", "폭탄", "살인", "마약","총기"] 
SPECIAL_CHARS = ["##", "$$", "@@", "!!", "%%", "^^", "&&"]

def _norm_space(s: str) -> str:
    return " ".join((s or "").split())

def _sent_split(s: str) -> list:
    s = s.replace("\n", ".").replace("?", ".").replace("!", ".").replace("|", ".").replace("/", ".").replace("-", ".")
    parts = [p.strip() for p in s.split(".")]
    return [p for p in parts if p]

def _score_sentence(sent: str) -> int:
    t = sent
    score = 0
    if any(k in t for k in TOK_D_GATE): score += 10
    if any(k in t for k in TOK_REGION): score += 2
    if any(k in t for k in TOK_POLICY): score += 2
    if any(k in t for k in TOK_STARTUP): score += 1
    if any(k in t for k in TOK_TREND): score += 1
    return score

def _check_special_char_spam(question: str) -> bool:
    return any(char_pattern in question for char_pattern in SPECIAL_CHARS)

def preprocess_for_llm(question: str, max_chars: int = 300):
    q_lower = question.lower()
    is_d_gate = any(k in q_lower for k in TOK_D_GATE)
    is_policy = any(k in q_lower for k in TOK_POLICY)
    
    if _check_special_char_spam(question):
        is_d_gate = True

    q = _norm_space(question)
    sents = _sent_split(q)
    if not sents:
        return q[:max_chars], is_d_gate, is_policy
    
    scored = sorted((( _score_sentence(s), len(s), s) for s in sents), reverse=True)
    picked = [s for _, __, s in scored[:2]] or sents[:2]
    return _norm_space(". ".join(picked))[:max_chars], is_d_gate, is_policy

class QuestionLabeling:
    def __init__(self):
        self.llm = llm_instance

    def label_category_with_mini(self, question, category):
        q_for_llm, is_d_gate, is_policy = preprocess_for_llm(question)
        
        # D 게이트 키워드 처리
        if is_d_gate and not is_policy:
            return "unknown"
        
        # 정책 키워드 처리
        elif is_policy:
            return CATEGORY_POLICY
        
        # LLM 기반 분류
        prompt = (
            '질문: "{q}"\n'
            "힌트(참고): 사용자가 선택한 카테고리 -> {cat} (startup=창업, policy=정책, trend=트렌드)\n"
            "역할: 동성로(대구 중구) 창업 전용 라벨러. 출력은 A/B/C/D 중 **대문자 한 글자**.\n\n"
            "규칙(우선순위 고정):\n"
            "1) D 게이트 — 하나라도 맞으면 즉시 D\n"
            "    • 지역이 명시되고 동성로/대구 중구가 아님: 서울/강남/부산/서면/달서구/수성구/대구 전역·전체/전국/전 세계(세계/글로벌/해외)등\n"
            "    • 날씨/기상(오늘 날씨·기온·미세먼지·비/눈), 정치/정당/정치인(인기·평가 포함)\n"
            "    • 스팸 특수문자 반복(예: ## $$ @@), 비현실 업종(우주/위성/발사체)\n"
            "    • 부동산 일반 가격전망(아파트/주택/‘부동산 전체’) — 상가·점포 임대료/권리금은 예외\n"
            "2) B 오버라이드 — 아래 단어가 보이면 지역 없어도 B\n"
            "    지원/정책/공고/모집/신청/접수/마감/요건/자격/대상/서류/절차/대출/무이자/보증/바우처/보조금/융자/\n"
            "    R&D/시제품/컨설팅/멘토링/교육/4대 보험/세금/부가세/인허가/사업자등록/임대차/계약/원상복구/SNS 마케팅 컨설팅\n"
            "3) 위 둘이 아니고 동성로 창업 도메인이면 A/C 중 선택\n"
            "    • A: 지표·수치/타당성(창업률/폐업률/생존율/점포수/통계/기간비교, ‘~창업 어때?’ 등)\n"
            "    • C: 인기/유행/트렌드/검색량/유동인구/‘요즘’ 등 동향, **유명 프랜차이즈/인기 품목(예: 탕후루, 마라탕, 요아정 등)**\n"
        ).format(q=q_for_llm, cat=category)

        messages = [
            {"role": "system",
             "content": "동성로 전용 라벨러. **타지역·전세계·대구 전역·날씨·정치·스팸·우주·부동산 일반**이면 **무조건 D**. "
                        "정책 키워드면 **무조건 B**. 그 외는 A 또는 C. 반드시 한 글자(A/B/C/D)만 출력."},
            {"role": "user", "content": prompt},
        ]

        response = self.llm.generate_response(messages, max_new_tokens=4, do_sample=False)
        resp = (response or "").strip().upper() 
        
        if resp == "A":
            return CATEGORY_STARTUP
        elif resp == "B":
            return CATEGORY_POLICY
        elif resp == "C":
            return CATEGORY_TREND
        elif resp == "D":
            return "unknown"
        return "unknown"

# 전역 인스턴스
labeling = QuestionLabeling()