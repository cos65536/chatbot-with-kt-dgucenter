# 기존 main.py에서 import할 수 있도록 호환성 유지
from config.constants import CATEGORY_STARTUP, CATEGORY_POLICY, CATEGORY_TREND
from services.labeling import labeling
from services.startup_service import startup_service
from services.policy_service import policy_service
from services.trend_service import trend_service

# 기존 함수명 유지 (호환성을 위해)
def label_category_with_mini(question, category):
    return labeling.label_category_with_mini(question, category)

def llm_answer_with_rag(question, chat_history=None):
    return startup_service.llm_answer_with_rag(question, chat_history)

def llm_answer_with_policy(question):
    return policy_service.llm_answer_with_policy(question)

def llm_answer_with_trend(question):
    return trend_service.llm_answer_with_trend(question)