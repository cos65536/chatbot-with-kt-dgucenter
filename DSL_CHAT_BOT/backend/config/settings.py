import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# API 키 설정 (환경변수로부터 가져오기)
SERVICE_KEY = os.getenv('SERVICE_KEY')

# API URL 설정
POLICY_API_URLS = {
    'url1': 'https://api.odcloud.kr/api/15132761/v1/uddi:181018f4-37d5-4500-b23f-9f9f2a840bc3',
    'url2': 'https://nidapi.k-startup.go.kr/api/kisedKstartupService/v1/getAnnouncementInformation/'
}

# 데이터 파일 경로
DATA_PATHS = {
    'startup_data': './data/master_summary_final.csv',
    'business_data': './data/final_data.csv'
}

# 데이터랩 API 설정 (환경변수로부터 가져오기)
NAVER_DATALAB_CONFIG = {
    'client_id': os.getenv('NAVER_CLIENT_ID'),
    'client_secret': os.getenv('NAVER_CLIENT_SECRET')
}