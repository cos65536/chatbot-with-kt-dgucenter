FROM node:18-alpine

WORKDIR /app

# 의존성 복사 및 설치
COPY package.json ./
COPY package-lock.json ./
RUN npm install

# 소스 코드 복사
COPY . .

# 빌드 (React 앱)
RUN npm run build

# 정적 파일 서빙 (serve 이용, 필요시 npm install -g serve)
RUN npm install -g serve

EXPOSE 3000

CMD ["serve", "-s", "build", "-l", "3000"]