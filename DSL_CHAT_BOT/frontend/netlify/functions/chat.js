/*
 * [Netlify Functions 전용] - Netlify 배포 시에만 사용됨
 * 도커 배포 시에는 이 파일 사용 안함
 */

const fetch = require('node-fetch');

exports.handler = async (event) => {
  try {
    // 클라이언트에서 전달된 데이터 파싱
    const { message, category } = JSON.parse(event.body);
    
    // 환경 변수에서 실제 API URL 가져오기
    const API_URL = process.env.REACT_APP_API_URL;
    
    // 실제 백엔드 API 호출
    const response = await fetch(`${API_URL}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, category })
    });

    const data = await response.json();
    
    return {
      statusCode: 200,
      body: JSON.stringify(data),
      headers: { 'Access-Control-Allow-Origin': '*' }
    };
  } catch (error) {
    return {
      statusCode: 500,
      body: JSON.stringify({ error: error.message }),
      headers: { 'Access-Control-Allow-Origin': '*' }
    };
  }
};