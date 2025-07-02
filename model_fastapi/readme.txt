uvicorn app.main:app

이후 커맨드창 하나 더 열고
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d "{\"text\": \"테스트.\"}"