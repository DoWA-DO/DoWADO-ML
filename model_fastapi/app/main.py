from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
from model import Model
from data_processing import preprocess_text_kiwi, label_decoding

app = FastAPI()

# 모델 인스턴스 생성
model = Model()

class TextInput(BaseModel):
    text: str

@app.post("/predict")
async def predict(request: TextInput):
    text = request.text
    text = preprocess_text_kiwi(text)
    pred = model.classify_dataframe(text)
    pred = label_decoding(pred)
    return pred

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
