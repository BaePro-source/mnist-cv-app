from fastapi import FastAPI, UploadFile, File
import torch

from models.cnn_model import MNISTCNN
from utils.preprocess import preprocess_image

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 서버 시작 시 모델 1번만 로드
model = MNISTCNN().to(device)
model.load_state_dict(torch.load("models/mnist_cnn.pth", map_location=device))
model.eval()


@app.get("/")
def home():
    return {"messeage": "server working"}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 1. 업로드된 이미지 읽기
        contents = await file.read()

        # 2. 전처리
        input_tensor = preprocess_image(contents).to(device)

        # 3. 예측
        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.argmax(output, dim=1).item()

        # 4. 결과 반환
        return {
            "filename": file.filename,
            "prediction": prediction
        }

    except Exception as e:
        return {
            "error": str(e)
        }