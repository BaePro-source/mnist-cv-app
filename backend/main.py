from fastapi import FastAPI
from fastapi import UploadFile, File

app = FastAPI()

@app.get("/")
def home():
    return {"message": "server working"}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    return {
        "filename": file.filename,
        "size": len(contents)
    }