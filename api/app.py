from fastapi import FastAPI, UploadFile, File
import uvicorn
from utils import load_model, process_prediction
import tempfile
import os

app = FastAPI()
model = load_model()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        content = await file.read()
        temp.write(content)
        temp.flush()
        
        # Make prediction
        outputs = model.predict(temp.name)
        predictions = process_prediction(outputs)
        
        # Clean up
        os.unlink(temp.name)
        
    return {"predictions": predictions}