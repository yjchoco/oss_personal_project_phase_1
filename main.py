from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
import torch
import torch.nn.functional as F
import numpy as np
from disease_model import DiseasePredictor


PATH = './disease_model.pth'
model = DiseasePredictor()
model.load_state_dict(torch.load(PATH, map_location='cpu'))
model.eval()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/")
def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
def predict(request: Request, age: int = Form(...), weight: float = Form(...), blood_pressure: float = Form(...)):
    try:
        # 입력받은 데이터를 텐서로 변환
        input_data = torch.tensor([[age, weight, blood_pressure]], dtype=torch.float32)
        
        # 모델 예측
        output = model(input_data)
        prediction = torch.argmax(F.softmax(output, dim=1), dim=1).item()

        if prediction == 0:
            result = "Low risk of disease"
        else:
            result = "High risk of disease"
            
    except Exception as e:
        return {"message": f"There was an error predicting the disease: {str(e)}"}

    return templates.TemplateResponse("result.html", {"request": request, "result": result})

