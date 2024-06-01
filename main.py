from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
import torch
import torch.nn.functional as F
import numpy as np
from model import DiseasePredictor


model = DiseasePredictor()
model.load_state_dict(torch.load('heart_disease_model.pth', map_location='cpu'))

def predict(input_data):
    model.eval()
    input_tensor = torch.tensor([input_data], dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
    return predicted_class

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/")
def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(
    request: Request,
    high_blood_pressure: str = Form(...),
    high_cholesterol: str = Form(...),
    bmi: float = Form(...),
    smoking: str = Form(...),
    fruit_consumption: str = Form(...),
    vegetable_consumption: str = Form(...),
    alcohol_consumption: str = Form(...)
):
    # 입력 데이터를 숫자로 변환
    high_blood_pressure = 1 if high_blood_pressure == "yes" else 0
    high_cholesterol = 1 if high_cholesterol == "yes" else 0
    smoking = 1 if smoking == "yes" else 0
    fruit_consumption = 1 if fruit_consumption == "yes" else 0
    vegetable_consumption = 1 if vegetable_consumption == "yes" else 0
    alcohol_consumption = 1 if alcohol_consumption == "yes" else 0

    input_data = [
        high_blood_pressure,
        high_cholesterol,
        bmi,
        smoking,
        fruit_consumption,
        vegetable_consumption,
        alcohol_consumption
    ]

    try:
        prediction = predict(input_data)
        if prediction == 0:
            result = "Low risk of disease"
        else:
            result = "High risk of disease"
            
    except Exception as e:
        return {"message": f"There was an error predicting the disease: {str(e)}"}

    return templates.TemplateResponse("result.html", {"request": request, "result": result})

