import torch
import torch.nn as nn

class DiseasePredictor(nn.Module):
    def __init__(self):
        super(DiseasePredictor, self).__init__()
        self.fc1 = nn.Linear(7, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = DiseasePredictor()
model.load_state_dict(torch.load('heart_disease_model.pth', map_location='cpu'))

def predict(input_data):
    model.eval()
    input_tensor = torch.tensor([input_data], dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
    return predicted_class

input_data = [1, 1, 28, 1, 1, 1, 0]  # 입력 데이터 예시
prediction = predict(input_data)
print(f"Predicted Class of Heart Disease: {prediction}")
