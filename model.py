import torch
import torch.nn as nn

class DiseasePredictor(nn.Module):
    def __init__(self):
        super(DiseasePredictor, self).__init__()
        self.fc1 = nn.Linear(4, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 모델 로드
PATH = './disease_model.pth'
model = DiseasePredictor()
model.load_state_dict(torch.load(PATH, map_location='cpu'))
model.eval()
