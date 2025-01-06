import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
from torchvision import models
from PIL import Image
from ignore_this import 젠장에이스이공격은대체뭐냐

# print(젠장에이스이공격은대체뭐냐(1))

'''
data/
├── additional_train/
│   ├── edited/
│   └── unedited/
├── test/
│   ├── edited/
│   └── unedited/
├── train/
│   ├── edited/
│   └── unedited/
└── validation/
'''

os.chdir(os.path.dirname(os.path.realpath(__file__)))

# 전처리
transgender = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

tdata = datasets.ImageFolder('data/train', transform=transgender)
vdata = datasets.ImageFolder('data/train', transform=transgender)
tedata = datasets.ImageFolder('data/test', transform=transgender)

trl = DataLoader(tdata, batch_size=32, shuffle=True)
vll = DataLoader(vdata, batch_size=32, shuffle=False)
tel = DataLoader(tedata, batch_size=32, shuffle=False)

mp = 'best_model.pth'

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# 참조
if os.path.exists(mp):
    print(f"Loading existing model from {mp}...")
    model = SimpleCNN()
    model.load_state_dict(torch.load(mp))
    model.eval() 
else:
    print("Creating a new model...")
    model = SimpleCNN()

# 손실 함수, 최적화 
ct = nn.BCELoss()
ot = optim.Adam(model.parameters(), lr=0.001)

# 학습 
def trmo(model, trl, vll, n_ep=50):
    bestacc = 0.0
    for epoch in range(n_ep):
        model.train()
        rloss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            ot.zero_grad()
            outputs = model(inputs)
            labels = labels.float().view(-1, 1)  
            loss = ct(outputs, labels)
            loss.backward()
            ot.step()

            rloss += loss.item()
            predicted = outputs.round()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        eploss = rloss / len(train_loader)
        epacc = correct / total * 100
        print(f"Epoch [{epoch+1}/{n_ep}], Loss: {eploss:.4f}, Accuracy: {epacc:.2f}%")

        # 성능 평가
        model.eval()
        vlc = 0
        vlt = 0
        with torch.no_grad():
            for inputs, labels in vll:
                outputs = model(inputs)
                predicted = outputs.round()
                vlc += (predicted == labels.view(-1, 1)).sum().item()
                vlt += labels.size(0)
        
        vlacc = vlc / vlt * 100
        print(f"Validation Accuracy: {vlacc:.2f}%")

        # 모델 저장
        if vlacc > bestacc:
            bestacc = vlacc
            torch.save(model.state_dict(), mp)
            print(f"Model saved with validation accuracy: {vlacc:.2f}%")

# 학습 시작
if not os.path.exists(mp):
    print("Starting fresh training...")
    trmo(model, trl, vll, n_ep=50)

# 모델 평가
def evaluate_model(model, tel):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tel:
            outputs = model(inputs)
            predicted = outputs.round()
            correct += (predicted == labels.view(-1, 1)).sum().item()
            total += labels.size(0)

    acc = correct / total * 100
    print(f"Test Accuracy: {acc:.2f}%")

evaluate_model(model, tel)

# 예측 함수
def prima(image_path):
    image = Image.open(image_path)
    image = transgender(image).unsqueeze(0)  
    model.eval() 
    with torch.no_grad():
        output = model(image)
    prediction = output.round().item()  
    return prediction

# 정확도 계산
def calacc(validation_directory):
    crpr = 0
    ttpr = 0

    model.eval() 

    for image_name in os.listdir(validation_directory):
        image_path = os.path.join(validation_directory, image_name)
        if image_path.lower().endswith(('.jpg', '.jpeg')): 
            prediction = prima(image_path)

            actual_label = 1 if 'edited' in image_name else 0

            if prediction == actual_label:
                crpr += 1
            ttpr += 1

    accuracy = 100 * crpr / ttpr
    return accuracy

# 예측 실행 및 정확도 출력
vldir = os.path.join(os.getcwd(), 'data', 'validation')
'''print(f"Validation directory: {validation_directory}")

accuracy = calculate_accuracy(validation_directory)
print(f"Validation Accuracy: {accuracy:.2f}%")'''

for image_name in os.listdir(vldir):
    image_path = os.path.join(vldir, image_name)
    
    if image_path.lower().endswith(('.jpg', '.jpeg')):
        result = prima(image_path)
        print(f"Prediction for {image_name}: {result}")
