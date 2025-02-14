import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader,Dataset
import pandas as pd
from PIL import Image
import os

from enum import Enum

class Column(Enum):
    dish_id = 0
    total_calories = 1
    total_mass = 2
    total_fat = 3
    total_carb = 4
    total_protein = 5
    num_ingrs = 6

# 1. Dataset Class
class FoodDataset(Dataset):
    def __init__(self, image_dir, metadata_csv, transform=None):
        self.image_dir = image_dir
        self.calories_data = pd.read_csv(metadata_csv, header = None, usecols=range(2))
        self.transform = transform

    def __len__(self):
        return len(self.calories_data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self. calories_data.iloc[idx].iloc[Column.dish_id] + ".png")
        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(self.calories_data.iloc[idx].iloc[Column.total_calories], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

transform = transforms.Compose([
    transforms.Resize((400, 400)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Load Dataset
train_dataset = FoodDataset("../realsense_overhead/", "../metadata/dish_metadata_cafe1.csv", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 4. Load Pretrained Model
model = models.resnet50(weights=True)
model.fc = nn.Linear(model.fc.in_features, 1)

# 5. Loss and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Save Model
torch.save(model.state_dict(), "calorie_estimation_model.pth")
