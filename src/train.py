import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
import os

# 1. Dataset Class
class FoodDataset(Dataset):
    def __init__(self, root_dir, metadata_csv, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = pd.read_csv(metadata_csv, header=None, usecols=range(8))  # Load only first 8 columns

        # Print column names for debugging
        print("Columns in CSV:", self.data.columns)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        dish_id = self.data.iloc[idx].iloc[0]
        calories = self.data.iloc[idx].iloc[1]  # Adjust if needed based on actual column name

        img_path = f"{dataset_path}{dish_id}.png"
        if not os.path.isfile(img_path):
            image = Image.new("RGB", (400, 400), (0, 0, 0))  # Black image
            return self.transform(image), torch.tensor(-1)  # -1 as an invalid label
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(calories, dtype=torch.float32)

# Example usage:
dataset_path = r"../realsense_overhead/"
metadata_path = r"../nutrition_data.csv"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = FoodDataset(dataset_path, metadata_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = models.resnet50(weights=True)
model.fc = nn.Linear(model.fc.in_features, 1)  # Modify last layer for regression

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

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