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
        self.data = pd.read_csv(metadata_csv, usecols=range(8))  # Load only first 8 columns

        # Print column names for debugging
        print("Columns in CSV:", self.data.columns)

        # Construct full image paths by iterating through subfolders
        self.image_paths = []
        for subdir in os.listdir(root_dir):
            img_path = os.path.join(root_dir, subdir, "rgb.png")
            if os.path.exists(img_path):
                self.image_paths.append(img_path)

        # Debugging: Check if paths exist
        for path in self.image_paths:
            if not os.path.exists(path):
                print(f"File not found: {path}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        calories = self.data.iloc[idx][self.data.columns[1]]  # Adjust if needed based on actual column name
        return image, torch.tensor(calories, dtype=torch.float32)

# Example usage:
dataset_path = r"D:/study/CS172B/realsense_overhead"
metadata_path = r"D:\study\CS172B\172B-food-project\nutrition5k_dataset_metadata_dish_metadata_cafe1.csv"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = FoodDataset(dataset_path, metadata_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

print(f"Loaded {len(train_dataset)} images.")

# 4. Load Pretrained Model
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 1)  # Modify last layer for regression

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