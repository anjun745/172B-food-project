import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image
import os

# 1. Dataset Class
class FoodDataset(Dataset):
    def __init__(self, image_dir, metadata_csv, transform=None):
        self.image_dir = image_dir
        self.data = pd.read_csv(metadata_csv)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.data.iloc[idx]["dish_id"] + ".jpg")
        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(self.data.iloc[idx]["total_calories"], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# 2. Data Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Load Dataset
train_dataset = FoodDataset("path_to_images", "metadata/dish_metadata_cafe1.csv", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

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
