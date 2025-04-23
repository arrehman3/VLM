from torch.utils.data import DataLoader
from model import UNet
from data_loader import MedSegDataset
import torch
import torch.nn as nn
import torch.optim as optim

# Paths
image_dir = 'D:/VLM/data/raw/images'
mask_dir = 'D:/VLM/data/raw/masks'

# Dataset and loader
dataset = MedSegDataset(
    image_dir="D:/VLM/data/raw/images",
    mask_dir="D:/VLM/data/raw/masks"
)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Model
model = UNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 5
for epoch in range(epochs):
    total_loss = 0
    for imgs, masks in dataloader:
        preds = model(imgs)
        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Save model
torch.save(model.state_dict(), r'D:/VLM/models/unet_medseg.pth')