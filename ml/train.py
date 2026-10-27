import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ml.model import OceanEmbedCNN
from ml.dataset import OceanDataset

def train_model():
    print("=" * 60)
    print("Starting OceanEmbed CNN Encoder-Decoder Training Pipeline")
    print("=" * 60)
    os.makedirs("models", exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Compute Device: {device}")
    
    # Initialize CNN Encoder-Decoder model
    model = OceanEmbedCNN(in_channels=5, out_channels=15, base_channels=32).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Architecture: OceanEmbedCNN (Encoder-Decoder with Skip Connections)")
    print(f"Trainable Parameters: {total_params:,}")
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    
    train_dataset = OceanDataset(data_dir="data/raw", mode="train")
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, Y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        scheduler.step()
        avg_loss = total_loss / max(1, len(train_loader))
        rmse = (avg_loss) ** 0.5
        print(f"Epoch [{epoch+1}/{epochs}] | Loss (MSE): {avg_loss:.4f} | Training RMSE: {rmse:.4f} °C")
        
    save_path = "models/oceanembed.pth"
    torch.save(model.state_dict(), save_path)
    torch.save(model.state_dict(), "models/model.pth")
    print("=" * 60)
    print(f"Training Complete! Weights successfully saved to {save_path}")
    print("=" * 60)

if __name__ == "__main__":
    train_model()
