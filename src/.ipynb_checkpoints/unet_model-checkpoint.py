"""
UNet Model for Multi-Image Super-Resolution
Takes pre and post slices as input and produces the middle slice as output
Supports two triplet configurations:
  1. pre=i, post=i+2, middle=i+1 (z-spacing: 1.5mm)
  2. pre=i, post=i+4, middle=i+2 (z-spacing: 3.0mm)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import json


class UNetBlock(nn.Module):
    """Double convolution block with batch norm"""
    def __init__(self, in_channels, out_channels):
        super(UNetBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    UNet Architecture for medical image super-resolution
    Input: (B, 2, H, W) - prior and posterior slices
    Output: (B, 1, H, W) - predicted middle slice
    """
    def __init__(self, in_channels=2, out_channels=1, init_features=64):
        super(UNet, self).__init__()
        
        features = init_features
        
        # Encoder
        self.enc1 = UNetBlock(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc2 = UNetBlock(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc3 = UNetBlock(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc4 = UNetBlock(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = UNetBlock(features * 8, features * 16)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.dec4 = UNetBlock(features * 16, features * 8)
        
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.dec3 = UNetBlock(features * 8, features * 4)
        
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.dec2 = UNetBlock(features * 4, features * 2)
        
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.dec1 = UNetBlock(features * 2, features)
        
        # Final output layer
        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder with skip connections
        enc1 = self.enc1(x)
        x = self.pool1(enc1)
        
        enc2 = self.enc2(x)
        x = self.pool2(enc2)
        
        enc3 = self.enc3(x)
        x = self.pool3(enc3)
        
        enc4 = self.enc4(x)
        x = self.pool4(enc4)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip connections
        x = self.upconv4(x)
        x = torch.cat([x, enc4], dim=1)
        x = self.dec4(x)
        
        x = self.upconv3(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)
        
        x = self.upconv2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)
        
        x = self.upconv1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)
        
        # Output
        x = self.final_conv(x)
        return x


class MRIDataset(Dataset):
    """Dataset wrapper for MRI triplets"""
    def __init__(self, triplets):
        """
        Args:
            triplets: list of tuples (prior_slice, middle_slice, posterior_slice)
                     each is numpy array of shape (H, W)
        """
        self.triplets = triplets
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        prior, middle, posterior = self.triplets[idx]
        
        # Stack prior and posterior as input (2 channels)
        input_data = np.stack([prior, posterior], axis=0).astype(np.float32)
        # Middle as target (1 channel)
        target_data = np.expand_dims(middle, axis=0).astype(np.float32)
        
        return (
            torch.from_numpy(input_data),
            torch.from_numpy(target_data)
        )


class UNetTrainer:
    """Trainer class for UNet model"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu', 
                 learning_rate=1e-4, model_save_dir='models'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        print(f"✅ Model initialized on device: {device}")
        print(f"📊 Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc="Training", leave=False)
        for inputs, targets in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validating", leave=False)
            for inputs, targets in pbar:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss
    
    def train(self, train_loader, val_loader, epochs=100, early_stopping_patience=15):
        """Train model with early stopping"""
        print(f"\n🚀 Starting training for {epochs} epochs...")
        print(f"⏱️  Early stopping patience: {early_stopping_patience} epochs\n")
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            print(f"Epoch {epoch}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f}", end="")
            
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_loss, is_best=True)
                print(" ✨ (Best)")
            else:
                self.patience_counter += 1
                print(f" (patience: {self.patience_counter}/{early_stopping_patience})")
                
                if self.patience_counter >= early_stopping_patience:
                    print(f"\n⚠️  Early stopping triggered after {epoch} epochs")
                    break
        
        print(f"\n✅ Training completed!")
        print(f"📊 Best validation loss: {self.best_val_loss:.4f}")
        self.save_training_logs()
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        if is_best:
            save_path = self.model_save_dir / 'unet_best.pt'
            torch.save(checkpoint, save_path)
            print(f"   💾 Best model saved: {save_path}")
        
        # Save latest checkpoint
        save_path = self.model_save_dir / 'unet_latest.pt'
        torch.save(checkpoint, save_path)
    
    def save_training_logs(self):
        """Save training history and plots"""
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        
        log_path = self.model_save_dir / 'training_history.json'
        with open(log_path, 'w') as f:
            json.dump(history, f, indent=4)
        print(f"📝 Training history saved: {log_path}")
        
        # Plot training curves
        self._plot_training_curves()
    
    def _plot_training_curves(self):
        """Plot and save training curves"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss', linewidth=2)
        plt.plot(self.val_losses, label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss (MSE)', fontsize=12)
        plt.title('UNet Training Progress', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        plot_path = self.model_save_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 Training curves saved: {plot_path}")


def create_dummy_dataset(num_samples=100, img_size=256):
    """Create dummy dataset for testing"""
    print("⚠️  Creating dummy dataset for testing...")
    triplets = []
    for _ in range(num_samples):
        prior = np.random.randn(img_size, img_size).astype(np.float32)
        middle = np.random.randn(img_size, img_size).astype(np.float32)
        posterior = np.random.randn(img_size, img_size).astype(np.float32)
        triplets.append((prior, middle, posterior))
    return triplets


def main():
    """Main training script"""
    # Configuration
    BATCH_SIZE = 4
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    EARLY_STOPPING_PATIENCE = 10
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL_SAVE_DIR = 'models'
    
    print("=" * 70)
    print("🧠 UNet Model for MRI Super-Resolution Training")
    print("=" * 70)
    print(f"📱 Device: {DEVICE}")
    print(f"📦 Batch size: {BATCH_SIZE}")
    print(f"🔄 Epochs: {EPOCHS}")
    print(f"📊 Learning rate: {LEARNING_RATE}")
    print()
    
    # Create or load dataset
    # TODO: Replace with actual dataset loading from data_generator.py
    print("📥 Loading dataset...")
    triplets = create_dummy_dataset(num_samples=1000)  # Replace with real data
    
    # Split into train/val
    split_idx = int(0.8 * len(triplets))
    train_triplets = triplets[:split_idx]
    val_triplets = triplets[split_idx:]
    
    print(f"   Training samples: {len(train_triplets)}")
    print(f"   Validation samples: {len(val_triplets)}")
    print()
    
    # Create datasets and dataloaders
    train_dataset = MRIDataset(train_triplets)
    val_dataset = MRIDataset(val_triplets)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Initialize model
    print("🏗️  Building UNet model...")
    model = UNet(in_channels=2, out_channels=1, init_features=64)
    
    # Initialize trainer
    trainer = UNetTrainer(model, device=DEVICE, learning_rate=LEARNING_RATE, 
                         model_save_dir=MODEL_SAVE_DIR)
    
    # Train model
    print()
    trainer.train(train_loader, val_loader, epochs=EPOCHS, 
                 early_stopping_patience=EARLY_STOPPING_PATIENCE)
    
    print("\n" + "=" * 70)
    print("✅ Training completed successfully!")
    print(f"📂 Models saved to: {MODEL_SAVE_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
