"""
Model Loader - Loads model architectures with correct implementations from notebooks
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# UNET MODEL - From UNet_Training.ipynb
# ============================================================================

class UNetBlock(nn.Module):
    """Double convolution block with batch normalization"""
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
    
    
class ProgressiveUNetBlock(nn.Module):
    """Basic UNet encoder/decoder block with convolution and skip connection support"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ProgressiveUNetBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)
    
class GANUNetBlock(nn.Module):
    """Basic UNet encoder/decoder block"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(GANUNetBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
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
    
    
class UNetStage(nn.Module):
    """
    Single UNet stage for Progressive UNet
    Input: 2 slices (e.g., i and i+4)
    Output: 1 slice prediction (e.g., i+2)
    """
    def __init__(self, in_channels=2, out_channels=1, base_features=64):
        super(UNetStage, self).__init__()
        
        # Encoder
        self.enc1 = ProgressiveUNetBlock(in_channels, base_features)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = ProgressiveUNetBlock(base_features, base_features * 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.enc3 = ProgressiveUNetBlock(base_features * 2, base_features * 4)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.enc4 = ProgressiveUNetBlock(base_features * 4, base_features * 8)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = ProgressiveUNetBlock(base_features * 8, base_features * 16)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(base_features * 16, base_features * 8, kernel_size=2, stride=2)
        self.dec4 = ProgressiveUNetBlock(base_features * 16, base_features * 8)
        
        self.upconv3 = nn.ConvTranspose2d(base_features * 8, base_features * 4, kernel_size=2, stride=2)
        self.dec3 = ProgressiveUNetBlock(base_features * 8, base_features * 4)
        
        self.upconv2 = nn.ConvTranspose2d(base_features * 4, base_features * 2, kernel_size=2, stride=2)
        self.dec2 = ProgressiveUNetBlock(base_features * 4, base_features * 2)
        
        self.upconv1 = nn.ConvTranspose2d(base_features * 2, base_features, kernel_size=2, stride=2)
        self.dec1 = ProgressiveUNetBlock(base_features * 2, base_features)
        
        # Final output
        self.final = nn.Conv2d(base_features, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder
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
        x = self.final(x)
        
        return x


class ProgressiveUNet(nn.Module):
    """
    Progressive UNet with 3 stages:
    - Stage 1: UNet1(i, i+4) -> i+2_pred
    - Stage 2A: UNet2(i, i+2_pred) -> i+1_pred
    - Stage 2B: UNet3(i+2_pred, i+4) -> i+3_pred
    """
    def __init__(self, base_features=64):
        super(ProgressiveUNet, self).__init__()
        
        # Stage 1: Predict middle slice (i+2)
        self.unet1 = UNetStage(in_channels=2, out_channels=1, base_features=base_features)
        
        # Stage 2: Predict adjacent slices
        self.unet2 = UNetStage(in_channels=2, out_channels=1, base_features=base_features)  # (i, i+2) -> i+1
        self.unet3 = UNetStage(in_channels=2, out_channels=1, base_features=base_features)  # (i+2, i+4) -> i+3
    
    def forward(self, slices):
        
        batch_size = slices.shape[0]
        
        # Extract individual slices
        i = slices[:, 0:1, :, :]      # (B, 1, H, W)
        i_plus_1 = slices[:, 1:2, :, :]  # (B, 1, H, W)
        i_plus_2 = slices[:, 2:3, :, :]  # (B, 1, H, W)
        i_plus_3 = slices[:, 3:4, :, :]  # (B, 1, H, W)
        i_plus_4 = slices[:, 4:5, :, :]  # (B, 1, H, W)
        
        # Stage 1: Predict i+2 from (i, i+4)
        input_stage1 = torch.cat([i, i_plus_4], dim=1)  # (B, 2, H, W)
        pred_i_plus_2 = self.unet1(input_stage1)  # (B, 1, H, W)
        
        # Stage 2A: Predict i+1 from (i, predicted i+2)
        input_stage2a = torch.cat([i, pred_i_plus_2], dim=1)  # (B, 2, H, W)
        pred_i_plus_1 = self.unet2(input_stage2a)  # (B, 1, H, W)
        
        # Stage 2B: Predict i+3 from (predicted i+2, i+4)
        input_stage2b = torch.cat([pred_i_plus_2, i_plus_4], dim=1)  # (B, 2, H, W)
        pred_i_plus_3 = self.unet3(input_stage2b)  # (B, 1, H, W)
        
        return pred_i_plus_1, pred_i_plus_2, pred_i_plus_3


# ============================================================================
# DEEPCNN MODEL - From DeepCNN_Training.ipynb
# ============================================================================

class ResidualBlock(nn.Module):
    """Residual Block for DeepCNN"""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class DeepCNN(nn.Module):
    """
    DeepCNN Architecture for medical image super-resolution
    Based on ResNet with residual blocks for skip connections
    
    Input: (B, 2, H, W) - prior and posterior slices
    Output: (B, 1, H, W) - predicted middle slice
    """
    def __init__(self, in_channels=2, out_channels=1, num_blocks=[2, 2, 2, 2], base_features=64):
        super(DeepCNN, self).__init__()
        
        self.base_features = base_features
        
        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, base_features, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_features)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(ResidualBlock, base_features, base_features, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(ResidualBlock, base_features, base_features * 2, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(ResidualBlock, base_features * 2, base_features * 4, num_blocks[2], stride=1)
        self.layer4 = self._make_layer(ResidualBlock, base_features * 4, base_features * 8, num_blocks[3], stride=1)
        
        # Output layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_conv = nn.Conv2d(base_features * 8, out_channels, kernel_size=1)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        
        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Output
        x = self.output_conv(x)
        
        return x

# ============================================================================
# UNET GAN - From UNet_GAN_Training.ipynb
# ============================================================================

class UNetGenerator(nn.Module):
    """
    UNet Generator for medical image super-resolution
    Used as generator in GAN framework
    
    Input: (B, 2, H, W) - prior and posterior slices
    Output: (B, 1, H, W) - super-resolved middle slice
    """
    def __init__(self, in_channels=2, out_channels=1, base_features=64):
        super(UNetGenerator, self).__init__()
        
        # Encoder
        self.enc1 = GANUNetBlock(in_channels, base_features)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = GANUNetBlock(base_features, base_features * 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.enc3 = GANUNetBlock(base_features * 2, base_features * 4)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.enc4 = GANUNetBlock(base_features * 4, base_features * 8)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = GANUNetBlock(base_features * 8, base_features * 16)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(base_features * 16, base_features * 8, kernel_size=2, stride=2)
        self.dec4 = GANUNetBlock(base_features * 16, base_features * 8)
        
        self.upconv3 = nn.ConvTranspose2d(base_features * 8, base_features * 4, kernel_size=2, stride=2)
        self.dec3 = GANUNetBlock(base_features * 8, base_features * 4)
        
        self.upconv2 = nn.ConvTranspose2d(base_features * 4, base_features * 2, kernel_size=2, stride=2)
        self.dec2 = GANUNetBlock(base_features * 4, base_features * 2)
        
        self.upconv1 = nn.ConvTranspose2d(base_features * 2, base_features, kernel_size=2, stride=2)
        self.dec1 = GANUNetBlock(base_features * 2, base_features)
        
        # Final output
        self.final = nn.Conv2d(base_features, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder
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
        x = self.final(x)
        
        return x

# ============================================================================
# MODEL LOADER FUNCTION
# ============================================================================

def load_model(model_name, device='cuda'):
    """
    Load the best model checkpoint for the given model name.
    Uses correct model architectures from notebooks.
    
    Args:
        model_name: Model identifier - 'unet', 'deepcnn', 'progressive_unet', 'unet_gan'
        device: 'cuda' or 'cpu'
    
    Returns:
        model: Loaded PyTorch model in eval mode on specified device
    
    Raises:
        ValueError: If model_name not recognized or checkpoint not found
    """
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(parent_dir, 'models')
    print(models_dir)
    
    # Map model names to (checkpoint filename, model class, init kwargs)
    checkpoint_map = {
        'unet': ('unet_best.pt', UNet, {'in_channels': 2, 'out_channels': 1, 'init_features': 64}),
        'deepcnn': ('deepcnn_best.pt', DeepCNN, {'in_channels': 2, 'out_channels': 1, 'num_blocks': [2, 2, 2, 2], 'base_features': 64}),
        'progressive_unet': ('progressive_unet_best.pt', ProgressiveUNet, {'base_features': 64}),
        'unet_gan': ('unet_gan_best.pt', UNetGenerator, {'in_channels': 2, 'out_channels': 1, 'base_features': 64}),
    }
    
    model_name_lower = model_name.lower()
    
    if model_name_lower not in checkpoint_map:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(checkpoint_map.keys())}")
    
    checkpoint_file, model_class, init_kwargs = checkpoint_map[model_name_lower]
    checkpoint_path = os.path.join(models_dir, checkpoint_file)
    print(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Initialize and load model
    model = model_class(**init_kwargs).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if(model_name_lower == 'unet_gan'):
        model.load_state_dict(checkpoint['generator_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    #model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    print(f"✓ Loaded {model_name.upper()} model from {os.path.basename(checkpoint_path)}")
    
    return model


if __name__ == "__main__":
    print("Model Loader Module - Correct implementations from notebooks")
    print("\nAvailable models:")
    print("  - unet")
    print("  - deepcnn")
    print("  - progressive_unet")
    print("  - unet_gan")
