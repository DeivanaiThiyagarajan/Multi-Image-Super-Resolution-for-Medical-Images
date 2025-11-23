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
    def __init__(self, in_channels, out_channels):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        return x


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
# PROGRESSIVE UNET - From ProgressiveUNet_Training.ipynb
# ============================================================================

class ProgressiveUNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ProgressiveUNetBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        return x


class ProgressiveUNet(nn.Module):
    """
    Progressive UNet for producing multiple outputs at different levels
    Input: (B, 2, H, W) - prior and posterior slices
    Output: (B, 3, H, W) - three predictions (coarse, medium, fine)
    """
    def __init__(self, in_channels=2, out_channels=3, init_features=64):
        super(ProgressiveUNet, self).__init__()
        
        features = init_features
        
        # Encoder
        self.enc1 = ProgressiveUNetBlock(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc2 = ProgressiveUNetBlock(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc3 = ProgressiveUNetBlock(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc4 = ProgressiveUNetBlock(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = ProgressiveUNetBlock(features * 8, features * 16)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.dec4 = ProgressiveUNetBlock(features * 16, features * 8)
        
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.dec3 = ProgressiveUNetBlock(features * 8, features * 4)
        
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.dec2 = ProgressiveUNetBlock(features * 4, features * 2)
        
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.dec1 = ProgressiveUNetBlock(features * 2, features)
        
        # Progressive output heads (1, 2, 3 predictions)
        self.output1 = nn.Conv2d(features * 4, 1, kernel_size=1)  # At dec3 level
        self.output2 = nn.Conv2d(features * 2, 1, kernel_size=1)  # At dec2 level
        self.output3 = nn.Conv2d(features, 1, kernel_size=1)      # At dec1 level
    
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
        
        # Decoder with skip connections and progressive outputs
        x = self.upconv4(x)
        x = torch.cat([x, enc4], dim=1)
        x = self.dec4(x)
        
        x = self.upconv3(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)
        out1 = self.output1(x)  # First output (coarse)
        
        x = self.upconv2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)
        out2 = self.output2(x)  # Second output (medium)
        
        x = self.upconv1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)
        out3 = self.output3(x)  # Third output (fine)
        
        # Combine outputs (B, 3, H, W)
        outputs = torch.cat([out1, out2, out3], dim=1)
        return outputs


# ============================================================================
# UNET GAN - From UNet_GAN_Training.ipynb
# ============================================================================

class UNetGenerator(nn.Module):
    """
    UNet-based Generator for GAN architecture
    Input: (B, 2, H, W) - prior and posterior slices
    Output: (B, 1, H, W) - predicted middle slice
    """
    def __init__(self, in_channels=2, out_channels=1, init_features=64):
        super(UNetGenerator, self).__init__()
        
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
    
    # Map model names to (checkpoint filename, model class, init kwargs)
    checkpoint_map = {
        'unet': ('unet_best.pt', UNet, {'in_channels': 2, 'out_channels': 1, 'init_features': 64}),
        'deepcnn': ('deepcnn_best.pt', DeepCNN, {'in_channels': 2, 'out_channels': 1, 'num_blocks': [2, 2, 2, 2], 'base_features': 64}),
        'progressive_unet': ('progressive_unet_best.pt', ProgressiveUNet, {'in_channels': 2, 'out_channels': 3, 'init_features': 64}),
        'unet_gan': ('unet_gan_generator_best.pt', UNetGenerator, {'in_channels': 2, 'out_channels': 1, 'init_features': 64}),
    }
    
    model_name_lower = model_name.lower()
    
    if model_name_lower not in checkpoint_map:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(checkpoint_map.keys())}")
    
    checkpoint_file, model_class, init_kwargs = checkpoint_map[model_name_lower]
    checkpoint_path = os.path.join(models_dir, checkpoint_file)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Initialize and load model
    model = model_class(**init_kwargs).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
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
