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
# FASTDDPM MODEL - From FastDDPM_Advanced.ipynb
# ============================================================================

def sinusoidal_timestep_embedding(timesteps, dim, max_period=10000):
    """
    Generate sinusoidal timestep embeddings.
    
    Args:
        timesteps: (B,) tensor of timestep indices
        dim: Embedding dimension (must be even)
        max_period: Maximum period for sinusoidal waves
    
    Returns:
        (B, dim) tensor of positional encodings
    """
    half_dim = dim // 2
    freqs = torch.exp(
        -torch.log(torch.tensor(max_period, dtype=torch.float32)) *
        torch.arange(half_dim, dtype=torch.float32) / half_dim
    )
    
    # (B,) -> (B, half_dim)
    args = timesteps[:, None] * freqs[None, :]
    
    # Concatenate sin and cos
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    
    return embedding


class FastNoiseScheduler:
    """
    Fast noise scheduler with non-uniform timestep sampling.
    Emphasizes early and late timesteps while maintaining DDPM's variance schedule.
    
    T=10 timesteps: 40% from early steps (0-699), 60% from late steps (699-999)
    """
    def __init__(self, num_diffusion_steps=1000, T=10):
        """
        Args:
            num_diffusion_steps: Total diffusion steps in full DDPM (default 1000)
            T: Number of steps to sample (default 10 for ~100x speedup)
        """
        self.num_diffusion_steps = num_diffusion_steps
        self.T = T
        
        # Compute alphas and betas for full schedule
        self.betas = torch.linspace(0.0001, 0.02, num_diffusion_steps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Non-uniform timestep selection
        early_steps = int(0.4 * T)
        late_steps = T - early_steps
        
        early_indices = torch.linspace(0, int(0.7 * num_diffusion_steps) - 1, early_steps).long()
        late_indices = torch.linspace(int(0.7 * num_diffusion_steps), num_diffusion_steps - 1, late_steps).long()
        
        self.timesteps = torch.cat([early_indices, late_indices])
    
    def get_timesteps(self):
        """Return sampled timesteps"""
        return self.timesteps
    
    def get_alpha_cumprod(self, t):
        """Get cumulative product of alphas for timestep t"""
        return self.alphas_cumprod[t]
    
    def get_sigma(self, t_prev, t):
        """Compute standard deviation for reverse step"""
        alpha_t = self.alphas_cumprod[t]
        alpha_t_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)
        beta_t = 1 - alpha_t / alpha_t_prev
        return torch.sqrt(beta_t)


class DoubleConv(nn.Module):
    """Double convolution block (Conv -> BN -> ReLU -> Conv -> BN -> ReLU)"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class UNet2D(nn.Module):
    """
    U-Net architecture for conditional image generation.
    Incorporates timestep embeddings via MLPs in skip connections.
    """
    def __init__(self, in_channels=2, out_channels=1, base_channels=64, time_embed_dim=256):
        super().__init__()
        
        self.time_embed_dim = time_embed_dim
        
        # Initial projection
        self.initial_conv = DoubleConv(in_channels, base_channels)
        
        # Encoder
        self.down1 = nn.MaxPool2d(2)
        self.conv1 = DoubleConv(base_channels, base_channels * 2)
        
        self.down2 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(base_channels * 2, base_channels * 4)
        
        self.down3 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(base_channels * 4, base_channels * 8)
        
        # Bottleneck
        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16)
        
        # Decoder with time conditioning
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.time_mlp3 = nn.Sequential(
            nn.Linear(time_embed_dim, base_channels * 8),
            nn.SiLU(),
            nn.Linear(base_channels * 8, base_channels * 8)
        )
        self.conv_up3 = DoubleConv(base_channels * 16, base_channels * 8)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.time_mlp2 = nn.Sequential(
            nn.Linear(time_embed_dim, base_channels * 4),
            nn.SiLU(),
            nn.Linear(base_channels * 4, base_channels * 4)
        )
        self.conv_up2 = DoubleConv(base_channels * 8, base_channels * 4)
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.time_mlp1 = nn.Sequential(
            nn.Linear(time_embed_dim, base_channels * 2),
            nn.SiLU(),
            nn.Linear(base_channels * 2, base_channels * 2)
        )
        self.conv_up1 = DoubleConv(base_channels * 4, base_channels * 2)
        
        # Final output
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
    
    def forward(self, x, time_embedding):
        """
        Args:
            x: (B, 2, H, W) input image
            time_embedding: (B, time_embed_dim) timestep embedding
        """
        # Encoder
        x0 = self.initial_conv(x)
        
        x = self.down1(x0)
        x1 = self.conv1(x)
        
        x = self.down2(x1)
        x2 = self.conv2(x)
        
        x = self.down3(x2)
        x3 = self.conv3(x)
        
        # Bottleneck
        x = self.bottleneck(x3)
        
        # Decoder with time conditioning
        x = self.up3(x)
        time_cond3 = self.time_mlp3(time_embedding)
        time_cond3 = time_cond3[:, :, None, None]  # (B, C, 1, 1)
        x3_conditioned = x3 * time_cond3
        x = torch.cat([x, x3_conditioned], dim=1)
        x = self.conv_up3(x)
        
        x = self.up2(x)
        time_cond2 = self.time_mlp2(time_embedding)
        time_cond2 = time_cond2[:, :, None, None]
        x2_conditioned = x2 * time_cond2
        x = torch.cat([x, x2_conditioned], dim=1)
        x = self.conv_up2(x)
        
        x = self.up1(x)
        time_cond1 = self.time_mlp1(time_embedding)
        time_cond1 = time_cond1[:, :, None, None]
        x1_conditioned = x1 * time_cond1
        x = torch.cat([x, x1_conditioned], dim=1)
        x = self.conv_up1(x)
        
        # Output
        x = self.final_conv(x)
        
        return x


class FastDDPM(nn.Module):
    """
    Fast Denoising Diffusion Probabilistic Model with T=10 timesteps.
    Uses non-uniform timestep sampling for 100x faster generation.
    
    Input: (B, 2, H, W) - prior and posterior slices
    Output: (B, 1, H, W) - predicted middle slice
    """
    def __init__(self, in_channels=2, out_channels=1, base_channels=64, 
                 time_embed_dim=256, num_diffusion_steps=1000, T=10):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.time_embed_dim = time_embed_dim
        self.num_diffusion_steps = num_diffusion_steps
        self.T = T
        
        # Time embedding
        self.time_embedding_layer = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # UNet2D
        self.unet = UNet2D(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            time_embed_dim=time_embed_dim
        )
        
        # Noise schedule
        self.scheduler = FastNoiseScheduler(num_diffusion_steps=num_diffusion_steps, T=T)
    
    def forward(self, x_t, t):
        """
        Training forward pass: predict noise.
        
        Args:
            x_t: (B, 2, H, W) noisy input at timestep t
            t: (B,) timestep indices
        
        Returns:
            noise_pred: (B, 1, H, W) predicted noise
        """
        # Get timestep embedding
        time_embed = sinusoidal_timestep_embedding(t, self.time_embed_dim).to(x_t.device)
        time_embed = self.time_embedding_layer(time_embed)
        
        # Predict noise
        noise_pred = self.unet(x_t, time_embed)
        
        return noise_pred
    
    def sample(self, conditions, num_steps=10, device='cuda'):
        """
        DDIM sampling for fast generation.
        
        Args:
            conditions: (B, 2, H, W) conditioning slices (prior and posterior)
            num_steps: Number of reverse diffusion steps (typically 10)
            device: Device to use
        
        Returns:
            samples: (B, 1, H, W) generated middle slices
        """
        self.eval()
        
        batch_size = conditions.shape[0]
        
        # Start from noise
        x = torch.randn(batch_size, self.out_channels, conditions.shape[2], conditions.shape[3], device=device)
        
        # Get timesteps
        timesteps = self.scheduler.get_timesteps()[:num_steps]
        
        with torch.no_grad():
            for i, t in enumerate(timesteps):
                t_tensor = torch.full((batch_size,), t, dtype=torch.long, device=device)
                
                # Concatenate conditioning with current noise
                x_in = torch.cat([conditions, x], dim=1)
                
                # Predict noise
                noise_pred = self.forward(x_in, t_tensor)
                
                # DDIM update
                alpha_t = self.scheduler.get_alpha_cumprod(t).to(device)
                
                if i < num_steps - 1:
                    t_prev = timesteps[i + 1]
                    alpha_t_prev = self.scheduler.get_alpha_cumprod(t_prev).to(device)
                else:
                    alpha_t_prev = torch.tensor(1.0, device=device)
                
                # DDIM step
                x = (torch.sqrt(alpha_t_prev) / torch.sqrt(alpha_t)) * (x - torch.sqrt(1 - alpha_t) * noise_pred) + \
                    torch.sqrt(1 - alpha_t_prev) * noise_pred
        
        return x

# ============================================================================
# MODEL LOADER FUNCTION
# ============================================================================

def load_model(model_name, device='cuda'):
    """
    Load the best model checkpoint for the given model name.
    Uses correct model architectures from notebooks.
    
    Args:
        model_name: Model identifier - 'unet', 'deepcnn', 'progressive_unet', 'unet_gan', 'fastddpm'
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
        'unet_combined': ('unet_combined_best.pt', UNet, {'in_channels': 2, 'out_channels': 1, 'init_features': 64}),
        'deepcnn': ('deepcnn_best.pt', DeepCNN, {'in_channels': 2, 'out_channels': 1, 'num_blocks': [2, 2, 2, 2], 'base_features': 64}),
        'progressive_unet': ('progressive_unet_best.pt', ProgressiveUNet, {'base_features': 64}),
        'unet_gan': ('unet_gan_best.pt', UNetGenerator, {'in_channels': 2, 'out_channels': 1, 'base_features': 64}),
        'fastddpm': ('fastddpm_advanced_best.pth', FastDDPM, {'in_channels': 2, 'out_channels': 1, 'base_channels': 64, 'time_embed_dim': 256, 'num_diffusion_steps': 1000, 'T': 10}),
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
    
    # Handle different checkpoint formats
    if model_name_lower == 'unet_gan':
        model.load_state_dict(checkpoint['generator_state_dict'])
    elif model_name_lower == 'fastddpm':
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
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
    print("  - fastddpm")
