# Multi-Image Super-Resolution for Medical Slice Interpolation

This repository contains code, experiments, and analysis for **multi-image super-resolution (slice interpolation)** in prostate MRI. The goal is to synthesize missing intermediate MRI slices by conditioning on neighboring slices, improving through-plane resolution without additional scanning.

The project compares **deterministic CNN/UNet models**, **adversarial training**, **progressive multi-stage interpolation**, and **fast diffusion models (Fast-DDPM)** on a large prostate MRI biopsy dataset with **3 mm and 6 mm inter-slice spacing**.

📄 This repository accompanies the report:  
**“Multi-Image Super-Resolution for Medical Slice Interpolation”**  
*Deivanai Thiyagarajan, University of Florida*

---

## Key Ideas

- **Multi-image slice interpolation**: Predict a missing middle slice using two neighboring slices.
- **Variable spacing awareness**:
  - **Short-range interpolation (3 mm gap)**: slices *(i, i+2 → i+1)*
  - **Long-range interpolation (6 mm gap)**: slices *(i, i+4 → i+2)*
- **Progressive interpolation**: Decompose large-gap prediction into easier sub-problems using a hierarchical UNet pipeline.
- **Loss design matters**: Combining pixel-wise, perceptual, and structural losses significantly improves reconstruction fidelity.
- **Diffusion models**: Fast-DDPM offers probabilistic generation and fast inference, but currently underperforms deterministic UNets for this task.

---

## Repository Structure

├── src/ # Model architectures, training loops, data loaders
│ ├── models/ # UNet, DeepCNN, Progressive UNet, Fast-DDPM
│ ├── losses/ # MSE, perceptual (VGG), SSIM losses
│ ├── data/ # Triplet and window-based data generators
│ └── visualization/ # Slice and volume visualization utilities
│
├── notebooks/ # Training, evaluation, and analysis notebooks
│
├── data/ # Dataset manifests (DICOM data excluded)
│
├── models/ # Saved checkpoints
│ ├── unet_mse_best.pt
│ ├── unet_combined_best.pt
│ ├── unet_gan_best.pt
│ ├── progressive_unet_best.pt
│ └── fast_ddpm_best.pt
│
├── results/ # Training curves and qualitative visualizations
└── requirements.txt
---

## Dataset

- **Dataset**: Prostate MRI–Ultrasound Fusion Biopsy dataset  
- **Modality**: T2-weighted MRI  
- **Patients**: ~1,151 total (≈840 with usable T2w scans)
- **Resolution**:
  - In-plane: ~0.66 × 0.66 mm
  - Through-plane: **3 mm and 6 mm**
- **Preprocessing**:
  - Per-volume z-score normalization
  - Spatial resizing to 256×256
  - Light augmentation (horizontal flips, ±5° rotations)
- **Splits**: Patient-level (70% train / 15% val / 15% test)

---

## Models Implemented

### DeepCNN (Baseline)
- Residual CNN inspired by early super-resolution models
- Optimized with **MSE loss**
- Serves as a simple convolutional baseline

---

### UNet (MSE)
- Standard encoder–decoder with skip connections
- Input: two neighboring slices `(B, 2, H, W)`
- Output: predicted middle slice `(B, 1, H, W)`
- Strong deterministic baseline

---

### UNet (Combined Loss)
- Same architecture as MSE UNet
- **Loss = MSE + perceptual (VGG) + SSIM**
- Improves texture realism and structural fidelity
- Best overall performance across spacings

---

### UNet-GAN
- UNet generator + PatchGAN discriminator
- Loss = reconstruction + perceptual + adversarial
- Produces sharper textures but only marginal gains over combined-loss UNet

---

### Progressive UNet (Multi-Stage)
Designed specifically for **large inter-slice gaps (6 mm)**.

**Stages**:
1. `(i, i+4) → i+2` (coarse middle slice)
2. `(i, pred_i+2) → i+1`
3. `(pred_i+2, i+4) → i+3`

- Each stage is a UNet
- Multi-scale MSE loss with higher weight on the central slice
- Best-performing model for **6 mm spacing**

---

### Fast-DDPM
- Conditional diffusion model with UNet backbone
- Accelerated sampling (T=10 steps)
- Generates anatomically coherent slices
- Faster inference, but **lower PSNR/SSIM than UNet-based methods**

---

## Quantitative Results

**Test Set Performance (Mean SSIM / PSNR)**

| Model | SSIM (3mm) | PSNR (3mm) | SSIM (6mm) | PSNR (6mm) |
|------|-----------|------------|------------|------------|
| DeepCNN | 0.8217 | 26.30 | 0.5940 | 20.83 |
| UNet (MSE) | 0.8797 | 29.21 | 0.6530 | 21.91 |
| **UNet (Combined Loss)** | **0.8804** | **29.21** | 0.6586 | 22.23 |
| UNet-GAN | 0.8808 | 29.14 | 0.6574 | 22.08 |
| **Progressive UNet** | 0.7958 | 25.80 | **0.6645** | **22.44** |
| Fast-DDPM (T=10) | 0.7590 | 24.14 | 0.4920 | 21.78 |

---

## Key Observations

- **UNet-based models dominate** diffusion models for slice interpolation.
- **Combined-loss UNet** achieves the best overall reconstruction quality.
- **Progressive UNet** is the most robust model for **6 mm spacing**.
- GAN training yields limited gains over well-designed deterministic losses.
- Fast-DDPM is promising but currently trades accuracy for speed.

---

## Visualizations

Saved under `results/`:
- Training curves for all models
- Axial & sagittal reconstructions (3 mm vs 6 mm)
- Single-triplet qualitative comparisons
- Progressive UNet multi-stage outputs

---

## Evaluation Notes

⚠️ **Always report metrics separately for 3 mm and 6 mm spacing**.  
Aggregating both gaps lowers reported SSIM/PSNR and can unfairly penalize models compared to single-spacing SOTA methods.

---

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the Dataset Under: data/manifest-1694710246744/

Run a Visualization:
```bash
python -c "from src.visualization import visualize_single_triplet; visualize_single_triplet(seed=42)"
```

## Citation
If you use this work, please cite:
```bash
Deivanai Thiyagarajan.
Multi-Image Super-Resolution for Medical Slice Interpolation.
University of Florida, 2025.
```
