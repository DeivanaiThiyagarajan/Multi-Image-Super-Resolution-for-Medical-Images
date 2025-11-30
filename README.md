# Multi-Image-Super-Resolution

This repository contains code and notebooks for multi-image medical image "super-resolution": predicting a middle MRI slice from two neighbouring slices. The project compares a set of models (UNet, DeepCNN, UNet-GAN, Progressive UNet and variations) on a prostate MRI dataset and provides visualizations and quantitative metrics (SSIM, PSNR, MAE) to compare reconstruction quality.

Key ideas:
- Predict the middle slice from two neighboring slices (triplet inputs). The code supports two triplet spacings: slices 1.5mm apart (i, i+2 → i+1) and 3.0mm apart (i, i+4 → i+2). Combining these configurations affects reported metrics like SSIM because larger inter-slice spacing is more challenging.
- Progressive UNet: a 3-stage approach which first predicts a coarse middle slice from (i, i+4) and then refines adjacent slices using the predicted middle. This hierarchical approach lets the model learn coarse-to-fine interpolation across larger gaps.

What this project contains
- `src/` : model definitions, data generators and visualization utilities.
- `notebooks/` : training and analysis notebooks (training history, metrics, visualizations).
- `data/` : dataset manifest and DICOM files (not included here for privacy).
- `models/` : place model checkpoints here (expected names: `unet_best.pt`, `deepcnn_best.pt`, `progressive_unet_best.pt`, `unet_gan_best.pt`, etc.).

Why metrics may be lower than SOTA on combined evaluation
- This project reports SSIM/PSNR across multiple triplet spacings. Predicting middle slices spaced 3mm apart (i & i+4 → i+2) is intrinsically harder than 1.5mm spacing (i & i+2 → i+1). When aggregated, mean SSIM can be lower because some test samples include the more difficult 3mm cases. We include per-model and per-case metrics in notebooks so that you can separate these effects.

Progressive UNet (brief)
- Stage 1: UNet1(i, i+4) → predict i+2 (coarse middle)
- Stage 2A: UNet2(i, i+2_pred) → predict i+1 (left quarter)
- Stage 2B: UNet3(i+2_pred, i+4) → predict i+3 (right quarter)

This enables reconstructing slices across larger gaps by first predicting the coarse intermediate and then predicting the neighbors conditioned on it.

Quickstart
1. Create and activate a Python environment (recommended):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Place the `Prostate-MRI-US-Biopsy` dataset folder under `data/manifest-1694710246744/` as expected by the data loaders.

4. Put trained model checkpoints in `models/` with expected names (see `src/ModelLoader.py`).

5. Run analysis notebooks in `notebooks/` or use the `src/VolumeVisualization.py` helpers to reproduce visualizations and metrics. Examples:

```powershell
# Run a single triplet visualization (from Python REPL)
python -c "from src.VolumeVisualization import visualize_single_triplet_all_models; visualize_single_triplet_all_models(seed=42, device='cpu')"

# Run full prediction & parallel visualization
python -c "from src.VolumeVisualization import predict_volume_and_visualize; predict_volume_and_visualize(seed=42, device='cpu', batch_size=8)"
```

Metrics and analysis
- Notebooks report: SSIM, PSNR, MAE, and training loss curves. Use the notebooks to inspect per-model distributions and to separate triplet types (1.5mm vs 3.0mm) when comparing performance.
- If SSIM looks lower than expected, filter results by triplet spacing in the notebooks to see per-case performance.

Contact / next steps
- If you want, I can:
	- Add a small script to split reported metrics by triplet spacing (1.5mm vs 3.0mm).
	- Add unit tests or a small demo runner that loads a single patient and saves example visualizations to `results/`.

---
Author: Deivanai Thiyagarajan

## Models

This project implements and compares several models for middle-slice prediction. Below is a concise description for each model and why you might pick it.

### UNet
- Architecture: standard encoder-decoder with skip connections (double conv blocks + transpose upsampling). Inputs are two slices stacked as channels (shape: `(B, 2, H, W)`) and output is a single predicted middle slice (shape: `(B, 1, H, W)`).
- Strengths: strong baseline for dense pixel-wise regression, stable training, relatively lightweight when scaled down.
- Use when: you want a straightforward end-to-end interpolation baseline with good locality and skip connections to preserve spatial detail.

### DeepCNN (ResNet-like)
- Architecture: deep residual style network using residual blocks (convolution + batchnorm + ReLU) and progressive feature expansions. Input format is the same as UNet (`(B,2,H,W)`).
- Strengths: good at learning hierarchical features and global context via deeper receptive fields; residual connections ease training in deep networks.
- Use when: you expect global context across slices to help reconstruction and want a model less focused on strict encoder-decoder skip pathways.

### UNet-GAN (Generator)
- Architecture: UNet-like generator used inside a GAN training framework. The generator structure mirrors UNet but is optimized together with a discriminator (not included here) to encourage realistic textures and sharper reconstructions.
- Strengths: can produce visually sharper outputs and recover finer textures compared to pure MSE-trained models, but may introduce hallucinated details and is harder to stabilize.
- Use when: visual fidelity (sharpness) is critical and you can tolerate GAN training complexity.

### FastDDPM (Fast Denoising Diffusion Probabilistic Models)
- Architecture: UNet-style generator conditioned on diffusion timestep embeddings. Time embeddings are learned via a TimeEmbedding layer and injected into each residual block via adaptive scaling.
- How it works:
  1. **Forward diffusion (training only):** Start with clean image, add Gaussian noise over T timesteps (typically 1000).
  2. **Reverse diffusion (inference):** Learn to iteratively denoise a noisy image back to the clean prediction. Key innovation: use only ~10-100 carefully selected timesteps instead of all 1000 (called "accelerated sampling").
  3. **Scheduling:** Two strategies available:
     - Uniform: select timesteps evenly spaced (t=0, 100, 200, ..., 1000).
     - Non-uniform: emphasize earlier timesteps where more structure is being recovered (t ~ t^1.1 schedule).
- Strengths:
  - **Probabilistic approach:** Can model data distribution and generate multiple plausible predictions (useful for high-uncertainty regions).
  - **Fast inference:** 100x faster than standard DDPM (10 steps vs 1000) with comparable quality.
  - **Robustness:** Can learn richer error distributions and generate sharp, realistic outputs.
- Weaknesses:
  - More complex training loop and hyperparameter tuning (scheduler choice, noise schedule, training steps).
  - Slower than deterministic models like UNet per forward pass (though fewer steps needed).
  - Can sometimes introduce subtle hallucinations if not carefully tuned.
- Use when: you want probabilistic predictions with uncertainty estimates, or when visual quality and sharpness are critical.

### Progressive UNet (detailed)
The Progressive UNet is the unique component in this repository and deserves a deeper explanation.

- Motivation: predicting slices with large z-spacing (e.g., 3.0mm between input slices) is more challenging because anatomical structures can change significantly across the gap. A single-stage model predicts the middle directly and can struggle when long-range interpolation is needed.

- High-level idea: break the hard interpolation into simpler sub-problems and solve them sequentially in stages (coarse-to-fine). This reduces the effective interpolation distance each sub-model has to handle.

- Architecture & stages:
	1. **UNet1 (Stage 1)** — Input: `(i, i+4)` → Output: predicted `i+2` (coarse middle). This stage learns to produce a reasonable central estimate across a large gap.
	2. **UNet2 (Stage 2A)** — Input: `(i, predicted_i+2)` → Output: predicted `i+1` (left neighbor). By conditioning on the predicted coarse middle, UNet2 solves a smaller interpolation gap.
	3. **UNet3 (Stage 2B)** — Input: `(predicted_i+2, i+4)` → Output: predicted `i+3` (right neighbor). Similar to UNet2 but for the other side.

- Training strategy:
	- Stage 1 is trained to minimize a reconstruction loss (MSE) between predicted `i+2` and the ground truth `i+2` using windows of 5 consecutive slices. Data augmentation and per-slice normalization are typically applied.
	- Stage 2A and 2B can be trained **conditionally**: using either ground-truth `i+2` (teacher forcing) or the Stage 1 predictions (to make the pipeline robust to Stage 1 errors). A typical regimen is to start with teacher forcing and progressively switch to predicted middle inputs (scheduled sampling) to avoid error accumulation at inference.

- Inference pipeline:
	1. Use UNet1 on `(i, i+4)` to obtain `pred_i+2`.
	2. Use UNet2 on `(i, pred_i+2)` to obtain `pred_i+1`.
	3. Use UNet3 on `(pred_i+2, i+4)` to obtain `pred_i+3`.
	4. Assemble the predicted slices back into the volume.

- Advantages:
	- Breaks down a hard interpolation into easier tasks, improving robustness on large spacing.
	- Enables reuse of the same UNet-style blocks but specialized for each sub-problem.
	- Can produce better global consistency because Stage 1 encourages a coherent coarse middle that conditions the others.

- Challenges & caveats:
	- Error propagation: mistakes from Stage 1 affect Stage 2. Mitigation strategies include scheduled sampling or training Stage 2 with both ground truth and predicted middles.
	- Slightly more complex training and inference logic compared to single-stage models.
	- Requires more memory/time if all three UNet stages are large.

### When to use Progressive UNet vs single-stage
- Use Progressive UNet when inter-slice spacing is large (e.g., 3.0mm) or structural changes between input slices are significant. For very small spacing where interpolation is easy, a single UNet may be sufficient and cheaper.

## Training & Losses (notes)
- Common losses: MSE (L2) for stable reconstructions; MAE (L1) can be used if robustness to outliers is desired.
- GAN setups: adversarial loss + perceptual losses (if used) can boost apparent sharpness but need careful balancing.
- Validation: track `val_loss`, and compute SSIM/PSNR per-epoch to monitor perceptual quality.

## Evaluation tips
- Report per-spacing metrics: separate results for (i,i+2→i+1) and (i,i+4→i+2) so readers can see the spacing effect.
- Report distribution (boxplots) of per-slice SSIM and PSNR and mean±std.
- Visualize sample triplets (pre/post/GT/pred) and difference maps — both axial and sagittal views help understand where models fail.

## Results Summary

Below are the key results from training all models on the Prostate MRI dataset. All models were trained on triplet and 5-slice window data split at the **patient level** (70% train, 15% val, 15% test) to avoid data leakage.

### Quantitative Results (Test Set)

| Model | SSIM Mean | SSIM Std | PSNR Mean (dB) | PSNR Std | Test Loss | Epochs |
|-------|-----------|----------|----------------|----------|-----------|--------|
| **UNet** | 0.711 | 0.129 | 23.61 | 4.11 | 0.0713 | 15 |
| **DeepCNN** | 0.710 | 0.129 | 23.61 | 4.11 | 0.0860 | 19 |
| **UNet-GAN** | 0.760 | 0.140 | 28.57 | 4.38 | 0.0715 | 20 |
| **Progressive UNet** | **0.724** (avg) | — | **26.97** (avg) | — | — | 27 |
| **FastDDPM** | — (trained) | — | — | — | — | — |

**Key observations:**

1. **UNet-GAN** achieves the highest PSNR (28.57 dB) and highest SSIM (0.760), indicating sharper, more realistic outputs. This is expected due to adversarial training encouraging visually appealing results.

2. **Progressive UNet** shows balanced performance across stages:
   - Stage 1 (i+2 prediction from i & i+4 — large gap): SSIM=0.635, PSNR=24.99 (hardest task)
   - Stage 2A (i+1 prediction from i & pred_i+2 — small gap): SSIM=0.768, PSNR=27.91 (refined by coarse middle)
   - Stage 2B (i+3 prediction from pred_i+2 & i+4 — small gap): SSIM=0.770, PSNR=28.01 (refined by coarse middle)
   - Average across stages: SSIM=0.724, PSNR=26.97
   - The hierarchical approach successfully decomposes the hard interpolation into easier sub-tasks, showing clear improvement in stages 2A/2B.

3. **UNet & DeepCNN** achieve similar baseline performance (~0.71 SSIM, 23.6 PSNR) when trained with MSE loss. DeepCNN's residual architecture provides no clear advantage on this task compared to standard UNet, suggesting that local encoder-decoder pathways are more important than deep residual connections for this dense prediction task.

4. **Spacing effect:** The lower SSIM on stage 1 (0.635 for i+2) vs stages 2A/2B (0.768/0.770) demonstrates that predicting across a large 3mm gap (i & i+4 → i+2) is intrinsically harder than small 1.5mm gaps. This explains why aggregated metrics can appear lower when mixing both spacings.

### Training Dynamics

- **UNet:** Fast convergence (15 epochs), stable learning curve, early stopping triggered by patience.
- **DeepCNN:** Slightly slower convergence (19 epochs), train loss continues declining, did not improve SSIM/PSNR over UNet.
- **UNet-GAN:** Stable GAN training (20 epochs), adversarial loss gradually decreased, indicating good generator-discriminator balance. Perceptual and L1 losses smoothly decreased.
- **Progressive UNet:** Longer training (27 epochs) due to 3-stage pipeline and weighted multi-task loss (w_i1=0.5, w_i2=1.0, w_i3=0.5). Train loss consistently decreased across all stages.

### Visualizations

Training curves and predictions are saved in `results/`:
- `training_curves.png` — UNet training loss curves (train & val).
- `deepcnn_training_curves.png`, `unet_gan_training_curves.png`, `progressive_unet_training_curves.png` — Model-specific curves.
- `unet_gan_predictions.png`, `predictions_visualization_progressive_unet_*.png` — Sample predictions and ground truth comparisons showing per-slice quality.
- `volume_visualization_hierarchical.png` — Full volume predictions using hierarchical reconstruction across all stages.

### Interpretation & Practical Guidance

- **For baseline usage:** Use **UNet** if you need a lightweight, stable baseline with minimal hyperparameter tuning. Use **UNet-GAN** if visual sharpness is important and you have compute/patience for GAN training.
  
- **For large spacing (3mm+):** **Progressive UNet** is specifically designed for this. The hierarchical decomposition (coarse → fine) reduces the interpolation distance each sub-model must handle, leading to better per-slice quality in stages 2A/2B.

- **For probabilistic predictions & uncertainty:** **FastDDPM** offers a principled way to model the data distribution and generate multiple plausible completions. This is useful when you need uncertainty estimates or want to ensemble predictions.

- **Combined evaluation caveats:** When reporting results for publication or comparison, **always separate metrics by spacing** (1.5mm vs 3.0mm) to fairly compare against SOTA methods that may focus on one spacing only. Our aggregated SSIM (~0.72) is lower than single-spacing methods (which may report 0.75-0.80) because we evaluate both difficult (3mm) and easy (1.5mm) cases together.