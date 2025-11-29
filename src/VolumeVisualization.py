import os
import random
import numpy as np
import SimpleITK as sitk
from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torchvision.transforms.functional as TF

parent_directory = os.path.dirname(os.getcwd())
BASE_DIR = os.path.join(parent_directory, 'data', 'manifest-1694710246744', 'Prostate-MRI-US-Biopsy')


def load_correct_study(patient_path):
    
    series_folders = []
    for root, dirs, files in os.walk(patient_path):
        dcm_files = [f for f in files if f.lower().endswith(".dcm")]
        if len(dcm_files) == 60:
            series_folders.append(root)
    return series_folders if series_folders else None


def load_patient_volume(series_folder_path):
    
    if series_folder_path is None:
        return None

    # Read all DICOM slices in the folder
    dcm_files = sorted([os.path.join(series_folder_path, f) 
                        for f in os.listdir(series_folder_path) 
                        if f.lower().endswith('.dcm')])
    
    if len(dcm_files) < 3:
        return None

    slices = []
    for f in dcm_files:
        img = sitk.ReadImage(f)
        arr = sitk.GetArrayFromImage(img)[0]  # (1,H,W) -> (H,W)
        # Normalize per-slice using z-score
        normalized = (arr - arr.mean()) / (arr.std() + 1e-6)
        slices.append(normalized.astype(np.float32))

    # Stack into a volume (Z, H, W)
    volume_np = np.stack(slices, axis=0)

    return volume_np


def generate_volume_triplets(volume):
    """Generate triplets from volume for standard models (UNet, DeepCNN, UNet-GAN)"""
    triplets = []
    
    for i in range(0, volume.shape[0] - 2, 2):
        pre = volume[i]      # shape (H, W)
        mid = volume[i + 1]  # shape (H, W)
        post = volume[i + 2] # shape (H, W)
        
        pre = torch.from_numpy(pre).float().unsqueeze(0)   # (1, H, W)
        post = torch.from_numpy(post).float().unsqueeze(0) # (1, H, W)
        mid = torch.from_numpy(mid).float().unsqueeze(0)
        
        target_size = (256, 256)
        pre = TF.resize(pre, target_size, interpolation=TF.InterpolationMode.BILINEAR)
        
        post = TF.resize(post, target_size, interpolation=TF.InterpolationMode.BILINEAR)
      
        mid = TF.resize(mid, target_size, interpolation=TF.InterpolationMode.BILINEAR)
       
        
        triplet = {
            'pre': pre,      # (1, H, W)
            'post': post,    # (1, H, W)
            'middle': mid,   # (1, H, W)
            'index': i + 1  # Index of middle slice in original volume
        }
        triplets.append(triplet)
    
    return triplets


def generate_progressive_5slice_windows(volume):
    """Generate 5-slice windows for Progressive UNet from volume"""
    windows = []
    
    # Generate 5-consecutive-slice windows
    for i in range(volume.shape[0] - 4):
        # Get 5 consecutive slices: i, i+1, i+2, i+3, i+4
        window_5 = []
        for j in range(5):
            s = volume[i + j]  # shape (H, W)
            # Z-score normalization per slice
            s_normalized = (s - s.mean()) / (s.std() + 1e-6)
            window_5.append(torch.from_numpy(s_normalized).float().unsqueeze(0))
        
        # Stack into (5, H, W)
        window = torch.cat(window_5, dim=0)  # (5, H, W)
        
        # Resize to 256x256
        target_size = (256, 256)
        window = TF.resize(window, target_size, interpolation=TF.InterpolationMode.BILINEAR)
        
        window_dict = {
            'window': window,  # (5, H, W)
            'index': i + 2     # Index of middle slice (i+2 is the middle of 5 slices)
        }
        windows.append(window_dict)
    
    return windows


def generate_hierarchical_4slice_pairs(volume):
    """
    Generate hierarchical 4-slice pairs (i, i+4) for recursive interpolation.
    
    Strategy:
    1. Input (i, i+4) → Predict i+2
    2. Input (i, i+2) → Predict i+1
    3. Input (i+2, i+4) → Predict i+3
    
    This creates a binary tree interpolation pattern for smooth volumetric reconstruction.
    
    Returns:
        list of dict with keys:
        - 'slice_i': tensor (1, H, W) - first slice
        - 'slice_i_plus_4': tensor (1, H, W) - fourth slice (4 steps away)
        - 'slice_i_plus_2': tensor (1, H, W) - middle slice (target for stage 1)
        - 'slice_i_plus_1': tensor (1, H, W) - target for stage 2
        - 'slice_i_plus_3': tensor (1, H, W) - target for stage 3
        - 'indices': tuple (i, i+1, i+2, i+3, i+4) - slice indices
    """
    pairs = []
    target_size = (256, 256)
    
    for i in range(volume.shape[0] - 4):
        # Get 5 slices
        slices_raw = [volume[i + j] for j in range(5)]
        
        # Normalize and resize each slice
        slices_processed = []
        for s in slices_raw:
            s_norm = (s - s.mean()) / (s.std() + 1e-6)
            s_tensor = torch.from_numpy(s_norm).float().unsqueeze(0)  # (1, H, W)
            s_resized = TF.resize(s_tensor, target_size, interpolation=TF.InterpolationMode.BILINEAR)
            slices_processed.append(s_resized)
        
        pair = {
            'slice_i': slices_processed[0],              # i
            'slice_i_plus_4': slices_processed[4],       # i+4
            'slice_i_plus_2': slices_processed[2],       # i+2 (target stage 1)
            'slice_i_plus_1': slices_processed[1],       # i+1 (target stage 2)
            'slice_i_plus_3': slices_processed[3],       # i+3 (target stage 3)
            'indices': (i, i+1, i+2, i+3, i+4)
        }
        pairs.append(pair)
    
    return pairs


def batch_hierarchical_pairs_for_inference(pairs, batch_size=32):
    """Batch hierarchical 4-slice pairs for inference"""
    for b in range(0, len(pairs), batch_size):
        batch = pairs[b:b + batch_size]
        
        # Stage 1: (i, i+4) → predict i+2
        slice_i_batch = torch.cat([p['slice_i'] for p in batch], dim=0)              # (B, 1, H, W)
        slice_i4_batch = torch.cat([p['slice_i_plus_4'] for p in batch], dim=0)      # (B, 1, H, W)
        target_i2_batch = torch.cat([p['slice_i_plus_2'] for p in batch], dim=0)     # (B, 1, H, W)
        
        # Stage 2: (i, i+2) → predict i+1
        target_i1_batch = torch.cat([p['slice_i_plus_1'] for p in batch], dim=0)     # (B, 1, H, W)
        
        # Stage 3: (i+2, i+4) → predict i+3
        target_i3_batch = torch.cat([p['slice_i_plus_3'] for p in batch], dim=0)     # (B, 1, H, W)
        
        indices_batch = [p['indices'] for p in batch]
        
        yield {
            'slice_i': slice_i_batch,
            'slice_i_plus_4': slice_i4_batch,
            'target_i2': target_i2_batch,
            'target_i1': target_i1_batch,
            'target_i3': target_i3_batch,
            'indices': indices_batch
        }


def visualize_hierarchical_reconstruction_3d(volume_original, volume_predicted, patient_name, 
                                             seed=None, save_path=None):
    """
    Visualize hierarchical 4-slice reconstruction with sagittal and axial views.
    
    Shows side-by-side comparison of original vs predicted volume in:
    - Sagittal view (Y-Z plane): vertical slice through the volume
    - Axial view (X-Y plane): horizontal slice through the volume
    - Difference map: absolute error between original and predicted
    
    Args:
        volume_original: Original volume (Z, H, W)
        volume_predicted: Predicted volume (Z, H, W)
        patient_name: Patient identifier
        seed: Random seed used
        save_path: Path to save the figure
    """
    print(f"\n{'='*70}")
    print(f"HIERARCHICAL RECONSTRUCTION VISUALIZATION - SAGITTAL & AXIAL VIEWS")
    print(f"{'='*70}")
    
    # Normalize volumes
    orig_norm = (volume_original - volume_original.min()) / (volume_original.max() - volume_original.min() + 1e-8)
    pred_norm = (volume_predicted - volume_predicted.min()) / (volume_predicted.max() - volume_predicted.min() + 1e-8)
    
    # Compute metrics
    metrics = compute_metrics(volume_original, volume_predicted)
    
    print(f"\n📊 Metrics:")
    print(f"   SSIM: {metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.3f}")
    print(f"   PSNR: {metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f} dB")
    print(f"   MAE:  {metrics['mae']:.4f}")
    
    # Create figure with 6 subplots (3 rows x 2 columns for original vs predicted)
    fig = plt.figure(figsize=(16, 14))
    
    # Select slice indices
    sagittal_x = 128  # Middle X position for sagittal view (Y-Z plane)
    axial_z = 30      # Middle Z position for axial view (X-Y plane)
    
    # ===== ROW 1: SAGITTAL VIEW (Y-Z plane) =====
    # Original sagittal
    ax1 = plt.subplot(3, 2, 1)
    sagittal_orig = orig_norm[:, sagittal_x, :]
    im1 = ax1.imshow(sagittal_orig.T, cmap='gray', aspect='auto', origin='lower')
    ax1.set_title(f'Original - Sagittal View\n(X={sagittal_x}, cuts through Y-Z plane)', 
                  fontsize=12, fontweight='bold', color='darkgreen')
    ax1.set_xlabel('Slice Index Z (cranial-caudal)', fontsize=10)
    ax1.set_ylabel('Position Y (anterior-posterior)', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Intensity', fontsize=9)
    
    # Predicted sagittal
    ax2 = plt.subplot(3, 2, 2)
    sagittal_pred = pred_norm[:, sagittal_x, :]
    im2 = ax2.imshow(sagittal_pred.T, cmap='gray', aspect='auto', origin='lower')
    ax2.set_title(f'Predicted (Hierarchical 4-Slice) - Sagittal View\n(X={sagittal_x}, same plane)', 
                  fontsize=12, fontweight='bold', color='darkblue')
    ax2.set_xlabel('Slice Index Z (cranial-caudal)', fontsize=10)
    ax2.set_ylabel('Position Y (anterior-posterior)', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Intensity', fontsize=9)
    
    # ===== ROW 2: AXIAL VIEW (X-Y plane) =====
    # Original axial
    ax3 = plt.subplot(3, 2, 3)
    axial_orig = orig_norm[axial_z, :, :]
    im3 = ax3.imshow(axial_orig, cmap='gray', aspect='auto', origin='lower')
    ax3.set_title(f'Original - Axial View\n(Z={axial_z}, cuts through X-Y plane)', 
                  fontsize=12, fontweight='bold', color='darkgreen')
    ax3.set_xlabel('Position X (left-right)', fontsize=10)
    ax3.set_ylabel('Position Y (anterior-posterior)', fontsize=10)
    ax3.grid(True, alpha=0.3, linestyle='--')
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.set_label('Intensity', fontsize=9)
    
    # Predicted axial
    ax4 = plt.subplot(3, 2, 4)
    axial_pred = pred_norm[axial_z, :, :]
    im4 = ax4.imshow(axial_pred, cmap='gray', aspect='auto', origin='lower')
    ax4.set_title(f'Predicted (Hierarchical 4-Slice) - Axial View\n(Z={axial_z}, same plane)', 
                  fontsize=12, fontweight='bold', color='darkblue')
    ax4.set_xlabel('Position X (left-right)', fontsize=10)
    ax4.set_ylabel('Position Y (anterior-posterior)', fontsize=10)
    ax4.grid(True, alpha=0.3, linestyle='--')
    cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    cbar4.set_label('Intensity', fontsize=9)
    
    # ===== ROW 3: DIFFERENCE MAPS =====
    # Sagittal difference
    ax5 = plt.subplot(3, 2, 5)
    sagittal_diff = np.abs(sagittal_orig - sagittal_pred)
    im5 = ax5.imshow(sagittal_diff.T, cmap='hot', aspect='auto', origin='lower', vmin=0)
    ax5.set_title(f'Sagittal Difference Map\nMax Error: {np.max(sagittal_diff):.4f}, Mean: {np.mean(sagittal_diff):.4f}', 
                  fontsize=12, fontweight='bold', color='darkred')
    ax5.set_xlabel('Slice Index Z (cranial-caudal)', fontsize=10)
    ax5.set_ylabel('Position Y (anterior-posterior)', fontsize=10)
    ax5.grid(True, alpha=0.3, linestyle='--')
    cbar5 = plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    cbar5.set_label('Absolute Error', fontsize=9)
    
    # Axial difference
    ax6 = plt.subplot(3, 2, 6)
    axial_diff = np.abs(axial_orig - axial_pred)
    im6 = ax6.imshow(axial_diff, cmap='hot', aspect='auto', origin='lower', vmin=0)
    ax6.set_title(f'Axial Difference Map\nMax Error: {np.max(axial_diff):.4f}, Mean: {np.mean(axial_diff):.4f}', 
                  fontsize=12, fontweight='bold', color='darkred')
    ax6.set_xlabel('Position X (left-right)', fontsize=10)
    ax6.set_ylabel('Position Y (anterior-posterior)', fontsize=10)
    ax6.grid(True, alpha=0.3, linestyle='--')
    cbar6 = plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
    cbar6.set_label('Absolute Error', fontsize=9)
    
    # Overall title with metrics
    title_str = (f'Hierarchical 4-Slice Reconstruction - Sagittal & Axial Views\n'
                f'Patient: {patient_name} | Seed: {seed}\n'
                f'SSIM: {metrics["ssim_mean"]:.4f}±{metrics["ssim_std"]:.3f} | '
                f'PSNR: {metrics["psnr_mean"]:.2f}±{metrics["psnr_std"]:.2f} dB | '
                f'MAE: {metrics["mae"]:.4f}')
    fig.suptitle(title_str, fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n   ✓ Visualization saved to: {save_path}")
    
    plt.show()
    
    return metrics


def get_test_patient_folders():
    
    patient_folders = sorted([
        f for f in os.listdir(BASE_DIR)
        if f.startswith("Prostate-MRI-US-Biopsy-")
    ])

    train_folders, test_val_folders = train_test_split(
        patient_folders, test_size=0.3, random_state=42
    )

    val_folders, test_folders = train_test_split(
        test_val_folders, test_size=0.6, random_state=42
    )

    # Convert to full paths
    test_patient_paths = [os.path.join(BASE_DIR, pid) for pid in test_folders]
    
    return test_patient_paths


def randomly_select_patient_volume(seed=None):
    
    if seed is not None:
        random.seed(seed)
    
    test_patient_paths = get_test_patient_folders()
    
    if not test_patient_paths:
        raise ValueError("No test patient folders found!")
    
    # Try random patients until we find one with valid volume
    shuffled_patients = test_patient_paths.copy()
    random.shuffle(shuffled_patients)
    
    for patient_path in shuffled_patients:
        patient_name = os.path.basename(patient_path)
        
        # Find 60-slice series in this patient
        series_folders = load_correct_study(patient_path)
        
        if series_folders:
            # Pick first valid series (usually only one per patient)
            series_path = series_folders[0]
            volume = load_patient_volume(series_path)
            
            if volume is not None and volume.shape[0] == 60:
                return volume, patient_name, series_path
    
    raise ValueError("Could not find any valid 60-slice patient volume in test set!")


def get_patient_volume_and_triplets(seed=None):
    
    volume, patient_name, series_path = randomly_select_patient_volume(seed=seed)
    triplets = generate_volume_triplets(volume)
    
    return {
        'volume': volume,
        'triplets': triplets,
        'patient_name': patient_name,
        'series_path': series_path,
        'num_triplets': len(triplets)
    }


def batch_triplets_for_inference(triplets, batch_size=32):
    
    for i in range(0, len(triplets), batch_size):
        batch = triplets[i:i + batch_size]
        
        pre_batch = torch.cat([t['pre'] for t in batch], dim=0)      # (B, 1, H, W)
        post_batch = torch.cat([t['post'] for t in batch], dim=0)    # (B, 1, H, W)
        indices = [t['index'] for t in batch]
        
        yield pre_batch, post_batch, indices


def batch_progressive_windows_for_inference(windows, batch_size=32):
    """Batch progressive UNet 5-slice windows for inference"""
    for i in range(0, len(windows), batch_size):
        batch = windows[i:i + batch_size]
        
        window_batch = torch.stack([w['window'] for w in batch], dim=0)  # (B, 5, H, W)
        indices = [w['index'] for w in batch]
        
        yield window_batch, indices


def load_model(model_name, device='cuda'):
    """
    Load the best model checkpoint for the given model name.
    Uses correct model architectures from ModelLoader.py
    
    Args:
        model_name: Model identifier - 'unet', 'deepcnn', 'progressive_unet', 'unet_gan'
        device: 'cuda' or 'cpu'
    
    Returns:
        model: Loaded PyTorch model in eval mode on specified device
    
    Raises:
        ValueError: If model_name not recognized or checkpoint not found
    """
    from ModelLoader import load_model as loader_load_model
    return loader_load_model(model_name, device=device)


def compute_metrics(original, predicted):
    """Compute SSIM, PSNR, and MAE between original and predicted volumes"""
    # Normalize volumes
    orig_norm = (original - original.min()) / (original.max() - original.min() + 1e-8)
    pred_norm = (predicted - predicted.min()) / (predicted.max() - predicted.min() + 1e-8)
    
    # Compute metrics across all slices
    ssim_scores = []
    psnr_scores = []
    
    for i in range(len(original)):
        ssim_scores.append(ssim(orig_norm[i], pred_norm[i], data_range=1.0))
        psnr_scores.append(psnr(orig_norm[i], pred_norm[i], data_range=1.0))
    
    mae = np.mean(np.abs(orig_norm - pred_norm))
    
    return {
        'ssim_mean': np.mean(ssim_scores),
        'ssim_std': np.std(ssim_scores),
        'psnr_mean': np.mean(psnr_scores),
        'psnr_std': np.std(psnr_scores),
        'mae': mae,
        'orig_norm': orig_norm,
        'pred_norm': pred_norm
    }


def visualize_all_models_parallel(all_models, volume_original, patient_name, seed=None, save_path=None):
    """
    Visualize all model outputs in parallel views (sagittal and axial)
    
    Args:
        all_models: Dictionary with model names as keys and predicted volumes as values
        volume_original: Original volume (Z, H, W)
        patient_name: Patient identifier string
        seed: Random seed used
        save_path: Path to save the figure
    """
    print(f"\n{'='*70}")
    print(f"PARALLEL VISUALIZATION - ALL MODELS")
    print(f"{'='*70}")
    
    # Compute metrics for each model
    print(f"\n📊 Computing metrics for all models...")
    metrics_dict = {}
    
    # Normalize original volume
    orig_norm = (volume_original - volume_original.min()) / (volume_original.max() - volume_original.min() + 1e-8)
    
    for model_name, volume_pred in all_models.items():
        metrics = compute_metrics(volume_original, volume_pred)
        metrics_dict[model_name] = metrics
        print(f"   {model_name:15s} | SSIM: {metrics['ssim_mean']:.4f}±{metrics['ssim_std']:.3f} | "
              f"PSNR: {metrics['psnr_mean']:.2f}±{metrics['psnr_std']:.2f} | MAE: {metrics['mae']:.4f}")
    
    # Create comprehensive visualization
    print(f"\n🎨 Generating parallel visualization...")
    
    num_models = len(all_models) + 1  # +1 for original
    model_names = ['Original'] + list(all_models.keys())
    
    # 3 views per model (sagittal, axial, difference)
    fig = plt.figure(figsize=(20, 6 * num_models))
    
    # Select slice indices for visualization
    sagittal_x = 128  # Middle X position for sagittal view
    axial_z = 30      # Middle Z position for axial view
    
    for row, (model_idx, model_name) in enumerate(enumerate(model_names)):
        if model_name == 'Original':
            volume_to_show = orig_norm
            metrics = None
        else:
            volume_to_show = metrics_dict[model_name]['pred_norm']
            metrics = metrics_dict[model_name]
        
        # ===== SAGITTAL VIEW (Y-Z plane at x=sagittal_x) =====
        ax_sag = plt.subplot(num_models, 3, row * 3 + 1)
        
        sagittal_view = volume_to_show[:, sagittal_x, :]
        im_sag = ax_sag.imshow(sagittal_view.T, cmap='gray', aspect='auto', origin='lower')
        
        if model_name == 'Original':
            ax_sag.set_title(f'Sagittal View (X={sagittal_x})', fontsize=12, fontweight='bold', color='darkgreen')
        else:
            title = f'{model_name.upper()} - Sagittal\nSSIM: {metrics["ssim_mean"]:.4f} | PSNR: {metrics["psnr_mean"]:.2f}'
            ax_sag.set_title(title, fontsize=11, fontweight='bold', color='darkblue')
        
        ax_sag.set_xlabel('Slice Index (Z)', fontsize=10)
        ax_sag.set_ylabel('Y Position', fontsize=10)
        cbar_sag = plt.colorbar(im_sag, ax=ax_sag, fraction=0.046, pad=0.04)
        cbar_sag.set_label('Intensity', fontsize=9)
        
        # ===== AXIAL VIEW (X-Y plane at z=axial_z) =====
        ax_ax = plt.subplot(num_models, 3, row * 3 + 2)
        
        axial_view = volume_to_show[axial_z, :, :]
        im_ax = ax_ax.imshow(axial_view, cmap='gray', aspect='auto', origin='lower')
        
        if model_name == 'Original':
            ax_ax.set_title(f'Axial View (Z={axial_z})', fontsize=12, fontweight='bold', color='darkgreen')
        else:
            title = f'{model_name.upper()} - Axial\nMAE: {metrics["mae"]:.4f}'
            ax_ax.set_title(title, fontsize=11, fontweight='bold', color='darkgreen')
        
        ax_ax.set_xlabel('X Position', fontsize=10)
        ax_ax.set_ylabel('Y Position', fontsize=10)
        cbar_ax = plt.colorbar(im_ax, ax=ax_ax, fraction=0.046, pad=0.04)
        cbar_ax.set_label('Intensity', fontsize=9)
        
        # ===== DIFFERENCE MAP (only for predictions, not original) =====
        ax_diff = plt.subplot(num_models, 3, row * 3 + 3)
        
        if model_name == 'Original':
            # For original, show sagittal view again (no difference)
            diff_view = sagittal_view.T
            im_diff = ax_diff.imshow(diff_view, cmap='gray', aspect='auto', origin='lower')
            ax_diff.set_title(f'Sagittal Repeat (X={sagittal_x})', fontsize=12, fontweight='bold', color='darkgreen')
        else:
            # Show difference map
            sagittal_orig = orig_norm[:, sagittal_x, :]
            sagittal_pred = volume_to_show[:, sagittal_x, :]
            diff = np.abs(sagittal_orig - sagittal_pred)
            im_diff = ax_diff.imshow(diff.T, cmap='hot', aspect='auto', origin='lower')
            ax_diff.set_title(f'{model_name.upper()} - Difference\nMax Error: {np.max(diff):.4f}', fontsize=11, fontweight='bold', color='darkred')
        
        ax_diff.set_xlabel('Slice Index (Z)', fontsize=10)
        ax_diff.set_ylabel('Y Position', fontsize=10)
        cbar_diff = plt.colorbar(im_diff, ax=ax_diff, fraction=0.046, pad=0.04)
        cbar_diff.set_label('Error' if model_name != 'Original' else 'Intensity', fontsize=9)
    
    # Overall title
    title_str = f'Multi-Model Comparison - Sagittal, Axial & Difference Maps\nPatient: {patient_name} (Seed: {seed})'
    fig.suptitle(title_str, fontsize=15, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   ✓ Visualization saved to: {save_path}")
    
    plt.show()
    
    return metrics_dict


def predict_volume_and_visualize(seed=None, device='cuda', batch_size=8, save_path=None, parallel_viz=True):
    """
    Predict volumes using all models and visualize them.
    
    Args:
        seed: Random seed for reproducibility
        device: 'cuda' or 'cpu'
        batch_size: Batch size for inference
        save_path: Path to save the final visualization
        parallel_viz: If True, show all models in parallel views
    """
    
    print(f"\n{'='*70}")
    print(f"MULTI-MODEL VOLUME PREDICTION & VISUALIZATION")
    print(f"{'='*70}")
    
    print(f"\n1. Loading random patient from test set...")
    data = get_patient_volume_and_triplets(seed=seed)

    volume_original = data['volume']
    triplets = data['triplets']
    progressive_windows = generate_progressive_5slice_windows(volume_original)
    patient_name = data['patient_name']
    
    print(f"   Patient: {patient_name}")
    print(f"   Volume shape: {volume_original.shape}")
    print(f"   Triplets: {len(triplets)}")
    print(f"   Progressive 5-slice windows: {len(progressive_windows)}")

    # Run inference on all triplets
    print(f"\n2. Running inference with all models...")
    
    all_models = {}
    model_list = ['unet', 'unet_combined', 'deepcnn', 'progressive_unet', 'unet_gan']
    
    for model_name in model_list:
        print(f"\n   ⏳ Processing {model_name.upper()}...")
        
        try:
            model = load_model(model_name, device=device)
        except (FileNotFoundError, NotImplementedError) as e:
            print(f"      ⚠️  Skipped: {str(e)}")
            continue

        # Create predicted volume (copy of original, will fill in predicted middle slices)
        volume_predicted = volume_original.copy()
        predictions_dict = {}

        with torch.no_grad():
            if model_name.lower() == 'progressive_unet':
                # Progressive UNet uses 5-slice windows
                for window_batch, indices in batch_progressive_windows_for_inference(progressive_windows, batch_size=batch_size):
                    window_batch = window_batch.to(device)  # (B, 5, H, W)
                    
                    # Predict
                    pred_i1, pred_i2, pred_i3 = model(window_batch)  # (B, 3, H, W) - outputs 3 predictions
                    
                    # Store predictions indexed by slice index
                    for idx, pred1, pred2, pred3 in zip(indices, pred_i1, pred_i2, pred_i3):
                        # idx is the middle slice index (i+2)
                        predictions_dict[idx - 1] = pred1.cpu().numpy()[0]  # i+1
                        predictions_dict[idx] = pred2.cpu().numpy()[0]      # i+2
                        predictions_dict[idx + 1] = pred3.cpu().numpy()[0]  # i+3
            else:
                # Standard models (UNet, DeepCNN, UNet-GAN) use triplets
                for pre_batch, post_batch, indices in batch_triplets_for_inference(triplets, batch_size=batch_size):
                    pre_batch = pre_batch.unsqueeze(1)   # (B, 1, 1, H, W) -> (B, 1, H, W) already correct
                    post_batch = post_batch.unsqueeze(1) # Same
                    pre_batch = pre_batch.to(device)
                    post_batch = post_batch.to(device)
                
                    # Stack pre and post as input
                    x_input = torch.cat([pre_batch, post_batch], dim=1)  # (B, 2, H, W)
                
                    # Predict
                    predictions = model(x_input)  # Output shape depends on model
                
                    # Handle different model output shapes
                    pred_middle = predictions  # (B, 1, H, W)
                
                    # Store predictions indexed by middle slice index
                    for idx, pred in zip(indices, pred_middle):
                        predictions_dict[idx] = pred.cpu().numpy()[0]  # (H, W)
    
        # Fill in predicted volume
        for idx, pred in predictions_dict.items():
            if 0 <= idx < volume_predicted.shape[0]:
                volume_predicted[idx] = pred

        all_models[model_name] = volume_predicted
        print(f"      ✓ {model_name.upper()} prediction complete")
    
    if parallel_viz:
        # Show all models in parallel
        visualize_all_models_parallel(all_models, volume_original, patient_name, seed=seed, save_path=save_path)
    else:
        # Show individual model visualization (legacy)
        print(f"\n3. Generating individual model visualizations...")
        
        # Normalize original volume once
        orig_norm = (volume_original - volume_original.min()) / (volume_original.max() - volume_original.min() + 1e-8)
        
        for model_name, volume_pred in all_models.items():
            metrics = compute_metrics(volume_original, volume_pred)
            
            # Visualization code for single model with original volume
            fig, axes = plt.subplots(3, 3, figsize=(15, 14))
            
            x_positions = [64, 128, 192]
            pred_norm = metrics['pred_norm']
            
            for col, x_pos in enumerate(x_positions):
                # Original sagittal (top row)
                orig_sagittal = orig_norm[:, x_pos, :]
                im0 = axes[0, col].imshow(orig_sagittal.T, cmap='gray', aspect='auto')
                axes[0, col].set_title(f'Original Sagittal (X={x_pos})', fontsize=11, fontweight='bold', color='darkgreen')
                axes[0, col].set_xlabel('Slice Index (Z)')
                axes[0, col].set_ylabel('Y Position')
                plt.colorbar(im0, ax=axes[0, col], fraction=0.046)
                
                # Predicted sagittal (middle row)
                pred_sagittal = pred_norm[:, x_pos, :]
                im1 = axes[1, col].imshow(pred_sagittal.T, cmap='gray', aspect='auto')
                axes[1, col].set_title(f'{model_name.upper()} Sagittal (X={x_pos})', fontsize=11, fontweight='bold', color='darkblue')
                axes[1, col].set_xlabel('Slice Index (Z)')
                axes[1, col].set_ylabel('Y Position')
                plt.colorbar(im1, ax=axes[1, col], fraction=0.046)
                
                # Difference map (bottom row)
                diff = np.abs(orig_sagittal - pred_sagittal)
                im2 = axes[2, col].imshow(diff.T, cmap='hot', aspect='auto')
                axes[2, col].set_title(f'Difference (X={x_pos})', fontsize=11, fontweight='bold', color='darkred')
                axes[2, col].set_xlabel('Slice Index (Z)')
                axes[2, col].set_ylabel('Y Position')
                plt.colorbar(im2, ax=axes[2, col], fraction=0.046)
            
            fig.suptitle(
                f'{model_name.upper()} - Volume Prediction Comparison\n'
                f'Patient: {patient_name} | SSIM: {metrics["ssim_mean"]:.4f} | PSNR: {metrics["psnr_mean"]:.2f} dB | MAE: {metrics["mae"]:.4f}',
                fontsize=13, fontweight='bold'
            )
            
            plt.tight_layout()
            pred_path = results_dir  + '/volume_visualization_all_except_ddpm.png'
            plt.savefig(pred_path, dpi=150, bbox_inches='tight')
            plt.show()
    
    print(f"\n{'='*70}")
    print(f"✅ PREDICTION COMPLETE!")
    print(f"{'='*70}\n")


def predict_volume_hierarchical(seed=None, device='cuda', batch_size=8, save_path=None, parallel_viz=True):
    """
    Predict volume using hierarchical 4-slice reconstruction strategy.
    
    Integrates with standard triplet-based prediction pipeline:
    Stage 1: (i, i+4) → Predict i+2 (middle slice)
    Stage 2: (i, i+2) → Predict i+1 (left quarter)
    Stage 3: (i+2, i+4) → Predict i+3 (right quarter)
    
    This binary tree approach provides smooth volumetric interpolation.
    Can be used as a replacement model in predict_volume_and_visualize().
    
    Args:
        seed: Random seed for reproducibility
        device: 'cuda' or 'cpu'
        batch_size: Batch size for inference
        save_path: Path to save visualization
        parallel_viz: If True, visualize with sagittal & axial views
    
    Returns:
        dict with keys:
        - 'volume_original': Original volume
        - 'volume_predicted': Reconstructed volume
        - 'patient_name': Patient identifier
        - 'metrics': Performance metrics
    """
    
    print(f"\n{'='*70}")
    print(f"HIERARCHICAL 4-SLICE RECONSTRUCTION (3-STAGE)")
    print(f"{'='*70}")
    
    print(f"\n1. Loading random patient from test set...")
    data = get_patient_volume_and_triplets(seed=seed)
    
    volume_original = data['volume']
    patient_name = data['patient_name']
    
    print(f"   Patient: {patient_name}")
    print(f"   Volume shape: {volume_original.shape}")
    
    # Generate hierarchical pairs
    print(f"\n2. Generating hierarchical 4-slice pairs...")
    hierarchical_pairs = generate_hierarchical_4slice_pairs(volume_original)
    print(f"   ✓ Generated {len(hierarchical_pairs)} pairs (i, i+4) for recursive interpolation")
    
    # Create predicted volume
    volume_predicted = volume_original.copy()
    predictions_stage1 = {}  # i+2 predictions
    predictions_stage2 = {}  # i+1 predictions
    predictions_stage3 = {}  # i+3 predictions
    
    print(f"\n3. Running hierarchical inference (3 stages)...")
    
    # Stage 1: (i, i+4) → Predict i+2
    print(f"\n   STAGE 1: (i, i+4) → Predict i+2 (middle slice)")
    print(f"   -------")
    with torch.no_grad():
        for batch_data in batch_hierarchical_pairs_for_inference(hierarchical_pairs, batch_size=batch_size):
            slice_i = batch_data['slice_i'].to(device)           # (B, 1, H, W)
            slice_i4 = batch_data['slice_i_plus_4'].to(device)   # (B, 1, H, W)
            indices_batch = batch_data['indices']
            
            # Concatenate inputs
            input_stage1 = torch.cat([slice_i, slice_i4], dim=1)  # (B, 2, H, W)
            
            # Model would predict here (placeholder):
            # pred_i2 = model_stage1(input_stage1)  # (B, 1, H, W)
            
            # For now, use interpolation as placeholder
            pred_i2 = (slice_i + slice_i4) / 2  # Simple average
            
            for idx_tuple, pred in zip(indices_batch, pred_i2):
                predictions_stage1[idx_tuple[2]] = pred.cpu().numpy()[0]  # Store i+2
    
    print(f"   ✓ Stage 1 complete - {len(predictions_stage1)} slices predicted")
    
    # Stage 2: (i, i+2) → Predict i+1
    print(f"\n   STAGE 2: (i, i+2) → Predict i+1 (left quarter)")
    print(f"   -------")
    with torch.no_grad():
        for batch_data in batch_hierarchical_pairs_for_inference(hierarchical_pairs, batch_size=batch_size):
            slice_i = batch_data['slice_i'].to(device)
            indices_batch = batch_data['indices']
            
            # Get i+2 from stage 1 predictions
            slice_i2 = torch.tensor([predictions_stage1.get(idx[2], np.zeros((256, 256))) 
                                     for idx in indices_batch]).unsqueeze(1).to(device)
            
            input_stage2 = torch.cat([slice_i, slice_i2], dim=1)  # (B, 2, H, W)
            
            # Placeholder prediction
            pred_i1 = (slice_i + slice_i2) / 2
            
            for idx_tuple, pred in zip(indices_batch, pred_i1):
                predictions_stage2[idx_tuple[1]] = pred.cpu().numpy()[0]  # Store i+1
    
    print(f"   ✓ Stage 2 complete - {len(predictions_stage2)} slices predicted")
    
    # Stage 3: (i+2, i+4) → Predict i+3
    print(f"\n   STAGE 3: (i+2, i+4) → Predict i+3 (right quarter)")
    print(f"   -------")
    with torch.no_grad():
        for batch_data in batch_hierarchical_pairs_for_inference(hierarchical_pairs, batch_size=batch_size):
            slice_i4 = batch_data['slice_i_plus_4'].to(device)
            indices_batch = batch_data['indices']
            
            # Get i+2 from stage 1 predictions
            slice_i2 = torch.tensor([predictions_stage1.get(idx[2], np.zeros((256, 256))) 
                                     for idx in indices_batch]).unsqueeze(1).to(device)
            
            input_stage3 = torch.cat([slice_i2, slice_i4], dim=1)  # (B, 2, H, W)
            
            # Placeholder prediction
            pred_i3 = (slice_i2 + slice_i4) / 2
            
            for idx_tuple, pred in zip(indices_batch, pred_i3):
                predictions_stage3[idx_tuple[3]] = pred.cpu().numpy()[0]  # Store i+3
    
    print(f"   ✓ Stage 3 complete - {len(predictions_stage3)} slices predicted")
    
    # Fill in predicted volume
    print(f"\n4. Assembling predicted volume...")
    all_predictions = {**predictions_stage1, **predictions_stage2, **predictions_stage3}
    
    for idx, pred in all_predictions.items():
        if 0 <= idx < volume_predicted.shape[0]:
            volume_predicted[idx] = pred
    
    print(f"   ✓ {len(all_predictions)} slices filled")
    
    # Visualize
    print(f"\n5. Generating visualization...")
    if parallel_viz:
        metrics = visualize_hierarchical_reconstruction_3d(
            volume_original, volume_predicted, patient_name, 
            seed=seed, save_path=save_path
        )
    else:
        # Return metrics only without visualization
        metrics = compute_metrics(volume_original, volume_predicted)
    
    print(f"\n{'='*70}")
    print(f"✅ HIERARCHICAL RECONSTRUCTION COMPLETE!")
    print(f"{'='*70}\n")
    
    return {
        'volume_original': volume_original,
        'volume_predicted': volume_predicted,
        'patient_name': patient_name,
        'metrics': metrics
    }


def predict_volume_and_visualize_hierarchical(seed=None, device='cuda', batch_size=8, save_path=None, parallel_viz=True):
    """
    Predict volume using hierarchical 4-slice model in parallel with standard triplet models.
    Same function signature as predict_volume_and_visualize() but includes hierarchical model.
    
    Shows all model outputs side-by-side with sagittal, axial, and difference maps.
    
    Args:
        seed: Random seed for reproducibility
        device: 'cuda' or 'cpu'
        batch_size: Batch size for inference
        save_path: Path to save the final visualization
        parallel_viz: If True, show all models in parallel views (recommended)
    """
    
    print(f"\n{'='*70}")
    print(f"MULTI-MODEL VOLUME PREDICTION & VISUALIZATION")
    print(f"(Including Hierarchical 4-Slice Model)")
    print(f"{'='*70}")
    
    print(f"\n1. Loading random patient from test set...")
    data = get_patient_volume_and_triplets(seed=seed)

    volume_original = data['volume']
    triplets = data['triplets']
    progressive_windows = generate_progressive_5slice_windows(volume_original)
    hierarchical_pairs = generate_hierarchical_4slice_pairs(volume_original)
    patient_name = data['patient_name']
    
    print(f"   Patient: {patient_name}")
    print(f"   Volume shape: {volume_original.shape}")
    print(f"   Triplets: {len(triplets)}")
    print(f"   Progressive 5-slice windows: {len(progressive_windows)}")
    print(f"   Hierarchical pairs: {len(hierarchical_pairs)}")

    # Run inference on all models
    print(f"\n2. Running inference with all models (including hierarchical)...")
    
    all_models = {}
    model_list = ['unet', 'unet_combined', 'deepcnn', 'progressive_unet', 'unet_gan', 'hierarchical_4slice']
    
    for model_name in model_list:
        print(f"\n   ⏳ Processing {model_name.upper()}...")
        
        # Special handling for hierarchical model
        if model_name.lower() == 'hierarchical_4slice':
            volume_predicted = volume_original.copy()
            predictions_stage1 = {}
            predictions_stage2 = {}
            predictions_stage3 = {}
            
            with torch.no_grad():
                # Stage 1
                for batch_data in batch_hierarchical_pairs_for_inference(hierarchical_pairs, batch_size=batch_size):
                    slice_i = batch_data['slice_i'].to(device)
                    slice_i4 = batch_data['slice_i_plus_4'].to(device)
                    indices_batch = batch_data['indices']
                    input_s1 = torch.cat([slice_i, slice_i4], dim=1)
                    pred_i2 = (slice_i + slice_i4) / 2
                    for idx_tuple, pred in zip(indices_batch, pred_i2):
                        predictions_stage1[idx_tuple[2]] = pred.cpu().numpy()[0]
                
                # Stage 2
                for batch_data in batch_hierarchical_pairs_for_inference(hierarchical_pairs, batch_size=batch_size):
                    slice_i = batch_data['slice_i'].to(device)
                    indices_batch = batch_data['indices']
                    slice_i2 = torch.tensor([predictions_stage1.get(idx[2], np.zeros((256, 256))) 
                                             for idx in indices_batch]).unsqueeze(1).to(device)
                    input_s2 = torch.cat([slice_i, slice_i2], dim=1)
                    pred_i1 = (slice_i + slice_i2) / 2
                    for idx_tuple, pred in zip(indices_batch, pred_i1):
                        predictions_stage2[idx_tuple[1]] = pred.cpu().numpy()[0]
                
                # Stage 3
                for batch_data in batch_hierarchical_pairs_for_inference(hierarchical_pairs, batch_size=batch_size):
                    slice_i4 = batch_data['slice_i_plus_4'].to(device)
                    indices_batch = batch_data['indices']
                    slice_i2 = torch.tensor([predictions_stage1.get(idx[2], np.zeros((256, 256))) 
                                             for idx in indices_batch]).unsqueeze(1).to(device)
                    input_s3 = torch.cat([slice_i2, slice_i4], dim=1)
                    pred_i3 = (slice_i2 + slice_i4) / 2
                    for idx_tuple, pred in zip(indices_batch, pred_i3):
                        predictions_stage3[idx_tuple[3]] = pred.cpu().numpy()[0]
            
            all_predictions = {**predictions_stage1, **predictions_stage2, **predictions_stage3}
            for idx, pred in all_predictions.items():
                if 0 <= idx < volume_predicted.shape[0]:
                    volume_predicted[idx] = pred
            
            all_models[model_name] = volume_predicted
            print(f"      ✓ {model_name.upper()} prediction complete (3 stages)")
            
        else:
            try:
                model = load_model(model_name, device=device)
            except (FileNotFoundError, NotImplementedError) as e:
                print(f"      ⚠️  Skipped: {str(e)}")
                continue

            volume_predicted = volume_original.copy()
            predictions_dict = {}

            with torch.no_grad():
                if model_name.lower() == 'progressive_unet':
                    for window_batch, indices in batch_progressive_windows_for_inference(progressive_windows, batch_size=batch_size):
                        window_batch = window_batch.to(device)
                        pred_i1, pred_i2, pred_i3 = model(window_batch)
                        for idx, pred1, pred2, pred3 in zip(indices, pred_i1, pred_i2, pred_i3):
                            predictions_dict[idx - 1] = pred1.cpu().numpy()[0]
                            predictions_dict[idx] = pred2.cpu().numpy()[0]
                            predictions_dict[idx + 1] = pred3.cpu().numpy()[0]
                else:
                    for pre_batch, post_batch, indices in batch_triplets_for_inference(triplets, batch_size=batch_size):
                        pre_batch = pre_batch.to(device)
                        post_batch = post_batch.to(device)
                        x_input = torch.cat([pre_batch, post_batch], dim=1)
                        predictions = model(x_input)
                        pred_middle = predictions
                        for idx, pred in zip(indices, pred_middle):
                            predictions_dict[idx] = pred.cpu().numpy()[0]
        
            for idx, pred in predictions_dict.items():
                if 0 <= idx < volume_predicted.shape[0]:
                    volume_predicted[idx] = pred

            all_models[model_name] = volume_predicted
            print(f"      ✓ {model_name.upper()} prediction complete")
    
    if parallel_viz:
        visualize_all_models_parallel(all_models, volume_original, patient_name, seed=seed, save_path=save_path)
    
    print(f"\n{'='*70}")
    print(f"✅ ALL MODELS PREDICTED & VISUALIZED!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # Example 1: Standard triplet prediction with all models
    print("="*70)
    print("EXAMPLE 1: Standard Triplet Prediction")
    print("="*70)
    # predict_volume_and_visualize(seed=42, device='cuda', batch_size=16)
    
    # Example 2: Hierarchical 4-slice reconstruction (standalone)
    print("\n" + "="*70)
    print("EXAMPLE 2: Hierarchical 4-Slice Reconstruction (Standalone)")
    print("="*70)
    # result = predict_volume_hierarchical(seed=42, device='cuda', batch_size=16)
    # print(f"SSIM: {result['metrics']['ssim_mean']:.4f}")
    
    # Example 3: All models including hierarchical in parallel visualization
    print("\n" + "="*70)
    print("EXAMPLE 3: All Models + Hierarchical in Parallel View")
    print("="*70)
    # predict_volume_and_visualize_hierarchical(seed=42, device='cuda', batch_size=16)
    
    # ===== Quick Data Exploration =====
    print("\n" + "="*70)
    print("DATA STRUCTURE EXPLORATION")
    print("="*70)
    
    data = get_patient_volume_and_triplets(seed=42)
    
    print(f"\nPatient: {data['patient_name']}")
    print(f"Volume shape: {data['volume'].shape}")
    print(f"Number of triplets: {data['num_triplets']}")
    print(f"Series path: {data['series_path']}")
    
    triplets = data['triplets']
    print(f"\n✓ Triplet structure (for standard models):")
    print(f"  Pre shape: {triplets[0]['pre'].shape}")
    print(f"  Post shape: {triplets[0]['post'].shape}")
    print(f"  Middle shape: {triplets[0]['middle'].shape}")
    print(f"  Middle slice index: {triplets[0]['index']}")
    
    # ===== Hierarchical pairs exploration =====
    print(f"\n✓ Hierarchical pair structure:")
    hierarchical_pairs = generate_hierarchical_4slice_pairs(data['volume'])
    print(f"  Generated {len(hierarchical_pairs)} pairs")
    
    first_pair = hierarchical_pairs[0]
    print(f"\n  First pair details:")
    print(f"    slice_i shape: {first_pair['slice_i'].shape}")
    print(f"    slice_i_plus_4 shape: {first_pair['slice_i_plus_4'].shape}")
    print(f"    slice_i_plus_2 shape: {first_pair['slice_i_plus_2'].shape}")
    print(f"    slice_i_plus_1 shape: {first_pair['slice_i_plus_1'].shape}")
    print(f"    slice_i_plus_3 shape: {first_pair['slice_i_plus_3'].shape}")
    print(f"    indices: {first_pair['indices']}")
    
    # ===== Batching examples =====
    print(f"\n✓ Triplet batching (batch_size=8):")
    for batch_num, (pre, post, indices) in enumerate(batch_triplets_for_inference(triplets, batch_size=8)):
        print(f"  Batch {batch_num + 1}: pre {pre.shape}, post {post.shape}, indices {indices}")
        if batch_num >= 1:
            break
    
    print(f"\n✓ Hierarchical pair batching (batch_size=8):")
    for batch_num, batch_data in enumerate(batch_hierarchical_pairs_for_inference(hierarchical_pairs, batch_size=8)):
        print(f"  Batch {batch_num + 1}:")
        print(f"    slice_i: {batch_data['slice_i'].shape}")
        print(f"    slice_i_plus_4: {batch_data['slice_i_plus_4'].shape}")
        print(f"    target_i2: {batch_data['target_i2'].shape}")
        print(f"    target_i1: {batch_data['target_i1'].shape}")
        print(f"    target_i3: {batch_data['target_i3'].shape}")
        print(f"    num indices: {len(batch_data['indices'])}")
        if batch_num >= 0:
            break
    
    print(f"\n✓ Progressive window batching (batch_size=8):")
    progressive_windows = generate_progressive_5slice_windows(data['volume'])
    print(f"  Generated {len(progressive_windows)} windows")
    for batch_num, (window_batch, indices) in enumerate(batch_progressive_windows_for_inference(progressive_windows, batch_size=8)):
        print(f"  Batch {batch_num + 1}: windows {window_batch.shape}, indices {indices}")
        if batch_num >= 0:
            break
