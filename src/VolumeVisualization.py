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
        
        # Concatenate slices: each slice is (1, H, W), so cat on dim=0 gives (N, H, W)
        # Then unsqueeze to get (N, 1, H, W) for proper batch format
        pre_list = [t['pre'] for t in batch]  # List of (1, H, W)
        post_list = [t['post'] for t in batch]  # List of (1, H, W)
        
        pre_batch = torch.cat(pre_list, dim=0)  # (B, H, W)
        post_batch = torch.cat(post_list, dim=0)  # (B, H, W)
        
        # Add channel dimension: (B, H, W) -> (B, 1, H, W)
        pre_batch = pre_batch.unsqueeze(1)  # (B, 1, H, W)
        post_batch = post_batch.unsqueeze(1)  # (B, 1, H, W)
        
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
    
    # Compute global intensity range for all models (for consistent colormapping)
    all_volumes = [orig_norm] + [metrics_dict[model_name]['pred_norm'] for model_name in list(all_models.keys())]
    global_vmin = min([v.min() for v in all_volumes])
    global_vmax = max([v.max() for v in all_volumes])
    
    # Compute global max error across all models for difference maps
    max_errors = []
    for model_name in all_models.keys():
        sagittal_orig = orig_norm[:, sagittal_x, :]
        sagittal_pred = metrics_dict[model_name]['pred_norm'][:, sagittal_x, :]
        diff = np.abs(sagittal_orig - sagittal_pred)
        max_errors.append(np.max(diff))
    global_max_error = max(max_errors) if max_errors else 0.1
    
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
        im_sag = ax_sag.imshow(sagittal_view.T, cmap='gray', aspect='auto', origin='lower', vmin=global_vmin, vmax=global_vmax)
        
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
        im_ax = ax_ax.imshow(axial_view, cmap='gray', aspect='auto', origin='lower', vmin=global_vmin, vmax=global_vmax)
        
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
            im_diff = ax_diff.imshow(diff_view, cmap='gray', aspect='auto', origin='lower', vmin=global_vmin, vmax=global_vmax)
            ax_diff.set_title(f'Sagittal Repeat (X={sagittal_x})', fontsize=12, fontweight='bold', color='darkgreen')
        else:
            # Show difference map with consistent scale
            sagittal_orig = orig_norm[:, sagittal_x, :]
            sagittal_pred = volume_to_show[:, sagittal_x, :]
            diff = np.abs(sagittal_orig - sagittal_pred)
            im_diff = ax_diff.imshow(diff.T, cmap='hot', aspect='auto', origin='lower', vmin=0, vmax=global_max_error)
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


def generate_hierarchical_4slice_pairs(volume):
    """
    Generate hierarchical 4-slice pairs (i, i+4) for recursive interpolation.
    
    Strategy:
    1. Input (i, i+4) → Predict i+2 (middle slice)
    2. Input (i, predicted i+2) → Predict i+1 (left quarter)
    3. Input (predicted i+2, i+4) → Predict i+3 (right quarter)
    
    Returns:
        list of dict with keys:
        - 'slice_i': tensor (1, H, W)
        - 'slice_i_plus_4': tensor (1, H, W)
        - 'indices': tuple (i, i+1, i+2, i+3, i+4)
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
            'indices': (i, i+1, i+2, i+3, i+4)
        }
        pairs.append(pair)
    
    return pairs


def batch_hierarchical_pairs_for_inference(pairs, batch_size=32):
    """Batch hierarchical 4-slice pairs for inference"""
    for b in range(0, len(pairs), batch_size):
        batch = pairs[b:b + batch_size]
        
        # Concatenate slices: each slice is (1, H, W), so cat on dim=0 gives (N, H, W)
        # Then unsqueeze to get (N, 1, H, W) for proper batch format
        slice_i_list = [p['slice_i'] for p in batch]  # List of (1, H, W)
        slice_i4_list = [p['slice_i_plus_4'] for p in batch]  # List of (1, H, W)
        
        slice_i_batch = torch.cat(slice_i_list, dim=0)  # (B, H, W)
        slice_i4_batch = torch.cat(slice_i4_list, dim=0)  # (B, H, W)
        
        # Add channel dimension: (B, H, W) -> (B, 1, H, W)
        slice_i_batch = slice_i_batch.unsqueeze(1)  # (B, 1, H, W)
        slice_i4_batch = slice_i4_batch.unsqueeze(1)  # (B, 1, H, W)
        
        indices_batch = [p['indices'] for p in batch]
        
        yield slice_i_batch, slice_i4_batch, indices_batch


def predict_volume_hierarchical(model_name, seed=None, device='cuda', batch_size=8, save_path=None):
    """
    Hierarchical prediction using existing trained models:
    Stage 1: (i, i+4) → Predict i+2
    Stage 2: (i, predicted i+2) → Predict i+1
    Stage 3: (predicted i+2, i+4) → Predict i+3
    
    Args:
        model_name: Which trained model to use ('unet', 'deepcnn', 'unet_gan', etc.)
        seed: Random seed for reproducibility
        device: 'cuda' or 'cpu'
        batch_size: Batch size for inference
        save_path: Path to save visualization
    
    Returns:
        dict with volume predictions and metrics
    """
    
    print(f"\n{'='*70}")
    print(f"HIERARCHICAL 4-SLICE RECONSTRUCTION - {model_name.upper()}")
    print(f"{'='*70}")
    
    print(f"\n1. Loading random patient...")
    data = get_patient_volume_and_triplets(seed=seed)
    volume_original = data['volume']
    patient_name = data['patient_name']
    
    print(f"   Patient: {patient_name}")
    print(f"   Volume shape: {volume_original.shape}")
    
    print(f"\n2. Generating hierarchical 4-slice pairs...")
    hierarchical_pairs = generate_hierarchical_4slice_pairs(volume_original)
    print(f"   ✓ Generated {len(hierarchical_pairs)} pairs")
    
    print(f"\n3. Loading model: {model_name.upper()}...")
    try:
        model = load_model(model_name, device=device)
    except (FileNotFoundError, NotImplementedError) as e:
        print(f"   ❌ Error: {str(e)}")
        return None
    
    # Create predicted volume
    volume_predicted = volume_original.copy()
    predictions_stage1 = {}  # i+2
    predictions_stage2 = {}  # i+1
    predictions_stage3 = {}  # i+3
    
    print(f"\n4. Running hierarchical inference (3 stages)...")
    
    # STAGE 1: (i, i+4) → Predict i+2
    print(f"\n   STAGE 1: (i, i+4) → Predict i+2 (middle slice)")
    with torch.no_grad():
        for slice_i, slice_i4, indices_batch in batch_hierarchical_pairs_for_inference(hierarchical_pairs, batch_size=batch_size):
            slice_i = slice_i.to(device)
            slice_i4 = slice_i4.to(device)
            
            # Concatenate as (B, 2, H, W) - standard model input format
            x_input = torch.cat([slice_i, slice_i4], dim=1)
            
            # Model predicts middle slice
            pred_i2 = model(x_input)  # (B, 1, H, W)
            
            # Store predictions
            for idx_tuple, pred in zip(indices_batch, pred_i2):
                predictions_stage1[idx_tuple[2]] = pred.cpu().numpy()[0]  # i+2
    
    print(f"   ✓ Stage 1 complete - predicted {len(predictions_stage1)} slices")
    
    # STAGE 2: (i, predicted i+2) → Predict i+1
    print(f"\n   STAGE 2: (i, predicted i+2) → Predict i+1 (left quarter)")
    with torch.no_grad():
        for slice_i, slice_i4, indices_batch in batch_hierarchical_pairs_for_inference(hierarchical_pairs, batch_size=batch_size):
            slice_i = slice_i.to(device)
            
            # Get predicted i+2 from stage 1
            pred_i2_list = []
            for idx_tuple in indices_batch:
                if idx_tuple[2] in predictions_stage1:
                    pred_i2 = predictions_stage1[idx_tuple[2]]  # (H, W)
                else:
                    pred_i2 = np.zeros((256, 256))
                pred_i2_list.append(torch.from_numpy(pred_i2).float().unsqueeze(0))  # (1, H, W)
            
            slice_i2 = torch.cat(pred_i2_list, dim=0).unsqueeze(1).to(device)  # (B, H, W) -> (B, 1, H, W)
            
            # Input: (i, i+2)
            x_input = torch.cat([slice_i, slice_i2], dim=1)  # (B, 2, H, W)
            
            # Predict i+1
            pred_i1 = model(x_input)  # (B, 1, H, W)
            
            # Store predictions
            for idx_tuple, pred in zip(indices_batch, pred_i1):
                predictions_stage2[idx_tuple[1]] = pred.cpu().numpy()[0]  # i+1
    
    print(f"   ✓ Stage 2 complete - predicted {len(predictions_stage2)} slices")
    
    # STAGE 3: (predicted i+2, i+4) → Predict i+3
    print(f"\n   STAGE 3: (predicted i+2, i+4) → Predict i+3 (right quarter)")
    with torch.no_grad():
        for slice_i, slice_i4, indices_batch in batch_hierarchical_pairs_for_inference(hierarchical_pairs, batch_size=batch_size):
            slice_i4 = slice_i4.to(device)
            
            # Get predicted i+2 from stage 1
            pred_i2_list = []
            for idx_tuple in indices_batch:
                if idx_tuple[2] in predictions_stage1:
                    pred_i2 = predictions_stage1[idx_tuple[2]]  # (H, W)
                else:
                    pred_i2 = np.zeros((256, 256))
                pred_i2_list.append(torch.from_numpy(pred_i2).float().unsqueeze(0))  # (1, H, W)
            
            slice_i2 = torch.cat(pred_i2_list, dim=0).unsqueeze(1).to(device)  # (B, H, W) -> (B, 1, H, W)
            
            # Input: (i+2, i+4)
            x_input = torch.cat([slice_i2, slice_i4], dim=1)  # (B, 2, H, W)
            
            # Predict i+3
            pred_i3 = model(x_input)  # (B, 1, H, W)
            
            # Store predictions
            for idx_tuple, pred in zip(indices_batch, pred_i3):
                predictions_stage3[idx_tuple[3]] = pred.cpu().numpy()[0]  # i+3
    
    print(f"   ✓ Stage 3 complete - predicted {len(predictions_stage3)} slices")
    
    # Fill volume
    print(f"\n5. Assembling predicted volume...")
    all_predictions = {**predictions_stage1, **predictions_stage2, **predictions_stage3}
    for idx, pred in all_predictions.items():
        if 0 <= idx < volume_predicted.shape[0]:
            volume_predicted[idx] = pred
    
    print(f"   ✓ Filled {len(all_predictions)} slices")
    
    # Compute metrics
    metrics = compute_metrics(volume_original, volume_predicted)
    
    print(f"\n📊 Metrics:")
    print(f"   SSIM: {metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.3f}")
    print(f"   PSNR: {metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f} dB")
    print(f"   MAE:  {metrics['mae']:.4f}")
    
    print(f"\n{'='*70}")
    print(f"✅ HIERARCHICAL RECONSTRUCTION COMPLETE!")
    print(f"{'='*70}\n")
    
    return {
        'volume_original': volume_original,
        'volume_predicted': volume_predicted,
        'patient_name': patient_name,
        'metrics': metrics
    }


def predict_volume_hierarchical_all_models(seed=None, device='cuda', batch_size=8, save_path=None):
    """
    Predict volumes using hierarchical 4-slice method with all models in parallel.
    Shows all model outputs side-by-side with sagittal, axial, and difference maps.
    
    Args:
        seed: Random seed for reproducibility
        device: 'cuda' or 'cpu'
        batch_size: Batch size for inference
        save_path: Path to save the final visualization
    """
    
    print(f"\n{'='*70}")
    print(f"HIERARCHICAL 4-SLICE RECONSTRUCTION - ALL MODELS")
    print(f"{'='*70}")
    
    print(f"\n1. Loading random patient...")
    data = get_patient_volume_and_triplets(seed=seed)
    volume_original = data['volume']
    patient_name = data['patient_name']
    
    print(f"   Patient: {patient_name}")
    print(f"   Volume shape: {volume_original.shape}")
    
    print(f"\n2. Generating hierarchical 4-slice pairs...")
    hierarchical_pairs = generate_hierarchical_4slice_pairs(volume_original)
    print(f"   ✓ Generated {len(hierarchical_pairs)} pairs")
    
    # Run inference for all models
    print(f"\n3. Running hierarchical inference with all models...")
    
    all_models = {}
    model_list = ['unet', 'deepcnn', 'unet_gan']
    
    for model_name in model_list:
        print(f"\n   ⏳ Processing {model_name.upper()}...")
        
        try:
            model = load_model(model_name, device=device)
        except (FileNotFoundError, NotImplementedError) as e:
            print(f"      ⚠️  Skipped: {str(e)}")
            continue
        
        volume_predicted = volume_original.copy()
        predictions_stage1 = {}
        predictions_stage2 = {}
        predictions_stage3 = {}
        
        with torch.no_grad():
            # STAGE 1: (i, i+4) → Predict i+2
            for slice_i, slice_i4, indices_batch in batch_hierarchical_pairs_for_inference(hierarchical_pairs, batch_size=batch_size):
                slice_i = slice_i.to(device)
                slice_i4 = slice_i4.to(device)
                x_input = torch.cat([slice_i, slice_i4], dim=1)  # (B, 2, H, W)
                pred_i2 = model(x_input)
                
                for idx_tuple, pred in zip(indices_batch, pred_i2):
                    predictions_stage1[idx_tuple[2]] = pred.cpu().numpy()[0]
            
            # STAGE 2: (i, predicted i+2) → Predict i+1
            for slice_i, slice_i4, indices_batch in batch_hierarchical_pairs_for_inference(hierarchical_pairs, batch_size=batch_size):
                slice_i = slice_i.to(device)
                
                pred_i2_list = []
                for idx_tuple in indices_batch:
                    if idx_tuple[2] in predictions_stage1:
                        pred_i2 = predictions_stage1[idx_tuple[2]]  # (H, W)
                    else:
                        pred_i2 = np.zeros((256, 256))
                    pred_i2_list.append(torch.from_numpy(pred_i2).float().unsqueeze(0))  # (1, H, W)
                
                slice_i2 = torch.cat(pred_i2_list, dim=0).unsqueeze(1).to(device)  # (B, H, W) -> (B, 1, H, W)
                x_input = torch.cat([slice_i, slice_i2], dim=1)  # (B, 2, H, W)
                pred_i1 = model(x_input)
                
                for idx_tuple, pred in zip(indices_batch, pred_i1):
                    predictions_stage2[idx_tuple[1]] = pred.cpu().numpy()[0]
            
            # STAGE 3: (predicted i+2, i+4) → Predict i+3
            for slice_i, slice_i4, indices_batch in batch_hierarchical_pairs_for_inference(hierarchical_pairs, batch_size=batch_size):
                slice_i4 = slice_i4.to(device)
                
                pred_i2_list = []
                for idx_tuple in indices_batch:
                    if idx_tuple[2] in predictions_stage1:
                        pred_i2 = predictions_stage1[idx_tuple[2]]  # (H, W)
                    else:
                        pred_i2 = np.zeros((256, 256))
                    pred_i2_list.append(torch.from_numpy(pred_i2).float().unsqueeze(0))  # (1, H, W)
                
                slice_i2 = torch.cat(pred_i2_list, dim=0).unsqueeze(1).to(device)  # (B, H, W) -> (B, 1, H, W)
                x_input = torch.cat([slice_i2, slice_i4], dim=1)  # (B, 2, H, W)
                pred_i3 = model(x_input)
                
                for idx_tuple, pred in zip(indices_batch, pred_i3):
                    predictions_stage3[idx_tuple[3]] = pred.cpu().numpy()[0]
        
        # Fill volume
        all_predictions = {**predictions_stage1, **predictions_stage2, **predictions_stage3}
        for idx, pred in all_predictions.items():
            if 0 <= idx < volume_predicted.shape[0]:
                volume_predicted[idx] = pred
        
        all_models[model_name] = volume_predicted
        print(f"      ✓ {model_name.upper()} prediction complete (3 stages)")
    
    # Visualize all models in parallel
    print(f"\n4. Generating parallel visualization...")
    visualize_all_models_parallel(all_models, volume_original, patient_name, seed=seed, save_path=save_path)
    
    print(f"\n{'='*70}")
    print(f"✅ ALL MODELS PREDICTED & VISUALIZED!")
    print(f"{'='*70}\n")


def visualize_single_triplet_all_models(seed=None, device='cuda', save_path=None):
    """
    Visualize predictions from all models for a single triplet pair.
    Shows: Pre slice, Post slice, Ground truth middle, and predicted middle for each model.
    
    Args:
        seed: Random seed for reproducibility
        device: 'cuda' or 'cpu'
        save_path: Path to save the figure
    """
    
    print(f"\n{'='*70}")
    print(f"SINGLE TRIPLET VISUALIZATION - ALL MODELS")
    print(f"{'='*70}")
    
    print(f"\n1. Loading random patient and selecting a triplet...")
    data = get_patient_volume_and_triplets(seed=seed)
    volume_original = data['volume']
    triplets = data['triplets']
    patient_name = data['patient_name']
    
    print(f"   Patient: {patient_name}")
    print(f"   Total triplets: {len(triplets)}")
    
    # Select a random triplet from the middle range (to avoid edge cases)
    if seed is not None:
        np.random.seed(seed)
    triplet_idx = np.random.randint(len(triplets) // 4, 3 * len(triplets) // 4)
    selected_triplet = triplets[triplet_idx]
    
    pre = selected_triplet['pre'].to(device)  # (1, H, W)
    post = selected_triplet['post'].to(device)  # (1, H, W)
    ground_truth_middle = selected_triplet['middle']  # (1, H, W)
    middle_index = selected_triplet['index']
    
    print(f"   Selected triplet index: {triplet_idx}")
    print(f"   Slice indices: {middle_index - 1}, {middle_index} (GT), {middle_index + 1}")
    
    # Prepare input for all models
    # pre and post are (1, H, W), need to reshape to (1, 1, H, W) each, then concatenate on channel dim
    pre_input = pre.unsqueeze(0)  # (1, 1, H, W) - add batch dimension
    post_input = post.unsqueeze(0)  # (1, 1, H, W) - add batch dimension
    x_input = torch.cat([pre_input, post_input], dim=1)  # (1, 2, H, W)
    
    print(f"\n2. Running inference with all models...")
    
    all_predictions = {}
    model_list = ['unet', 'unet_combined', 'deepcnn', 'progressive_unet', 'unet_gan']
    
    for model_name in model_list:
        print(f"   ⏳ Processing {model_name.upper()}...")
        
        try:
            model = load_model(model_name, device=device)
        except (FileNotFoundError, NotImplementedError) as e:
            print(f"      ⚠️  Skipped: {str(e)}")
            continue
        
        with torch.no_grad():
            if model_name.lower() == 'progressive_unet':
                # Progressive UNet expects 5-slice window
                # For this visualization, we'll skip it since we only have a triplet
                print(f"      ⚠️  Skipped: Requires 5-slice window (triplet only available)")
                continue
            else:
                # Standard models (UNet, DeepCNN, UNet-GAN, unet_combined)
                pred_middle = model(x_input)  # (1, 1, H, W)
                all_predictions[model_name] = pred_middle.cpu().squeeze(0)  # (1, H, W)
    
    print(f"\n3. Generating visualization...")
    
    num_models = len(all_predictions)
    
    # Normalize all slices independently (each to its own range)
    pre_np = pre.cpu().squeeze(0).numpy()  # (H, W)
    post_np = post.cpu().squeeze(0).numpy()  # (H, W)
    gt_middle_np = ground_truth_middle.squeeze(0).numpy() if ground_truth_middle.dim() > 2 else ground_truth_middle.numpy()  # (H, W)
    
    # Each slice normalized independently to use full grayscale range
    pre_norm = (pre_np - pre_np.min()) / (pre_np.max() - pre_np.min() + 1e-8)
    post_norm = (post_np - post_np.min()) / (post_np.max() - post_np.min() + 1e-8)
    gt_middle_norm = (gt_middle_np - gt_middle_np.min()) / (gt_middle_np.max() - gt_middle_np.min() + 1e-8)
    
    # Normalize model predictions independently too
    model_predictions_norm = {}
    for model_name, pred in all_predictions.items():
        pred_np = pred.numpy() if isinstance(pred, torch.Tensor) else pred
        pred_np = pred_np.squeeze() if pred_np.ndim > 2 else pred_np
        pred_norm_val = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min() + 1e-8)
        model_predictions_norm[model_name] = pred_norm_val
    
    # Create figure with one row per model (no reference row)
    fig = plt.figure(figsize=(16, 4 * num_models))
    
    # Each row: PRE | POST | GROUND TRUTH | MODEL PREDICTION
    for row_idx, (model_name, pred_norm) in enumerate(model_predictions_norm.items()):
        # PRE slice
        ax_pre_model = plt.subplot(num_models, 4, row_idx * 4 + 1)
        im_pre_model = ax_pre_model.imshow(pre_norm, cmap='gray')
        ax_pre_model.set_title(f'PRE\n(Slice {middle_index - 1})', fontsize=11, fontweight='bold', color='darkblue')
        ax_pre_model.axis('off')
        plt.colorbar(im_pre_model, ax=ax_pre_model, fraction=0.046, pad=0.04)
        
        # POST slice
        ax_post_model = plt.subplot(num_models, 4, row_idx * 4 + 2)
        im_post_model = ax_post_model.imshow(post_norm, cmap='gray')
        ax_post_model.set_title(f'POST\n(Slice {middle_index + 1})', fontsize=11, fontweight='bold', color='darkblue')
        ax_post_model.axis('off')
        plt.colorbar(im_post_model, ax=ax_post_model, fraction=0.046, pad=0.04)
        
        # GROUND TRUTH slice
        ax_gt_model = plt.subplot(num_models, 4, row_idx * 4 + 3)
        im_gt_model = ax_gt_model.imshow(gt_middle_norm, cmap='gray')
        ax_gt_model.set_title(f'GROUND TRUTH\n(Slice {middle_index})', fontsize=11, fontweight='bold', color='darkgreen')
        ax_gt_model.axis('off')
        plt.colorbar(im_gt_model, ax=ax_gt_model, fraction=0.046, pad=0.04)
        
        # MODEL PREDICTION
        ax_pred_model = plt.subplot(num_models, 4, row_idx * 4 + 4)
        im_pred_model = ax_pred_model.imshow(pred_norm, cmap='gray')
        
        # Calculate MSE with ground truth
        mse = np.mean((gt_middle_norm - pred_norm) ** 2)
        
        ax_pred_model.set_title(f'{model_name.upper()}\nMSE: {mse:.4f}', fontsize=11, fontweight='bold', color='darkred')
        ax_pred_model.axis('off')
        plt.colorbar(im_pred_model, ax=ax_pred_model, fraction=0.046, pad=0.04)
    
    # Overall title
    title_str = f'Single Triplet Prediction Comparison - All Models\nPatient: {patient_name} (Triplet Index: {triplet_idx}, Seed: {seed})'
    fig.suptitle(title_str, fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   ✓ Visualization saved to: {save_path}")
    
    plt.show()
    
    print(f"\n{'='*70}")
    print(f"✅ SINGLE TRIPLET VISUALIZATION COMPLETE!")
    print(f"{'='*70}\n")


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
                    pre_batch = pre_batch.to(device)
                    post_batch = post_batch.to(device)
                
                    # Stack pre and post as input (B, 2, H, W)
                    x_input = torch.cat([pre_batch, post_batch], dim=1)
                
                    # Predict
                    predictions = model(x_input)
                
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


def predict_volume_all_models_with_fastddpm(seed=None, device='cuda', batch_size=8, save_path=None, view='sagittal'):
    """
    Predict full volume using all models including FastDDPM Advanced.
    Display views of all models side-by-side for easy comparison.
    
    Args:
        seed: Random seed for reproducibility
        device: 'cuda' or 'cpu'
        batch_size: Batch size for inference
        save_path: Path to save the final visualization
        view: 'sagittal' (Y-Z plane) or 'axial' (X-Y plane) - default 'sagittal'
    
    Returns:
        Dictionary with all predictions and metrics
    """
    
    print(f"\n{'='*70}")
    print(f"FULL VOLUME PREDICTION - ALL MODELS + FastDDPM")
    print(f"View: {view.upper()}")
    print(f"{'='*70}")
    
    print(f"\n1. Loading random patient from test set...")
    data = get_patient_volume_and_triplets(seed=seed)
    volume_original = data['volume']
    triplets = data['triplets']
    patient_name = data['patient_name']
    
    print(f"   Patient: {patient_name}")
    print(f"   Volume shape: {volume_original.shape}")
    print(f"   Triplets: {len(triplets)}")
    
    # Run inference on all triplets with standard models
    print(f"\n2. Running inference with standard models...")
    
    all_models = {}
    standard_models = ['unet', 'deepcnn', 'unet_gan']
    
    for model_name in standard_models:
        print(f"\n   ⏳ Processing {model_name.upper()}...")
        
        try:
            model = load_model(model_name, device=device)
        except (FileNotFoundError, NotImplementedError) as e:
            print(f"      ⚠️  Skipped: {str(e)}")
            continue
        
        volume_predicted = volume_original.copy()
        predictions_dict = {}
        
        with torch.no_grad():
            for pre_batch, post_batch, indices in batch_triplets_for_inference(triplets, batch_size=batch_size):
                pre_batch = pre_batch.to(device)
                post_batch = post_batch.to(device)
                x_input = torch.cat([pre_batch, post_batch], dim=1)
                predictions = model(x_input)
                
                for idx, pred in zip(indices, predictions):
                    predictions_dict[idx] = pred.cpu().numpy()[0]
        
        # Fill predicted volume
        for idx, pred in predictions_dict.items():
            if 0 <= idx < volume_predicted.shape[0]:
                volume_predicted[idx] = pred
        
        all_models[model_name] = volume_predicted
        print(f"      ✓ {model_name.upper()} prediction complete")
    
    # Load and run FastDDPM Advanced
    print(f"\n   ⏳ Processing FASTDDPM ADVANCED...")
    try:
        model = load_model('fastddpm', device=device)
        
        volume_predicted = volume_original.copy()
        predictions_dict = {}
        
        with torch.no_grad():
            for pre_batch, post_batch, indices in batch_triplets_for_inference(triplets, batch_size=batch_size):
                pre_batch = pre_batch.to(device)
                post_batch = post_batch.to(device)
                
                # Concatenate as (B, 2, H, W) for FastDDPM
                cond = torch.cat([pre_batch, post_batch], dim=1)
                
                # Generate predictions using DDIM sampling
                pred = model.sample(cond, device)  # (B, 1, H, W)
                
                for idx, p in zip(indices, pred):
                    predictions_dict[idx] = p.cpu().numpy()[0]
        
        # Fill predicted volume
        for idx, pred in predictions_dict.items():
            if 0 <= idx < volume_predicted.shape[0]:
                volume_predicted[idx] = pred
        
        all_models['fastddpm'] = volume_predicted
        print(f"      ✓ FASTDDPM prediction complete")
        
    except (FileNotFoundError, NotImplementedError) as e:
        print(f"      ⚠️  Skipped: {str(e)}")
    except Exception as e:
        print(f"      ⚠️  Skipped: {str(e)}")
    
    # Compute metrics for all models
    print(f"\n3. Computing metrics for all models...")
    metrics_dict = {}
    
    for model_name, volume_pred in all_models.items():
        metrics = compute_metrics(volume_original, volume_pred)
        metrics_dict[model_name] = metrics
        print(f"   {model_name:15s} | SSIM: {metrics['ssim_mean']:.4f}±{metrics['ssim_std']:.3f} | "
              f"PSNR: {metrics['psnr_mean']:.2f}±{metrics['psnr_std']:.2f}")
    
    # Create comprehensive view visualization
    print(f"\n4. Generating {view.upper()} view comparison...")
    
    num_models = len(all_models) + 1  # +1 for original
    model_names = ['Original'] + list(all_models.keys())
    
    # Normalize original volume
    orig_norm = (volume_original - volume_original.min()) / (volume_original.max() - volume_original.min() + 1e-8)
    
    # Determine positions based on view type
    if view.lower() == 'sagittal':
        # Sagittal view: Y-Z plane at different X positions
        positions = [64, 128, 192]
        view_label = 'Sagittal (X=%d)'
        view_extractor = lambda vol, pos: vol[:, pos, :]  # (Z, Y)
    elif view.lower() == 'axial':
        # Axial view: X-Y plane at different Z positions
        positions = [10, 30, 50]
        view_label = 'Axial (Z=%d)'
        view_extractor = lambda vol, pos: vol[pos, :, :]  # (Y, X)
    else:
        raise ValueError(f"Invalid view: {view}. Must be 'sagittal' or 'axial'")
    
    # Create figure with views
    fig = plt.figure(figsize=(20, 4 * num_models))
    
    # Compute global intensity range for consistent colormapping
    all_volumes = [orig_norm] + [metrics_dict[model_name]['pred_norm'] for model_name in list(all_models.keys())]
    global_vmin = min([v.min() for v in all_volumes])
    global_vmax = max([v.max() for v in all_volumes])
    
    for row, (model_idx, model_name) in enumerate(enumerate(model_names)):
        if model_name == 'Original':
            volume_to_show = orig_norm
            metrics = None
        else:
            volume_to_show = metrics_dict[model_name]['pred_norm']
            metrics = metrics_dict[model_name]
        
        for col, pos in enumerate(positions):
            ax = plt.subplot(num_models, len(positions), row * len(positions) + col + 1)
            
            # Extract view
            view_data = view_extractor(volume_to_show, pos)
            
            # Transpose for proper display
            if view.lower() == 'sagittal':
                im = ax.imshow(view_data.T, cmap='gray', aspect='auto', origin='lower', 
                              vmin=global_vmin, vmax=global_vmax)
            else:  # axial
                im = ax.imshow(view_data, cmap='gray', aspect='auto', origin='lower', 
                              vmin=global_vmin, vmax=global_vmax)
            
            if model_name == 'Original':
                ax.set_title(f'Original {view_label % pos}', fontsize=12, fontweight='bold', color='darkgreen')
            else:
                title = f'{model_name.upper()}\n{view_label % pos} | SSIM: {metrics["ssim_mean"]:.4f} | PSNR: {metrics["psnr_mean"]:.2f}'
                color = 'darkblue' if model_name != 'fastddpm' else 'darkred'
                ax.set_title(title, fontsize=11, fontweight='bold', color=color)
            
            if view.lower() == 'sagittal':
                ax.set_xlabel('Slice Index (Z)', fontsize=9)
                ax.set_ylabel('Y Position', fontsize=9)
            else:  # axial
                ax.set_xlabel('X Position', fontsize=9)
                ax.set_ylabel('Y Position', fontsize=9)
            
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Intensity', fontsize=8)
    
    # Overall title
    title_str = f'{view.upper()} View Comparison - All Models\nPatient: {patient_name} (Seed: {seed})'
    fig.suptitle(title_str, fontsize=15, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   ✓ Visualization saved to: {save_path}")
    
    plt.show()
    
    print(f"\n{'='*70}")
    print(f"✅ FULL VOLUME PREDICTION COMPLETE!")
    print(f"{'='*70}\n")
    
    return {
        'volume_original': volume_original,
        'all_predictions': all_models,
        'metrics': metrics_dict,
        'patient_name': patient_name
    }


if __name__ == "__main__":
    # Example 1: Standard triplet prediction with all models
    # predict_volume_and_visualize(seed=42, device='cuda', batch_size=16)
    
    # Example 2: Hierarchical 4-slice reconstruction with a specific model
    print("="*70)
    print("EXAMPLE 1: Hierarchical 4-Slice (Single Model)")
    print("="*70)
    
    # result = predict_volume_hierarchical(
    #     model_name='unet',  # Choose: 'unet', 'deepcnn', 'unet_gan'
    #     seed=42,
    #     device='cuda',
    #     batch_size=16
    # )
    
    # Example 2: Full volume prediction with all models + FastDDPM
    print("\n" + "="*70)
    print("EXAMPLE 2: Full Volume Prediction (All Models + FastDDPM)")
    print("="*70)
    
    # results = predict_volume_all_models_with_fastddpm(
    #     seed=42,
    #     device='cuda',
    #     batch_size=8,
    #     save_path='results/sagittal_comparison_all_models.png'
    # )
    
    # Example 3: Data structure exploration
    print("\n" + "="*70)
    print("DATA STRUCTURE")
    print("="*70)
    
    data = get_patient_volume_and_triplets(seed=42)
    print(f"\nPatient: {data['patient_name']}")
    print(f"Volume shape: {data['volume'].shape}")
    
    triplets = data['triplets']
    print(f"\nTriplet (for standard models):")
    print(f"  Pre shape: {triplets[0]['pre'].shape}")
    print(f"  Post shape: {triplets[0]['post'].shape}")
    
    pairs = generate_hierarchical_4slice_pairs(data['volume'])
    print(f"\nHierarchical pair (i, i+4):")
    print(f"  Slice i shape: {pairs[0]['slice_i'].shape}")
    print(f"  Slice i+4 shape: {pairs[0]['slice_i_plus_4'].shape}")
    print(f"  Indices: {pairs[0]['indices']}")