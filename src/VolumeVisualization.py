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
    
    # 2 views per model (sagittal and axial)
    fig = plt.figure(figsize=(22, 5 * num_models))
    
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
        ax_sag = plt.subplot(num_models, 2, row * 2 + 1)
        
        sagittal_view = volume_to_show[:, sagittal_x, :]
        im_sag = ax_sag.imshow(sagittal_view.T, cmap='gray', aspect='auto', origin='lower')
        
        if model_name == 'Original':
            ax_sag.set_title(f'Sagittal View (X={sagittal_x})', fontsize=12, fontweight='bold', color='black')
        else:
            title = f'{model_name.upper()} - Sagittal\nSSIM: {metrics["ssim_mean"]:.4f} | PSNR: {metrics["psnr_mean"]:.2f}'
            ax_sag.set_title(title, fontsize=11, fontweight='bold', color='darkblue')
        
        ax_sag.set_xlabel('Slice Index (Z)', fontsize=10)
        ax_sag.set_ylabel('Y Position', fontsize=10)
        cbar_sag = plt.colorbar(im_sag, ax=ax_sag, fraction=0.046, pad=0.04)
        cbar_sag.set_label('Intensity', fontsize=9)
        
        # ===== AXIAL VIEW (X-Y plane at z=axial_z) =====
        ax_ax = plt.subplot(num_models, 2, row * 2 + 2)
        
        axial_view = volume_to_show[axial_z, :, :]
        im_ax = ax_ax.imshow(axial_view, cmap='gray', aspect='auto', origin='lower')
        
        if model_name == 'Original':
            ax_ax.set_title(f'Axial View (Z={axial_z})', fontsize=12, fontweight='bold', color='black')
        else:
            title = f'{model_name.upper()} - Axial\nMAE: {metrics["mae"]:.4f}'
            ax_ax.set_title(title, fontsize=11, fontweight='bold', color='darkgreen')
        
        ax_ax.set_xlabel('X Position', fontsize=10)
        ax_ax.set_ylabel('Y Position', fontsize=10)
        cbar_ax = plt.colorbar(im_ax, ax=ax_ax, fraction=0.046, pad=0.04)
        cbar_ax.set_label('Intensity', fontsize=9)
    
    # Overall title
    title_str = f'Multi-Model Comparison - Sagittal & Axial Views\nPatient: {patient_name} (Seed: {seed})'
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
    model_list = ['unet', 'deepcnn', 'progressive_unet', 'unet_gan']
    
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
                    predictions = model(window_batch)  # (B, 3, H, W) - outputs 3 predictions
                    
                    # Progressive UNet outputs 3 slices: i+1, i+2, i+3
                    # indices contain the middle slice index (i+2)
                    pred_i1 = predictions[:, 0:1, :, :]  # (B, 1, H, W) - i+1
                    pred_i2 = predictions[:, 1:2, :, :]  # (B, 1, H, W) - i+2
                    pred_i3 = predictions[:, 2:3, :, :]  # (B, 1, H, W) - i+3
                    
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
        for model_name, volume_pred in all_models.items():
            metrics = compute_metrics(volume_original, volume_pred)
            
            # Visualization code for single model
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            x_positions = [64, 128, 192]
            orig_norm = metrics['orig_norm']
            pred_norm = metrics['pred_norm']
            
            for col, x_pos in enumerate(x_positions):
                # Original sagittal
                orig_sagittal = orig_norm[:, x_pos, :]
                im0 = axes[0, col].imshow(orig_sagittal.T, cmap='gray', aspect='auto')
                axes[0, col].set_title(f'Original Sagittal (X={x_pos})', fontsize=11, fontweight='bold')
                axes[0, col].set_xlabel('Slice Index (Z)')
                axes[0, col].set_ylabel('Y Position')
                plt.colorbar(im0, ax=axes[0, col], fraction=0.046)
                
                # Predicted sagittal
                pred_sagittal = pred_norm[:, x_pos, :]
                im1 = axes[1, col].imshow(pred_sagittal.T, cmap='gray', aspect='auto')
                axes[1, col].set_title(f'{model_name.upper()} Sagittal (X={x_pos})', fontsize=11, fontweight='bold')
                axes[1, col].set_xlabel('Slice Index (Z)')
                axes[1, col].set_ylabel('Y Position')
                plt.colorbar(im1, ax=axes[1, col], fraction=0.046)
            
            fig.suptitle(
                f'{model_name.upper()} - Volume Prediction & Sagittal Comparison\n'
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


if __name__ == "__main__":
    # Example usage
    print("Loading random patient from test set...")
    data = get_patient_volume_and_triplets(seed=42)
    
    print(f"\nPatient: {data['patient_name']}")
    print(f"Volume shape: {data['volume'].shape}")
    print(f"Number of triplets: {data['num_triplets']}")
    print(f"Series path: {data['series_path']}")
    
    triplets = data['triplets']
    print(f"\nFirst triplet:")
    print(f"  Pre shape: {triplets[0]['pre'].shape}")
    print(f"  Post shape: {triplets[0]['post'].shape}")
    print(f"  Middle shape: {triplets[0]['middle'].shape}")
    print(f"  Middle slice index: {triplets[0]['index']}")
    
    # Demonstrate batching
    print(f"\nBatching triplets for inference (batch_size=8):")
    for batch_num, (pre, post, indices) in enumerate(batch_triplets_for_inference(triplets, batch_size=8)):
        print(f"  Batch {batch_num + 1}: pre {pre.shape}, post {post.shape}, indices {indices}")
