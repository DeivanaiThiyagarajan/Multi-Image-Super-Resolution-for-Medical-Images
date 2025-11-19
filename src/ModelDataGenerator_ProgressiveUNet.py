"""
Data Generator for Progressive UNet Model
Provides 5 consecutive slices (i, i+1, i+2, i+3, i+4) for progressive SR

Key design:
- Patient-level split: 70% train, 15% val, 15% test (NO data leakage across splits)
- 5-slice windows are generated WITHIN each patient's volume
- All 5 consecutive slices must come from the same patient (no cross-patient windows)
- Follows the same patient-based splitting strategy as ModelDataGenerator.py
"""

import os
import torch
import torch.nn.functional as TF
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import SimpleITK as sitk

# Setup base directory
parent_directory = os.path.dirname(os.getcwd())
BASE_DIR = os.path.join(parent_directory, 'data', 'manifest-1694710246744_1', 'Prostate-MRI-US-Biopsy')


def load_correct_study(patient_path):
    """
    Find all subfolders containing exactly 60 DICOM files.
    Returns a list of paths.
    """
    series_folders = []
    for root, dirs, files in os.walk(patient_path):
        dcm_files = [f for f in files if f.lower().endswith(".dcm")]
        if len(dcm_files) == 60:
            series_folders.append(root)
    return series_folders if series_folders else None


def load_patient_volume(series_folder_path):
    """
    Load volume from a specific series folder path.
    Args:
        series_folder_path: Direct path to folder containing 60 DICOM files
    Returns:
        volume_np: (Z, H, W) numpy array
    """
    if series_folder_path is None:
        return None

    # Read all DICOM slices in the folder
    dcm_files = sorted([os.path.join(series_folder_path, f) 
                        for f in os.listdir(series_folder_path) 
                        if f.lower().endswith('.dcm')])
    
    if len(dcm_files) < 5:
        # Not enough slices for 5-slice windows
        return None

    slices = []
    for f in dcm_files:
        img = sitk.ReadImage(f)
        arr = sitk.GetArrayFromImage(img)[0]  # (1,H,W) -> (H,W)
        slices.append(arr.astype(np.float32))

    # Stack into a volume (Z,H,W)
    volume_np = np.stack(slices, axis=0)
    return volume_np


def generate_consecutive_5tuples(volume):
    """
    Generate 5-consecutive-slice windows from volume:
    [slice[i], slice[i+1], slice[i+2], slice[i+3], slice[i+4]]
    
    Returns list of (5, H, W) windows
    """
    windows = []
    
    # Create all 5-consecutive windows
    for i in range(volume.shape[0] - 4):
        # Normalize each slice independently
        slices_5 = []
        for j in range(5):
            s = volume[i + j].astype(np.float32)
            # Z-score normalization per slice
            s_normalized = (s - s.mean()) / (s.std() + 1e-6)
            slices_5.append(s_normalized)
        
        # Stack into (5, H, W)
        window = np.stack(slices_5, axis=0)
        windows.append(window)
    
    return windows


class ProgressiveUNetDataset(Dataset):
    """
    Dataset for Progressive UNet Model
    
    Returns 5 consecutive slices: [i, i+1, i+2, i+3, i+4]
    
    Progressive UNet stages:
    - Stage 1: UNet1(i, i+4) -> i+2 prediction
    - Stage 2A: UNet2(i, i+2_gen) -> i+1 prediction
    - Stage 2B: UNet3(i+2_gen, i+4) -> i+3 prediction
    
    Ground truth targets: i+1, i+2, i+3
    
    IMPORTANT: 5-slice windows are generated WITHIN each patient's volume only.
    No windows span across different patients (prevents data leakage).
    
    Uses lazy loading (like ModelDataGenerator.py) - only loads volumes on demand in __getitem__
    """
    
    def __init__(self, patient_folders, augment=False):
        """
        Args:
            patient_folders: List of full paths to patient folders
            augment: Whether to apply augmentations
        """
        self.patient_folders = patient_folders
        self.augment = augment
        
        # Cache for loaded volumes: {(patient_idx, series_idx): (volume, windows)}
        self._volume_cache = {}
        
        # Build patient-series mappings (lazy - only list folders, don't load)
        self._build_patient_series_map()
    
    def _build_patient_series_map(self):
        """
        Build mapping of patient -> series folders.
        Does NOT load volumes yet (lazy loading in __getitem__)
        """
        self.patient_series_map = {}  # patient_idx -> [series_folders]
        self.patient_series_counts = {}  # patient_idx -> num_series
        
        for patient_idx, patient_folder in enumerate(self.patient_folders):
            series_folders = load_correct_study(patient_folder)
            
            if series_folders is None or len(series_folders) == 0:
                continue
            
            self.patient_series_map[patient_idx] = series_folders
            self.patient_series_counts[patient_idx] = len(series_folders)
    
    def _get_num_windows_for_series(self, patient_idx, series_idx):
        """
        Get number of 5-slice windows for a specific series.
        Loads volume to check but caches it.
        """
        cache_key = (patient_idx, series_idx)
        
        if cache_key not in self._volume_cache:
            # Load and cache the volume
            series_folder = self.patient_series_map[patient_idx][series_idx]
            volume = load_patient_volume(series_folder)
            
            if volume is None:
                return 0
            
            windows = generate_consecutive_5tuples(volume)
            self._volume_cache[cache_key] = (volume, windows)
        
        _, windows = self._volume_cache[cache_key]
        return len(windows)
    
    def __len__(self):
        """
        Total number of 5-slice windows across all patients and series.
        Computed lazily by checking each series.
        """
        total = 0
        for patient_idx in self.patient_series_map.keys():
            n_series = self.patient_series_counts[patient_idx]
            for series_idx in range(n_series):
                total += self._get_num_windows_for_series(patient_idx, series_idx)
        return total
    
    def __getitem__(self, idx):
        """
        Lazy loading: convert flat index to (patient_idx, series_idx, window_idx)
        Load volume only when needed, then cache it.
        """
        # Convert flat index to (patient_idx, series_idx, window_idx)
        current_idx = 0
        
        for patient_idx in sorted(self.patient_series_map.keys()):
            n_series = self.patient_series_counts[patient_idx]
            
            for series_idx in range(n_series):
                n_windows = self._get_num_windows_for_series(patient_idx, series_idx)
                
                if current_idx + n_windows > idx:
                    # Found the right series
                    window_idx = idx - current_idx
                    
                    # Get cached volume and windows
                    cache_key = (patient_idx, series_idx)
                    volume, windows = self._volume_cache[cache_key]
                    
                    # Get the specific window
                    window = windows[window_idx]  # (5, H, W)
                    
                    # Convert to tensor
                    window_tensor = torch.from_numpy(window).float()  # (5, H, W)
                    
                    # Resize to 256x256 for uniform batch sizes
                    window_tensor = TF.interpolate(
                        window_tensor.unsqueeze(0),  # (1, 5, H, W)
                        size=(256, 256),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)  # (5, H, W)
                    
                    # Apply augmentation if needed
                    if self.augment:
                        window_tensor = self._apply_augmentation(window_tensor)
                    
                    return window_tensor
                
                current_idx += n_windows
        
        raise IndexError(f"Index {idx} out of range for dataset")
    
    def _apply_augmentation(self, window_tensor):
        """Apply augmentations consistently across all 5 slices"""
        # Random horizontal flip
        if np.random.rand() > 0.5:
            window_tensor = torch.flip(window_tensor, dims=[-1])
        
        # Random vertical flip
        if np.random.rand() > 0.5:
            window_tensor = torch.flip(window_tensor, dims=[-2])
        
        # Random rotation (0 or 90 degree increments for medical images)
        rot_k = np.random.randint(0, 4)
        if rot_k > 0:
            window_tensor = torch.rot90(window_tensor, k=rot_k, dims=[-2, -1])
        
        return window_tensor


def build_progressive_dataloader(split='train', batch_size=4, augment=False, num_workers=8):
    """
    Build dataloader for Progressive UNet
    
    Args:
        split: 'train', 'val', or 'test'
        batch_size: Batch size
        augment: Whether to apply augmentations
        num_workers: Number of workers for dataloader
    
    Returns:
        DataLoader with 5 consecutive slices per sample
        
    Important: Patient-level split ensures NO data leakage between train/val/test
    """
    
    # Get all patient folders
    patient_folders_all = sorted([
        f for f in os.listdir(BASE_DIR)
        if f.startswith("Prostate-MRI-US-Biopsy-")
    ])
    
    # Split at patient level: 70% train, 15% val, 15% test
    train_folders, test_val_folders = train_test_split(
        patient_folders_all, test_size=0.3, random_state=42
    )
    
    val_folders, test_folders = train_test_split(
        test_val_folders, test_size=0.6, random_state=42
    )
    
    # Select folders based on split
    if split == "train":
        selected_folders = train_folders
    elif split == "val":
        selected_folders = val_folders
    else:  # test
        selected_folders = test_folders
    
    # Build full paths
    patient_folder_paths = [
        os.path.join(BASE_DIR, folder_name)
        for folder_name in selected_folders
    ]
    
    # Create dataset
    dataset = ProgressiveUNetDataset(
        patient_folders=patient_folder_paths,
        augment=augment
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader
