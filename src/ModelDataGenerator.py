import os
import random
import numpy as np
import SimpleITK as sitk
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

parent_directory = os.path.dirname(os.getcwd())
BASE_DIR = os.path.join(parent_directory, 'data','manifest-1694710246744','Prostate-MRI-US-Biopsy')


def load_correct_study(patient_path):
    """
    Find subfolder containing exactly 60 DICOM files.
    """
    for root, dirs, files in os.walk(patient_path):
        dcm_files = [f for f in files if f.lower().endswith(".dcm")]
        if len(dcm_files) == 60:
            return root
    return None


def load_patient_volume(patient_folder):

    # 1) Find the series folder with exactly 60 dicom slices
    study_folder = load_correct_study(patient_folder)
    if study_folder is None:
        return None

    # 2) Let SimpleITK detect the series inside the folder
    reader = sitk.ImageSeriesReader()
    series_IDs = reader.GetGDCMSeriesIDs(study_folder)

    if not series_IDs:
        print(f"No DICOM series found in {study_folder}")
        return None

    # Usually just 1 series — get it
    series_files = reader.GetGDCMSeriesFileNames(study_folder, series_IDs[0])

    # 3) Read the entire volume in a SINGLE call
    reader.SetFileNames(series_files)
    img = reader.Execute() 

    # 4) Convert to numpy (Z,H,W)
    vol = sitk.GetArrayFromImage(img).astype(np.float32)

    # volume shape check
    if vol.shape[0] != 60:
        print(f"Warning: expected 60 slices, got {vol.shape[0]}")
        return None

    return vol

def generate_consecutive_triplets(volume):
    """
    Generate overlapping triplets:
    (slice[i], slice[i+2]) -> slice[i+1]
    (slice[i], slice[i+4]) -> slice[i+2]
    """
    pre, post, mid = [], [], []

    # Distance 2
    for i in range(volume.shape[0] - 2):
        pre.append(volume[i])
        post.append(volume[i + 2])
        mid.append(volume[i + 1])

    # Distance 4
    for i in range(volume.shape[0] - 4):
        pre.append(volume[i])
        post.append(volume[i + 4])
        mid.append(volume[i + 2])

    return pre, post, mid

class PairedTransforms:
    def __call__(self, sample):
        pre = sample["pre"]
        post = sample["post"]
        mid = sample["target"]

        # Horizontal flip
        if random.random() < 0.5:
            pre  = TF.hflip(pre)
            post = TF.hflip(post)
            mid  = TF.hflip(mid)

        # Vertical flip
        if random.random() < 0.5:
            pre  = TF.vflip(pre)
            post = TF.vflip(post)
            mid  = TF.vflip(mid)

        # Rotation (use bilinear!)
        angle = random.uniform(-5, 5)
        pre  = TF.rotate(pre, angle, interpolation=TF.InterpolationMode.BILINEAR)
        post = TF.rotate(post, angle, interpolation=TF.InterpolationMode.BILINEAR)
        mid  = TF.rotate(mid, angle, interpolation=TF.InterpolationMode.BILINEAR)

        return {"pre": pre, "post": post, "target": mid}
    

class TripletSliceDataset(Dataset):
    def __init__(self, patient_folders, transform=None):
        self.patient_folders = patient_folders  # only folder paths
        self.transform = transform

    def __len__(self):
        return len(self.patient_folders)

    def load_volume(self, folder):
        vol = load_patient_volume(folder)  # load volume on-the-fly
        return vol

    def __getitem__(self, idx):
        folder = self.patient_folders[idx]
        vol = self.load_volume(folder)  # (Z,H,W)

        # Select random triplet inside this volume
        i = np.random.randint(0, vol.shape[0]-2)
        pre = vol[i]
        mid = vol[i+1]
        post = vol[i+2]

        # Normalize slices
        pre = (pre - pre.mean()) / (pre.std()+1e-6)
        mid = (mid - mid.mean()) / (mid.std()+1e-6)
        post = (post - post.mean()) / (post.std()+1e-6)

        # Convert to tensors
        pre  = torch.tensor(pre).unsqueeze(0)
        mid  = torch.tensor(mid).unsqueeze(0)
        post = torch.tensor(post).unsqueeze(0)

        sample = {"pre": pre, "post": post, "target": mid}

        if self.transform:
            sample = self.transform(sample)

        return (sample["pre"], sample["post"]), sample["target"]


def build_dataloader(split="train",
                     batch_size=4,
                     augment=False,
                     num_workers=4):
    """
    base_dir: path to Prostate-MRI-US-Biopsy dataset folder
    split: "train", "val", "test"
    batch_size: dataloader batch size
    augment: whether to apply PairedTransforms
    """

    # ---------------------------------------------------------
    # Split patient folders
    # ---------------------------------------------------------

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

    if split == "train":
        folders = train_folders
    elif split == "val":
        folders = val_folders
    else:
        folders = test_folders

    # ---------------------------------------------------------
    # Read patient volumes + generate triplets
    # ---------------------------------------------------------

    all_pre, all_post, all_mid = [], [], []

    for pid in folders:
        vol = load_patient_volume(os.path.join(BASE_DIR, pid))
        if vol is None:
            continue

        pre, post, mid = generate_consecutive_triplets(vol)
        all_pre.extend(pre)
        all_post.extend(post)
        all_mid.extend(mid)

    transform = PairedTransforms() if augment else None

    dataset = TripletSliceDataset(folders, transform)

    # ---------------------------------------------------------
    # Build DataLoader
    # ---------------------------------------------------------

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader

if __name__ == "__main__":
    BASE = "/path/to/data/manifest-1694710246744/Prostate-MRI-US-Biopsy"

    train_loader = build_dataloader(
        split="train",
        batch_size=4,
        augment=True
    )

    for batch in train_loader:
        (pre, post), mid = batch
        print(pre.shape, post.shape, mid.shape)
        break
