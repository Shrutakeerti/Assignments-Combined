import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
from sklearn.model_selection import train_test_split
from monai.transforms import (
    Compose,
    LoadImaged,
    AddChanneld,
    ScaleIntensityRanged,
    EnsureTyped,
)

class CTAbdomenDataset(Dataset):
    def __init__(self, image_paths, label_paths, transform=None):
        """
        Args:
            image_paths (list): List of paths to CT scan images.
            label_paths (list): List of paths to corresponding label masks.
            transform (callable, optional): Optional transform to be applied.
        """
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = sitk.ReadImage(self.image_paths[idx])
        label = sitk.ReadImage(self.label_paths[idx])

        image = sitk.GetArrayFromImage(image).astype(np.float32)
        label = sitk.GetArrayFromImage(label).astype(np.int64)

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample['image'], sample['label']

def get_data_loaders(data_dir, batch_size=2, val_split=0.2, test_split=0.1):
    """
    Creates training, validation, and test DataLoaders.

    Args:
        data_dir (str): Directory containing 'images' and 'labels' folders.
        batch_size (int): Batch size for DataLoaders.
        val_split (float): Fraction of data for validation.
        test_split (float): Fraction of data for testing.

    Returns:
        train_loader, val_loader, test_loader (DataLoader): DataLoaders.
    """
    images_dir = os.path.join(data_dir, 'images')
    labels_dir = os.path.join(data_dir, 'labels')

    image_files = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.nii')])
    label_files = sorted([os.path.join(labels_dir, f) for f in os.listdir(labels_dir) if f.endswith('.nii')])

    # Split data
    train_imgs, temp_imgs, train_labels, temp_labels = train_test_split(
        image_files, label_files, test_size=(val_split + test_split), random_state=42
    )
    val_imgs, test_imgs, val_labels, test_labels = train_test_split(
        temp_imgs, temp_labels, test_size=test_split / (val_split + test_split), random_state=42
    )

    # Define transforms
    train_transforms = Compose([
        AddChanneld(keys=['image', 'label']),
        ScaleIntensityRanged(
            keys=['image'],
            a_min=-57,
            a_max=164,
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),
        EnsureTyped(keys=['image', 'label'])
    ])

    val_test_transforms = Compose([
        AddChanneld(keys=['image', 'label']),
        ScaleIntensityRanged(
            keys=['image'],
            a_min=-57,
            a_max=164,
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),
        EnsureTyped(keys=['image', 'label'])
    ])

    # Create datasets
    train_ds = CTAbdomenDataset(train_imgs, train_labels, transform=train_transforms)
    val_ds = CTAbdomenDataset(val_imgs, val_labels, transform=val_test_transforms)
    test_ds = CTAbdomenDataset(test_imgs, test_labels, transform=val_test_transforms)

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader
