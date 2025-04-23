import os
import torch
from torch.utils.data import Dataset
import nibabel as nib

class MedSegDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith('.nii.gz') and 'img' in f
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]

        image_path = os.path.join(self.image_dir, image_filename)
        mask_filename = image_filename.replace("img", "mask")
        mask_path = os.path.join(self.mask_dir, mask_filename)

        image = nib.load(image_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()

        # Use center slice for simplicity
        z = image.shape[2] // 2
        image = torch.tensor(image[:, :, z], dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask[:, :, z], dtype=torch.float32)
        mask = (mask > 0).float().unsqueeze(0)  # Binarize: everything >0 becomes 1

        return image, mask
