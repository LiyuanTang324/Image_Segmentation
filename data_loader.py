import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.transforms import functional as TF


class SegmentationDataset(Dataset):
    """Unified dataset loader for all 7 segmentation datasets.

    Handles mixed image modes (L / RGB) and mask modes (bool / L / RGB)
    by converting images to RGB and masks to single-channel binary.
    """

    def __init__(self, image_dir, mask_dir, split_file,
                 image_size=224, mode='train', augmentation_prob=0.4):
        with open(split_file, 'r') as f:
            self.filenames = [line.strip() for line in f if line.strip()]
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.mode = mode
        self.augmentation_prob = augmentation_prob
        self.color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        fname = self.filenames[index]
        image = Image.open(os.path.join(self.image_dir, fname)).convert('RGB')
        mask = Image.open(os.path.join(self.mask_dir, fname)).convert('L')

        image = TF.resize(image, [self.image_size, self.image_size],
                          interpolation=T.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.image_size, self.image_size],
                         interpolation=T.InterpolationMode.NEAREST)

        if self.mode == 'train' and np.random.random() < self.augmentation_prob:
            if np.random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if np.random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            angle = np.random.uniform(-15, 15)
            image = TF.rotate(image, angle,
                              interpolation=T.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle,
                             interpolation=T.InterpolationMode.NEAREST)
            image = self.color_jitter(image)

        image = TF.to_tensor(image)                                          # [3,H,W] float32
        mask = torch.from_numpy(
            np.array(mask, dtype=np.float32) / 255.0
        ).unsqueeze(0)                                                        # [1,H,W]
        mask = (mask > 0.5).float()

        image = TF.normalize(image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        return image, mask


def get_loader(image_dir, mask_dir, split_file, image_size=224,
               batch_size=4, num_workers=4, mode='train',
               augmentation_prob=0.4):
    dataset = SegmentationDataset(
        image_dir, mask_dir, split_file,
        image_size=image_size, mode=mode,
        augmentation_prob=augmentation_prob,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(mode == 'train'),
    )
