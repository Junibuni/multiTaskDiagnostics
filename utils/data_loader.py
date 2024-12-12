import os

import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, task='classification'):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.task = task

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            row = self.data.iloc[idx]
        else:
            row = self.data.loc[idx]
        
        img_path = os.path.join(self.img_dir, row['Image Index'])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = row['Finding Labels']
        
        if self.task == 'classification':
            # Convert label to binary format (e.g., 0 for normal, 1 for abnormal)
            label = 1 if 'No Finding' not in label else 0
            return image, label
        elif self.task == 'segmentation':
            # Placeholder: Load segmentation mask (if available)
            mask = np.zeros(image.size[::-1], dtype=np.uint8)  # Dummy mask
            return image, mask
        elif self.task == 'detection':
            # Placeholder: Return bounding box data (if available)
            bbox = np.array([0, 0, image.size[0], image.size[1]])  # Dummy bbox
            return image, bbox
        else:
            raise ValueError(f"Unknown task type: {self.task}")

def get_transforms(input_size=224):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def create_dataloaders(csv_file, img_dir, batch_size=32, task='classification', input_size=224, split_ratios=(0.7, 0.2, 0.1)):
    data = pd.read_csv(csv_file)
    
    data = data.sample(frac=1).reset_index(drop=True)
    
    train_end = int(len(data) * split_ratios[0])
    val_end = train_end + int(len(data) * split_ratios[1])
    
    train_data = data.iloc[:train_end]
    val_data = data.iloc[train_end:val_end]
    test_data = data.iloc[val_end:]
    
    transform = get_transforms(input_size)
    
    train_dataset = ChestXrayDataset(train_data, img_dir, transform, task)
    val_dataset = ChestXrayDataset(val_data, img_dir, transform, task)
    test_dataset = ChestXrayDataset(test_data, img_dir, transform, task)
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    }
    
    return dataloaders
