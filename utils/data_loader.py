import os
import pandas as pd
import numpy as np
import torch
from PIL import Image

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]

def load_label_vector(finding_labels):
    vector = np.zeros(len(LABELS), dtype=np.float32)
    for label in finding_labels.split('|'):
        if label in LABELS:
            vector[LABELS.index(label)] = 1
    return vector
class ChestXrayMultiTaskDataset(Dataset):
    def __init__(self, img_dir, meta_file, bbox_file=None, split_file=None, transform=None):
        self.img_dir = img_dir
        self.meta_data = pd.read_csv(meta_file)
        self.transform = transform

        if bbox_file:
            self.bbox_data = pd.read_csv(bbox_file)
            self.bbox_data = self.bbox_data.set_index("Image Index")
        else:
            self.bbox_data = None

        if split_file:
            if isinstance(split_file, str):  # filepath로 주어졌을때
                with open(split_file, 'r') as f:
                    split_images = set(f.read().splitlines())
            elif isinstance(split_file, list):  # for train/val split
                split_images = set(split_file)
            else:
                raise ValueError("split_file must be either a file path or a list of image indices.")
            
            # Filter metadata based on split images
            self.meta_data = self.meta_data[self.meta_data["Image Index"].isin(split_images)]

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        row = self.meta_data.iloc[idx]
        img_path = os.path.join(self.img_dir, row["Image Index"])
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        label_vector = load_label_vector(row["Finding Labels"])

        if self.bbox_data is not None and row["Image Index"] in self.bbox_data.index:
            bboxes = self.bbox_data.loc[[row["Image Index"]]]
            bbox_list = bboxes[["x", "y", "w", "h"]].values
            bbox_tensor = torch.tensor([[x, y, x + w, y + h] for x, y, w, h in bbox_list], dtype=torch.float32)
        else:
            bbox_tensor = torch.zeros((0, 4), dtype=torch.float32)

        return image, bbox_tensor, label_vector



def get_transforms(input_size=224):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def create_dataloaders(
    img_dir, meta_file, bbox_file=None, train_split=None, test_split=None,
    batch_size=32, input_size=224, train_ratio=0.9
):
    transform = get_transforms(input_size)

    with open(train_split, 'r') as f:
        train_images = f.read().splitlines()

    np.random.shuffle(train_images)
    split_idx = int(len(train_images) * train_ratio)
    train_subset = train_images[:split_idx]
    val_subset = train_images[split_idx:]

    train_dataset = ChestXrayMultiTaskDataset(
        img_dir=img_dir,
        meta_file=meta_file,
        bbox_file=bbox_file,
        split_file=train_subset,
        transform=transform
    )

    val_dataset = ChestXrayMultiTaskDataset(
        img_dir=img_dir,
        meta_file=meta_file,
        bbox_file=bbox_file,
        split_file=val_subset,
        transform=transform
    )

    test_dataset = ChestXrayMultiTaskDataset(
        img_dir=img_dir,
        meta_file=meta_file,
        bbox_file=bbox_file,
        split_file=test_split,
        transform=transform
    )

    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    }

    return dataloaders
