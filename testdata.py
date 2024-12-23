from utils.data_loader import create_dataloaders
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from utils.dynamic_splitting import adjust_split
import torch

img_dir = "data/raw/images"
meta_file = "data/raw/Data_Entry_2017_v2020.csv"
bbox_file = "data/raw/BBox_List_2017.csv"
train_split = "data/raw/data_train_val_list.txt"
test_split = "data/raw/data_test_list.txt"

import pandas as pd
bbox_data = pd.read_csv(bbox_file)
print(f"Sample bounding box image indices: {bbox_data['Image Index'].head()}")
print(f"Total bounding box images: {len(bbox_data['Image Index'].unique())}")

with open(train_split, 'r') as f:
    train_images = set(f.read().splitlines())
with open(test_split, 'r') as f:
    test_images = set(f.read().splitlines())

bbox_images = set(bbox_data["Image Index"])

train_bbox_overlap = train_images.intersection(bbox_images)
test_bbox_overlap = test_images.intersection(bbox_images)

print(f"Number of bounding box images in train set: {len(train_bbox_overlap)}")
print(f"Number of bounding box images in test set: {len(test_bbox_overlap)}")

def redistribute_bbox_images(train_file, test_file, bbox_file, bbox_to_train_ratio=0.8):
    with open(train_file, 'r') as f:
        train_images = set(f.read().splitlines())
    with open(test_file, 'r') as f:
        test_images = set(f.read().splitlines())

    bbox_data = pd.read_csv(bbox_file)
    bbox_images = set(bbox_data["Image Index"])

    test_bbox_images = test_images.intersection(bbox_images)

    num_to_move = int(len(test_bbox_images) * bbox_to_train_ratio)
    bbox_images_to_move = list(test_bbox_images)[:num_to_move]

    new_train_images = train_images.union(bbox_images_to_move)
    new_test_images = test_images.difference(bbox_images_to_move)

    return list(new_train_images), list(new_test_images)

new_train_images, new_test_images = redistribute_bbox_images(
    train_split,
    test_split,
    bbox_file,
    bbox_to_train_ratio=0.8
)

with open("data/raw/new_train_val_list.txt", 'w') as f:
    f.write("\n".join(new_train_images))
with open("data/raw/new_test_list.txt", 'w') as f:
    f.write("\n".join(new_test_images))


with open("data/raw/new_train_val_list.txt", 'r') as f:
    new_train_images = set(f.read().splitlines())

bbox_images = set(bbox_data["Image Index"])
train_bbox_overlap = new_train_images.intersection(bbox_images)

print(f"Number of bounding box images in new training set: {len(train_bbox_overlap)}")


dataloaders = create_dataloaders(
    img_dir=img_dir,
    meta_file=meta_file,
    bbox_file=bbox_file,
    train_split="data/raw/new_train_val_list.txt",
    test_split="data/raw/new_test_list.txt",
    batch_size=128,
    input_size=224
    )

train_dataloader = dataloaders['train']

for batch in train_dataloader:
    images, bboxes, labels = batch
    for i, bbox in enumerate(bboxes):
        if not torch.all(bbox == -1):
            image = images[i]
            label = labels[i]
            break
    else:
        continue
    break

image = image.permute(1, 2, 0).numpy()

image = np.clip(image, 0, 1)

fig, ax = plt.subplots(1)
ax.imshow(image)
x_min, y_min, x_max, y_max = bbox.numpy()[0]
rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                         linewidth=2, edgecolor='r', facecolor='none')
ax.add_patch(rect)

plt.title(f"Label: {label[0]}")
plt.savefig("test.png")