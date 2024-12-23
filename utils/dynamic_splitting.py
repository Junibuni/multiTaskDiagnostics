import pandas as pd
import numpy as np

def adjust_split(train_file, test_file, bbox_file, train_ratio=0.9):
    with open(train_file, 'r') as f:
        train_images = f.read().splitlines()
    with open(test_file, 'r') as f:
        test_images = f.read().splitlines()

    bbox_data = pd.read_csv(bbox_file)
    bbox_images = set(bbox_data["Image Index"])

    train_with_bbox = [img for img in train_images if img in bbox_images]
    train_without_bbox = [img for img in train_images if img not in bbox_images]

    num_to_add = int(len(train_with_bbox) * train_ratio)

    new_train_images = train_with_bbox[:num_to_add] + train_without_bbox
    new_test_images = train_with_bbox[num_to_add:] + test_images

    return new_train_images, new_test_images

    """
    new_train_images, new_test_images = adjust_split(
    "data_train_val_list.txt",
    "data_test_list.txt",
    "BBox_List_2017.csv",
    train_ratio=0.8
)

with open("new_train_val_list.txt", 'w') as f:
    f.write("\n".join(new_train_images))
with open("new_test_list.txt", 'w') as f:
    f.write("\n".join(new_test_images))
    """
