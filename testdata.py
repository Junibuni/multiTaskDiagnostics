from utils.data_loader import create_dataloaders
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

img_dir = "data/raw/images"
meta_file = "data/raw/Data_Entry_2017_v2020.csv"
bbox_file = "data/raw/BBox_List_2017.csv"
train_split = "data/raw/data_train_val_list.txt"
test_split = "data/raw/data_test_list.txt"

dataloaders = create_dataloaders(
    img_dir=img_dir,
    meta_file=meta_file,
    bbox_file=bbox_file,
    train_split=train_split,
    test_split=test_split,
    batch_size=128,
    input_size=224
    )

train_dataloader = dataloaders['train']

from collections import Counter

def verify_oversampling(dataloader):
    bbox_counts = Counter()

    for batch_idx, (images, bboxes, labels) in enumerate(dataloader):
        print(batch_idx)

        for bbox in bboxes:
            if bbox.size(0) > 0: 
                bbox_counts["with_bbox"] += 1
            else:
                bbox_counts["without_bbox"] += 1
        if batch_idx >= 100:
            break

    print("Sample distribution in the first 10 batches:")
    print(f"With bounding boxes: {bbox_counts['with_bbox']}")
    print(f"Without bounding boxes: {bbox_counts['without_bbox']}")
    total = bbox_counts['with_bbox'] + bbox_counts['without_bbox']
    print(f"Percentage with bounding boxes: {100 * bbox_counts['with_bbox'] / total:.2f}%")
    print(f"Percentage without bounding boxes: {100 * bbox_counts['without_bbox'] / total:.2f}%")
    
print("in")
verify_oversampling(train_dataloader)
# for batch in train_dataloader:
#     image, bbox, label = batch
#     bbox = bbox[0]
#     if len(bbox) > 0:
#         print(f"Found non-empty bbox: {bbox}")
#         image = image[0].numpy()
#         break

# fig, ax = plt.subplots(1)
# ax.imshow(image)

# for box in bbox:
#     x, y, w, h = box # [x_min, y_min, x_max, y_max]
#     rect = patches.Rectangle((x, y), w - x, h - y, linewidth=2, edgecolor='r', facecolor='none')
#     ax.add_patch(rect)

# plt.title(f"Label: {label[0]}")
# plt.show()