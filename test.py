from utils.data_loader import create_dataloaders

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
    batch_size=16,
    input_size=224
)

for images, bboxes, labels in dataloaders["train"]:
    # print(f"Images: {images.shape}")
    # print(f"Bounding Boxes: {bboxes}")
    # print(f"Labels: {labels.shape}")
    pass
