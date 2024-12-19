from scripts.train import main

img_dir = "data/raw/images"
meta_file = "data/raw/Data_Entry_2017_v2020.csv"
bbox_file = "data/raw/BBox_List_2017.csv"
train_split = "data/raw/data_train_val_list.txt"
test_split = "data/raw/data_test_list.txt"

main()