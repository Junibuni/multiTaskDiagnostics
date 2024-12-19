import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from src.unified_model import UnifiedModel
from utils.data_loader import create_dataloaders
from tqdm import tqdm

def train_model(model, dataloaders, criterion_class, criterion_box, optimizer, device, num_epochs=10):
    model.to(device)
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_class_loss = 0.0
            running_box_loss = 0.0
            
            for inputs, bboxes, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                bboxes = [bbox.to(device) for bbox in bboxes]

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    classification_output, detection_boxes, detection_scores = model(inputs)
                    
                    class_loss = criterion_class(classification_output, labels)

                    # Detection loss (bbox 있는 image만)
                    box_loss = 0.0
                    if any(bbox.shape[0] > 0 for bbox in bboxes):
                        box_predictions = detection_boxes.view(-1, 4)
                        box_targets = torch.cat([bbox for bbox in bboxes if bbox.shape[0] > 0])
                        box_loss = criterion_box(box_predictions, box_targets)

                    loss = class_loss + box_loss

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_class_loss += class_loss.item() * inputs.size(0)
                running_box_loss += box_loss.item() * inputs.size(0) if box_loss > 0 else 0

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_class_loss = running_class_loss / len(dataloaders[phase].dataset)
            epoch_box_loss = running_box_loss / len(dataloaders[phase].dataset) if running_box_loss > 0 else 0

            print(f"{phase} Loss: {epoch_loss:.4f} | Class Loss: {epoch_class_loss:.4f} | Box Loss: {epoch_box_loss:.4f}")

    print("Training complete")


def main():
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

    model = UnifiedModel(num_classes=14, num_detection_classes=1)

    criterion_class = nn.BCELoss()
    criterion_box = nn.SmoothL1Loss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_model(model, dataloaders, criterion_class, criterion_box, optimizer, device, num_epochs=100)
    
if __name__ == "__main__":
    main()
