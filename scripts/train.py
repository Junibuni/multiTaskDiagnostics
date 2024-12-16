import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from utils.data_loader import create_dataloaders
from src.unified_head import UnifiedModel

def segmentation_loss_fn(pred, target):
    smooth = 1e-6
    intersection = (pred * target).sum()
    dice_loss = 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice_loss

def detection_loss_fn(pred, target):
    return nn.SmoothL1Loss()(pred, target)

def classification_loss_fn(pred, target):
    return nn.BCEWithLogitsLoss()(pred, target)

def compute_total_loss(seg_loss, det_loss, cls_loss, seg_weight=1.0, det_weight=1.0, cls_weight=1.0):
    return seg_weight * seg_loss + det_weight * det_loss + cls_weight * cls_loss

def train_epoch(model, dataloader, optimizer, device, epoch, writer=None):
    model.train()
    epoch_loss = 0

    for batch_idx, (images, seg_targets, det_targets, cls_targets) in enumerate(dataloader):
        images, seg_targets, det_targets, cls_targets = (
            images.to(device),
            seg_targets.to(device),
            det_targets.to(device),
            cls_targets.to(device),
        )

        optimizer.zero_grad()

        seg_output, det_output, cls_output = model(images)

        seg_loss = segmentation_loss_fn(seg_output, seg_targets)
        det_loss = detection_loss_fn(det_output, det_targets)
        cls_loss = classification_loss_fn(cls_output, cls_targets)
        total_loss = compute_total_loss(seg_loss, det_loss, cls_loss)

        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()

        if writer:
            writer.add_scalar("Batch Loss", total_loss.item(), epoch * len(dataloader) + batch_idx)

    return epoch_loss / len(dataloader)

def validate_epoch(model, dataloader, device):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for images, seg_targets, det_targets, cls_targets in dataloader:
            images, seg_targets, det_targets, cls_targets = (
                images.to(device),
                seg_targets.to(device),
                det_targets.to(device),
                cls_targets.to(device),
            )

            seg_output, det_output, cls_output = model(images)

            seg_loss = segmentation_loss_fn(seg_output, seg_targets)
            det_loss = detection_loss_fn(det_output, det_targets)
            cls_loss = classification_loss_fn(cls_output, cls_targets)
            total_loss = compute_total_loss(seg_loss, det_loss, cls_loss)

            epoch_loss += total_loss.item()

    return epoch_loss / len(dataloader)

def train():
    batch_size = 16
    num_epochs = 20
    learning_rate = 1e-4
    checkpoint_dir = "outputs/models/"
    os.makedirs(checkpoint_dir, exist_ok=True)

    csv_file = "data/annotations/ChestX-ray14_labels.csv"
    img_dir = "data/raw/images/"
    checkpoint_path = "checkpoints/sam_vit_h_4b8939.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloaders = create_dataloaders(csv_file, img_dir, batch_size=batch_size, task="multi-task")
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    model = UnifiedModel(sam_checkpoint=checkpoint_path, backbone_type="vit_h", num_classes=1, num_detection_classes=1)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    writer = SummaryWriter(log_dir="outputs/logs/")

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, writer)
        val_loss = validate_epoch(model, val_loader, device)

        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pth"))

    writer.close()

if __name__ == "__main__":
    train()
