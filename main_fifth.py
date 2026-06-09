import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from torchvision import transforms
from torch.utils.data import DataLoader

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.data.dataset import ImageClassificationDataset
from src.models.baseline_model import SimpleCNN
from src.training.trainer import Trainer
from src.utils.config_loader import load_config


def main():

    # =========================================================
    # 1. LOAD CONFIG
    # =========================================================
    config_path = Path(__file__).resolve().parent / "config" / "config.yaml"
    cfg = load_config(config_path)

    # =========================================================
    # 2. EXPERIMENT TRACKING
    # =========================================================
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("runs") / experiment_id

    print("\n====================================")
    print(" TensorBoard Experiment Started")
    print("====================================")
    print(f"Experiment ID : {experiment_id}")
    print(f"Log Directory  : {log_dir}")
    print("====================================\n")

    # =========================================================
    # 3. DEVICE
    # =========================================================
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Using device:", device)

    # =========================================================
    # 4. DATA
    # =========================================================
    train_dir = cfg["data"]["train_dir"]
    val_dir = cfg["data"]["val_dir"]

    transform = transforms.Compose([
        transforms.Resize((
            cfg["data"]["image_size"],
            cfg["data"]["image_size"]
        )),
        transforms.ToTensor()
    ])

    train_dataset = ImageClassificationDataset(train_dir, transform)
    val_dataset = ImageClassificationDataset(val_dir, transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False
    )

    # =========================================================
    # 5. MODEL
    # =========================================================
    model = SimpleCNN(num_classes=len(train_dataset.classes))

    # =========================================================
    # 6. TRAINER (WITH TENSORBOARD)
    # =========================================================
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=cfg["training"]["learning_rate"],
        device=device,
        log_dir=str(log_dir)
    )

    # =========================================================
    # 7. TRAINING LOOP
    # =========================================================
    num_epochs = cfg["training"]["epochs"]

    train_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(num_epochs):

        train_loss, train_acc = trainer.train_one_epoch()
        val_acc = trainer.evaluate(epoch)

        # Store locally (optional for plotting later)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # =====================================================
        # PRINT LOGS
        # =====================================================
        print("\n====================================")
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print("====================================")
        print(f"Train Loss : {train_loss:.4f}")
        print(f"Train Acc  : {train_acc:.4f}")
        print(f"Val Acc    : {val_acc:.4f}")

        # =====================================================
        # TENSORBOARD LOGGING
        # =====================================================
        trainer.writer.add_scalar("Train/Loss", train_loss, epoch)
        trainer.writer.add_scalar("Train/Accuracy", train_acc, epoch)
        trainer.writer.add_scalar("Val/Accuracy", val_acc, epoch)

    # =========================================================
    # 8. CLOSE WRITER
    # =========================================================
    trainer.writer.close()

    print("\nTraining completed.")
    print(f"Run TensorBoard with: tensorboard --logdir runs")


if __name__ == "__main__":
    main()