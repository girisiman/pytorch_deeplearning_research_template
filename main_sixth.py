import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from torchvision import transforms
from torch.utils.data import DataLoader

import os
import argparse
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.data.dataset import ImageClassificationDataset
from src.training.trainer import Trainer
from src.utils.config_loader import load_config
from src.models.build_model import get_model


def main():

    # =========================================================
    # 1. ARGPARSE (CONFIG SELECTION)
    # =========================================================
    parser = argparse.ArgumentParser(description="PyTorch Training Pipeline")

    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to YAML config file"
    )

    args = parser.parse_args()

    # =========================================================
    # 2. LOAD CONFIG
    # =========================================================
    cfg = load_config(args.config)

    # =========================================================
    # 3. EXPERIMENT TRACKING
    # =========================================================
    experiment_name = cfg["experiment"]["name"]

    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("runs") / experiment_name

    print("\n====================================")
    print(" TensorBoard Experiment Started")
    print("====================================")
    print(f"Config Path    : {args.config}")
    print(f"Experiment Name: {experiment_name}")
    print(f"Experiment ID  : {experiment_id}")
    print(f"Log Directory  : {log_dir}")
    print("====================================\n")

    # =========================================================
    # 4. DEVICE
    # =========================================================
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Using device:", device)

    # =========================================================
    # 5. DATA
    # =========================================================
    transform = transforms.Compose([
        transforms.Resize((
            cfg["data"]["image_size"],
            cfg["data"]["image_size"]
        )),
        transforms.ToTensor()
    ])

    train_dataset = ImageClassificationDataset(
        cfg["data"]["train_dir"],
        transform
    )

    val_dataset = ImageClassificationDataset(
        cfg["data"]["val_dir"],
        transform
    )

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
    # 6. MODEL (ABALATION FACTORY)
    # =========================================================
    model = get_model(
        cfg["model"]["name"],
        num_classes=len(train_dataset.classes)
    )

    # =========================================================
    # 7. TRAINER (WITH TENSORBOARD)
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
    # 8. TRAINING LOOP
    # =========================================================
    num_epochs = cfg["training"]["epochs"]

    train_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(num_epochs):

        train_loss, train_acc = trainer.train_one_epoch()
        val_acc = trainer.evaluate(epoch)

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

        trainer.writer.flush()

    # =========================================================
    # 9. CLOSE WRITER
    # =========================================================
    trainer.writer.close()

    print("\n====================================")
    print("Training Completed")
    print("====================================")
    print(f"Run TensorBoard with: tensorboard --logdir runs")
    print("====================================\n")


# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    main()