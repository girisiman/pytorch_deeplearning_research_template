import os
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from torchvision import transforms
from torch.utils.data import DataLoader

from src.data.dataset import ImageClassificationDataset
from src.models.baseline_model import SimpleCNN
from src.training.trainer import Trainer
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():

    # =========================================================
    # 1. EXPERIMENT ID
    # =========================================================
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\nStarting Experiment: {experiment_id}\n")

    # =========================================================
    # 2. PROJECT + OUTPUT STRUCTURE
    # =========================================================
    PROJECT_ROOT = Path(__file__).resolve().parent

    experiment_dir = PROJECT_ROOT / "outputs" / "experiments" / experiment_id
    plots_dir = experiment_dir / "plots"
    models_dir = experiment_dir / "models"

    plots_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # 3. DATA PATHS
    # =========================================================
    train_dir = PROJECT_ROOT / "raw_data" / "FruitinAmazon" / "train"
    val_dir   = PROJECT_ROOT / "raw_data" / "FruitinAmazon" / "val"

    # =========================================================
    # 4. TRANSFORMS
    # =========================================================
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # =========================================================
    # 5. DATASETS + LOADERS
    # =========================================================
    train_dataset = ImageClassificationDataset(str(train_dir), transform)
    val_dataset   = ImageClassificationDataset(str(val_dir), transform)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # =========================================================
    # 6. MODEL
    # =========================================================
    model = SimpleCNN(num_classes=len(train_dataset.classes))

    # =========================================================
    # 7. TRAINER
    # =========================================================
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=0.001,
        device="cpu"
    )

    # =========================================================
    # 8. TRAINING CONFIG
    # =========================================================
    num_epochs = 20
    patience = 3
    counter = 0
    best_val_acc = 0.0

    train_losses = []
    train_accs = []
    val_accs = []

    # =========================================================
    # 9. TRAINING LOOP
    # =========================================================
    for epoch in range(num_epochs):

        train_loss, train_acc = trainer.train_one_epoch()
        val_acc = trainer.evaluate()

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print("\n" + "=" * 50)
        print(f"Experiment: {experiment_id}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("=" * 50)
        print(f"Train Loss : {train_loss:.4f}")
        print(f"Train Acc  : {train_acc:.4f}")
        print(f"Val Acc    : {val_acc:.4f}")

        # =====================================================
        # CHECKPOINTING
        # =====================================================
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0

            model_path = models_dir / "best_model.pth"
            torch.save(model.state_dict(), model_path)

            print(f"Saved best model: {model_path}")

        else:
            counter += 1

        # =====================================================
        # EARLY STOPPING
        # =====================================================
        if counter >= patience:
            print("Early stopping triggered")
            break

    # =========================================================
    # 10. SAVE TRAINING CURVES
    # =========================================================
    plt.figure()

    plt.plot(train_losses, label="Train Loss")
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")

    plt.legend()
    plt.title(f"Training Curves - {experiment_id}")

    plot_path = plots_dir / "training_curves.png"

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved training curves at: {plot_path}")
    print(f"Experiment completed: {experiment_id}\n")


if __name__ == "__main__":
    main()