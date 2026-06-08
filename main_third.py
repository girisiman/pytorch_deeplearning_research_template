import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from torchvision import transforms
from torch.utils.data import DataLoader

from src.data.dataset import ImageClassificationDataset
from src.models.baseline_model import SimpleCNN
from src.training.trainer import Trainer
from src.utils.config_loader import load_config
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():

    # =========================================================
    # 1. LOAD CONFIG
    # =========================================================
    config_path = Path(__file__).resolve().parent / "config" / "config.yaml"
    cfg = load_config(config_path)

    # =========================================================
    # 2. EXPERIMENT ID
    # =========================================================
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\nRunning Config Experiment: {cfg['experiment']['name']}")
    print(f"Experiment ID: {experiment_id}\n")

    # =========================================================
    # 3. OUTPUT STRUCTURE
    # =========================================================
    PROJECT_ROOT = Path(__file__).resolve().parent

    experiment_dir = PROJECT_ROOT / "outputs" / "experiments" / experiment_id
    plots_dir = experiment_dir / "plots"
    models_dir = experiment_dir / "models"

    plots_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # 4. DATA
    # =========================================================
    train_dir = cfg["data"]["train_dir"]
    val_dir = cfg["data"]["val_dir"]

    transform = transforms.Compose([
        transforms.Resize((cfg["data"]["image_size"], cfg["data"]["image_size"])),
        transforms.ToTensor()
    ])

    train_dataset = ImageClassificationDataset(train_dir, transform)
    val_dataset = ImageClassificationDataset(val_dir, transform)

    train_loader = DataLoader(train_dataset, batch_size=cfg["data"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg["data"]["batch_size"], shuffle=False)

    # =========================================================
    # 5. MODEL
    # =========================================================
    model = SimpleCNN(num_classes=len(train_dataset.classes))

    # =========================================================
    # 6. TRAINER
    # =========================================================
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=cfg["training"]["learning_rate"],
        device=cfg["training"]["device"]
    )

    # =========================================================
    # 7. TRAINING LOOP
    # =========================================================
    num_epochs = cfg["training"]["epochs"]
    patience = cfg["training"]["patience"]

    best_val_acc = 0.0
    counter = 0

    train_losses, train_accs, val_accs = [], [], []

    for epoch in range(num_epochs):

        train_loss, train_acc = trainer.train_one_epoch()
        val_acc = trainer.evaluate()

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print("\n" + "=" * 50)
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("=" * 50)
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Acc : {train_acc:.4f}")
        print(f"Val Acc   : {val_acc:.4f}")

        # CHECKPOINT
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0

            model_path = models_dir / "best_model.pth"
            torch.save(model.state_dict(), model_path)
            print(f"Saved model: {model_path}")

        else:
            counter += 1

        # EARLY STOPPING
        if counter >= patience:
            print("Early stopping triggered")
            break

    # =========================================================
    # 8. SAVE PLOTS
    # =========================================================
    plt.figure()

    plt.plot(train_losses, label="Train Loss")
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")

    plt.legend()
    plt.title(f"Training Curves - {experiment_id}")

    plot_path = plots_dir / "training_curves.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved plot at: {plot_path}")
    print("Experiment finished.")


if __name__ == "__main__":
    main()