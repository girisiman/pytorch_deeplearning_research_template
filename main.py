from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader

from src.data.dataset import ImageClassificationDataset
from src.models.baseline_model import SimpleCNN
from src.training.trainer import Trainer


def main():

    # ------------------------
    # 1. Project Paths
    # ------------------------
    PROJECT_ROOT = Path(__file__).resolve().parent

    train_dir = PROJECT_ROOT / "raw_data" / "FruitinAmazon" / "train"
    val_dir   = PROJECT_ROOT / "raw_data" / "FruitinAmazon" / "val"

    print("Train path:", train_dir)
    print("Val path:", val_dir)

    # ------------------------
    # 2. Transform
    # ------------------------
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # ------------------------
    # 3. Dataset
    # ------------------------
    train_dataset = ImageClassificationDataset(
        root_dir=str(train_dir),
        transform=transform
    )

    val_dataset = ImageClassificationDataset(
        root_dir=str(val_dir),
        transform=transform
    )

    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))

    # ------------------------
    # 4. DataLoader
    # ------------------------
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # ------------------------
    # 5. Model
    # ------------------------
    model = SimpleCNN(num_classes=len(train_dataset.classes))

    print(model)

    # ------------------------
    # 6. Trainer
    # ------------------------
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=0.001,
        device="cpu"   # change to "cuda" if available
    )

    # ------------------------
    # 7. Training Loop
    # ------------------------
    num_epochs = 5

    for epoch in range(num_epochs):
        train_loss, train_acc = trainer.train_one_epoch()
        val_acc = trainer.evaluate()

        print("\n" + "=" * 40)
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("=" * 40)
        print(f"Train Loss : {train_loss:.4f}")
        print(f"Train Acc  : {train_acc:.4f}")
        print(f"Val Acc    : {val_acc:.4f}")
        print("=" * 40)


if __name__ == "__main__":
    main()