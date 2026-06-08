from src.data.dataloader import get_dataloader
from torchvision import transforms

# Path to your dataset
train_dir = r"raw_data\FruitinAmazon\train"

# Simple transform (no augmentation yet)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Create DataLoader
train_loader = get_dataloader(
    root_directory=train_dir,
    transform=transform,
    batch_size=4,
    shuffle=True
)

# ---- TEST LOOP ----
print("Testing DataLoader...\n")

for batch_idx, (images, labels) in enumerate(train_loader):
    print(f"Batch {batch_idx}")
    print("Images shape:", images.shape)   # (B, C, H, W)
    print("Labels:", labels)
    
    # only check first 2 batches
    if batch_idx == 1:
        break

print("\n DataLoader works correctly!")