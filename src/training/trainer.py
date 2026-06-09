import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import time
class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        lr=0.001,
        device="cpu",
        log_dir="runs/experiment_1"
    ):
        self.device = torch.device(device)

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr
        )

        # TensorBoard
        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0

        print(f"TensorBoard logging at: {log_dir}")

    # =========================================================
    # LIVE TRAINING (BATCH LEVEL LOGGING)
    # =========================================================
    def train_one_epoch(self):
        self.model.train()

        total_loss = 0
        correct = 0
        total = 0

        for images, labels in self.train_loader:

            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # -----------------------------
            # LIVE LOGGING (IMPORTANT FIX)
            # -----------------------------
            self.writer.add_scalar(
                "Train/Loss",
                loss.item(),
                self.global_step
            )
            # LIVE DEMO SLOWDOWN (PUT IT HERE)
            time.sleep(1)

            _, preds = torch.max(outputs, 1)
            batch_acc = (preds == labels).sum().item() / labels.size(0)

            self.writer.add_scalar(
                "Train/BatchAccuracy",
                batch_acc,
                self.global_step
            )

            self.writer.flush()

            # -----------------------------
            # tracking
            # -----------------------------
            total_loss += loss.item()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            self.global_step += 1

        epoch_acc = correct / total
        epoch_loss = total_loss / len(self.train_loader)

        # Optional: epoch summary
        self.writer.add_scalar("Train/EpochLoss", epoch_loss, self.global_step)
        self.writer.add_scalar("Train/EpochAccuracy", epoch_acc, self.global_step)

        return epoch_loss, epoch_acc

    # =========================================================
    # VALIDATION (EPOCH LEVEL)
    # =========================================================
    def evaluate(self, epoch):
        self.model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:

                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total

        # epoch-level logging
        self.writer.add_scalar(
            "Val/Accuracy",
            val_acc,
            epoch
        )

        self.writer.flush()

        return val_acc

    # =========================================================
    def close(self):
        self.writer.close()