import torch
import torch.nn as nn


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        lr=0.001,
        device="cpu"
    ):
        self.device = torch.device(device)

        # Move model to device FIRST
        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = nn.CrossEntropyLoss()

        # IMPORTANT: use moved model parameters
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr
        )

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

            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = correct / total

        return total_loss / len(self.train_loader), acc

    def evaluate(self):
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

        return correct / total