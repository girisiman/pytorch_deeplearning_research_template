from src.models.baseline_model import SimpleCNN
from src.models.wider_model import WiderCNN
from src.models.dropout_model import DropoutCNN


def get_model(model_name, num_classes=10):
    if model_name == "baseline":
        return SimpleCNN(num_classes=num_classes)

    elif model_name == "wider":
        return WiderCNN(num_classes=num_classes)

    elif model_name == "dropout":
        return DropoutCNN(num_classes=num_classes)

    else:
        raise ValueError(f"Unknown model: {model_name}")