import torch.nn as nn
from torchvision import models

MODELS = [
    {"model_name": "resnet", "size": 64},
    {"model_name": "vgg", "size": 64},
    {"model_name": "efficientnet", "size": 64},
    {"model_name": "mobilenet", "size": 64},
]

def get_model(name, num_classes, dropout, device):

    if name == "resnet":
        model = models.resnet18(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(model.fc.in_features, num_classes)
        )

    elif name == "vgg":
        model = models.vgg11(weights=None)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[0].in_features, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(4096, num_classes)
        )

    elif name == "efficientnet":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )

    elif name == "mobilenet":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )

    else:
        raise ValueError()

    return model.to(device)