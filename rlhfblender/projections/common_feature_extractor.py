import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import transforms


class RemoveFirstChannel:
    def __init__(self):
        pass

    def __call__(self, x):
        return x[:3]


class ResnetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        resnet = resnet18(weights=weights, progress=False).eval()
        self.model = nn.Sequential(*list(resnet.children())[:-1])
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                RemoveFirstChannel(),
                transforms.Resize(224),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # default for imagenet
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            print("x Shape", x.shape)
            x = [self.transforms(img) for img in x]
            # Get features from the last layer of the ResNet18
            features = self.model(torch.stack(x))
            print("Features Shape", features.shape)
            return features
