# Define and return the ViT model
import torch
import torch.nn as nn
import torchvision


def get_model(
    num_classes: int = 10, num_of_layers_to_unfreeze: int = 2, device="cpu"
) -> nn.Module:
    weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    model = torchvision.models.vit_b_16(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.heads.head.in_features
    model.heads = nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(in_features=num_features, out_features=num_classes, bias=True),
    ).to(device)
    for param in model.heads.parameters():
        param.requires_grad = True
    num_of_layers_to_unfreeze = len(model.encoder.layers) // 6
    for layer in model.encoder.layers[-num_of_layers_to_unfreeze:]:
        for param in layer.parameters():
            param.requires_grad = True
    return model.to(device)
