import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data import get_datasets
from model import get_model
from modules.going_modular import engine

BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()
NUM_CLASSES = 10
NUM_EPOCHS = 10


def main():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"

    train_dataset, val_dataset, test_dataset = get_datasets()
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = get_model(num_classes=NUM_CLASSES, device=device)
    params_to_optimize = [
        {"params": model.heads.parameters(), "lr": 1e-3},
    ]

    num_of_layers_to_unfreeze = len(model.encoder.layers) // 2
    for layer in model.encoder.layers[-num_of_layers_to_unfreeze:]:
        params_to_optimize.append({"params": layer.parameters(), "lr": 1e-6})
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=1e-3,
        weight_decay=1e-2,
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    writer = SummaryWriter("runs/urbansound8k")

    model_results = engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=NUM_EPOCHS,
        device=device,
        writer=writer,
    )

if __name__ == "__main__":
    main()