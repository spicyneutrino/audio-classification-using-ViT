import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from scripts.data import get_datasets
from scripts.model import get_model
from modules.going_modular import engine

NUM_CLASSES = 10


def main(num_epochs: int, num_workers: int, batch_size: int):
    device_name = "cpu"
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        device_name = "cuda"
    elif torch.backends.mps.is_available():
        device_name = "mps"
    device = torch.device(device_name)
    print(f"Using device: {device}")
    print(f"Number of workers: {num_workers}")
    print(f"Batch size: {batch_size}")

    train_dataset, val_dataset, test_dataset = get_datasets()
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    head_lr = 3e-4
    encoder_lr = 3e-5 
    model = get_model(num_classes=NUM_CLASSES, device=device)
    params_to_optimize = [
        {"params": model.heads.parameters(), "lr": head_lr},
    ]

    num_of_layers_to_unfreeze = len(model.encoder.layers) // 3
    for layer in model.encoder.layers[-num_of_layers_to_unfreeze:]:
        params_to_optimize.append({"params": layer.parameters(), "lr": encoder_lr})
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=1e-3,
        weight_decay=1e-2,
    )
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    writer = SummaryWriter("runs/urbansound8k")

    model_results = engine.train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=num_epochs,
        device=device,
        writer=writer,
    )
    # Test the model on unseen data
    # Load the best model
    best_model_path = os.path.join("checkpoints", "best_model_checkpoint.pth")
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Best model loaded successfully.")
    else:
        print(
            f"Best model not found at {best_model_path}. Evaluating the model from last epoch."
        )

    print("\nEvaluating best model on Test Dataset...")
    test_loss, test_acc = engine.test_step(
        model=model,
        dataloader=test_dataloader,
        loss_fn=loss_fn,
        device=device,
    )
    print("\n--- Final Test Results ---")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of epochs to train the model",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=os.cpu_count() // 2 if os.cpu_count() > 1 else 0,
        help="Number of workers for the data loaders",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of workers for the data loaders",
    )
    args = parser.parse_args()
    main(args.num_epochs, args.num_workers, args.batch_size)
