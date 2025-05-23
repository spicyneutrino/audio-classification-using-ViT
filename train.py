import argparse
import os
import datetime
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from scripts.data import get_datasets
from scripts.model import get_model
from scripts import engine
import numpy as np

NUM_CLASSES = 10


def main(
    num_epochs: int,
    num_workers: int,
    batch_size: int,
    head_lr: float,
    encoder_lr: float,
    use_time_augment: bool,
    num_unfrozen_encoder_layers: int,
):

    random_seed = 42
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

    job_id = os.environ.get("SLURM_JOBID")
    now = datetime.datetime.now()
    if job_id:
        run_identifier = f"slurm_{job_id}_{now}"
    else:
        timestamp = now.strftime("%m-%d_%H-%M-%S")
        run_identifier = f"local_{timestamp}"

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_filename = f"best_model_{run_identifier}.pth"
    best_model_path = os.path.join(checkpoint_dir, best_model_filename)

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

    head_lr = 2e-4
    encoder_lr = 1e-5

    model = get_model(
        num_classes=NUM_CLASSES,
        num_of_layers_to_unfreeze=num_unfrozen_encoder_layers,
        device=device,
    )
    params_to_optimize = [
        {"params": model.heads.parameters(), "lr": head_lr},
    ]

    num_of_layers_to_unfreeze = num_unfrozen_encoder_layers
    for layer in model.encoder.layers[-num_of_layers_to_unfreeze:]:
        params_to_optimize.append({"params": layer.parameters(), "lr": encoder_lr})
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=1e-3,
        weight_decay=5e-2,
    )
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    # Initialize the LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=num_epochs, eta_min=1e-6
    )

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
        best_model_save_path=best_model_path,
        scheduler=scheduler,
    )
    # Test the model on unseen data
    # Load the best model

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
    parser.add_argument(
        "--head_lr",
        type=float,
        default=8e-5,
        help="Learning rate for the head/classifier of the model",
    )
    parser.add_argument(
        "--encoder_lr",
        type=float,
        default=1e-5,
        help="Learning rate for unfrozen encoder layers of the model",
    )
    parser.add_argument(
        "--use_time_augment",
        type=bool,
        default=True,
        help="Use time augmentations for training",
    )
    parser.add_argument(
        "--num_unfrozen_encoder_layers",
        type=int,
        default=2,
        help="Number of encoder layers to unfreeze for training",
    )
    args = parser.parse_args()
    main(
        args.num_epochs,
        args.num_workers,
        args.batch_size,
        args.head_lr,
        args.encoder_lr,
        args.use_time_augment,
        args.num_unfrozen_encoder_layers,
    )
    # main(num_epochs=10, num_workers=4, batch_size=32, head_lr=8e-5, encoder_lr=1e-5, use_time_augment=True, unfreeze_encoder_layers=2)
