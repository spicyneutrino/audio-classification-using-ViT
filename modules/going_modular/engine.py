"""
Contains functions for training and testing a PyTorch model.
"""

import torch
import datetime
import numpy as np
from torch.amp import GradScaler

import torch.utils.tensorboard
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import os

# from torch.utils.tensorboard import SummaryWriter


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for _, batch_dict in enumerate(dataloader):
        # Send data to target device
        X = batch_dict["pixel_values"].to(device)
        y = batch_dict["labels"].to(device)

        # Automatic mixed precision (AMP) context manager
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
            # 1. Forward pass
            y_pred = model(X)

            # 2. Calculate  and accumulate loss
            loss = loss_fn(y_pred, y)

        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        scaler.scale(loss).backward()

        # 5. Optimizer step
        # optimizer.step()
        scaler.step(optimizer)
        scaler.update()
        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode(), torch.amp.autocast(
        device_type=device.type, dtype=torch.float16
    ):
        # Loop through DataLoader batches
        for batch_idx, batch_dict in enumerate(dataloader):
            # Send data to target device
            X = batch_dict["pixel_values"].to(device)
            y = batch_dict["labels"].to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device,
    writer: torch.utils.tensorboard.SummaryWriter,
    save_dir: str = "checkpoints",
    patience: int = 10,
    best_model_save_path: str = "best_model.pth",
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
) -> Dict[str, List]:

    results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # It's good practice to trace in eval mode
    model.train()

    # Use automatic mixed precision (AMP) if CUDA is available
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # Initialize early stopping variables
    best_val_loss = np.inf
    epoch_no_improvement = 0
    # Create directory to save checkpoints
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f"Created directory: {save_dir}")

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
        )
        val_loss, val_acc = test_step(
            model=model, dataloader=val_dataloader, loss_fn=loss_fn, device=device
        )

        # Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_acc: {val_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

        writer.add_scalars(
            main_tag="Loss",
            tag_scalar_dict={"train_loss": train_loss, "val_loss": val_loss},
            global_step=epoch,
        )

        writer.add_scalars(
            main_tag="Accuracy",
            tag_scalar_dict={
                "train_acc": train_acc,
                "val_acc": val_acc,
            },
            global_step=epoch,
        )

        if scheduler:
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar(
                "Learning Rate",
                current_lr,
                epoch,
            )

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epoch_no_improvement = 0

            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }

            try:
                torch.save(checkpoint, best_model_save_path)
                print(f"Saved best model checkpoint to {best_model_save_path}")
            except Exception as e:
                print(f"Error saving best model checkpoint: {e}")
        else:
            epoch_no_improvement += 1
            print(f"Epoch {epoch + 1} - No improvement in validation loss.")
            if epoch_no_improvement >= patience:
                print(
                    f"Early stopping triggered after {patience} epochs without improvement."
                )
                break

    model.eval()
    try:
        dummy_input = torch.empty(1, 3, 224, 224).to(device)
        writer.add_graph(model=model, input_to_model=dummy_input)
    except Exception as e:
        pass

    writer.close()

    print(
        f"Training finished. Best validation loss: {best_val_loss:.4f} achieved. Best model saved to {best_model_save_path}"
    )

    return results
