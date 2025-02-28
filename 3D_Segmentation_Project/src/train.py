import torch
import torch.nn as nn
from tqdm import tqdm
import os

def train_model(model, train_loader, val_loader, num_epochs, device, output_dir):
    """
    Trains the model and evaluates on validation set.

    Args:
        model (nn.Module): The segmentation model.
        train_loader (DataLoader): Training DataLoader.
        val_loader (DataLoader): Validation DataLoader.
        num_epochs (int): Number of training epochs.
        device (torch.device): Device to train on.
        output_dir (str): Directory to save model checkpoints.

    Returns:
        None
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch in train_loader:
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
                pbar.update(1)

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss}")

        # Save checkpoint
        checkpoint_path = os.path.join(output_dir, f'vnet_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        # Validate after each epoch
        evaluate_model(model, val_loader, device)

def evaluate_model(model, val_loader, device):
    """
    Evaluates the model on the validation set using Dice score.

    Args:
        model (nn.Module): The segmentation model.
        val_loader (DataLoader): Validation DataLoader.
        device (torch.device): Device to evaluate on.

    Returns:
        None
    """
    model.eval()
    dice_scores = {}
    num_classes = 4  # Liver, Right Kidney, Left Kidney, Spleen

    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            for cls in range(1, num_classes + 1):  # Assuming background is 0
                pred_inds = (preds == cls)
                target_inds = (labels == cls)
                intersection = (pred_inds & target_inds).sum().item()
                pred_sum = pred_inds.sum().item()
                target_sum = target_inds.sum().item()
                dice = (2. * intersection) / (pred_sum + target_sum + 1e-8)
                dice_scores[cls] = dice_scores.get(cls, 0) + dice

    # Calculate average Dice score for each class
    for cls in dice_scores:
        dice_scores[cls] /= len(val_loader)

    class_names = {1: 'Liver', 2: 'Right Kidney', 3: 'Left Kidney', 4: 'Spleen'}
    for cls, score in dice_scores.items():
        print(f"Validation Dice Score for {class_names[cls]}: {score:.4f}")
# src/train.py

import torch
import torch.nn as nn
from tqdm import tqdm
import os

def train_model(model, train_loader, val_loader, num_epochs, device, output_dir):
    """
    Trains the model and evaluates on validation set.

    Args:
        model (nn.Module): The segmentation model.
        train_loader (DataLoader): Training DataLoader.
        val_loader (DataLoader): Validation DataLoader.
        num_epochs (int): Number of training epochs.
        device (torch.device): Device to train on.
        output_dir (str): Directory to save model checkpoints.

    Returns:
        None
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch in train_loader:
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
                pbar.update(1)

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss}")

        # Save checkpoint
        checkpoint_path = os.path.join(output_dir, f'vnet_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        # Validate after each epoch
        evaluate_model(model, val_loader, device)

def evaluate_model(model, val_loader, device):
    """
    Evaluates the model on the validation set using Dice score.

    Args:
        model (nn.Module): The segmentation model.
        val_loader (DataLoader): Validation DataLoader.
        device (torch.device): Device to evaluate on.

    Returns:
        None
    """
    model.eval()
    dice_scores = {}
    num_classes = 4  # Liver, Right Kidney, Left Kidney, Spleen

    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            for cls in range(1, num_classes + 1):  # Assuming background is 0
                pred_inds = (preds == cls)
                target_inds = (labels == cls)
                intersection = (pred_inds & target_inds).sum().item()
                pred_sum = pred_inds.sum().item()
                target_sum = target_inds.sum().item()
                dice = (2. * intersection) / (pred_sum + target_sum + 1e-8)
                dice_scores[cls] = dice_scores.get(cls, 0) + dice

    # Calculate average Dice score for each class
    for cls in dice_scores:
        dice_scores[cls] /= len(val_loader)

    class_names = {1: 'Liver', 2: 'Right Kidney', 3: 'Left Kidney', 4: 'Spleen'}
    for cls, score in dice_scores.items():
        print(f"Validation Dice Score for {class_names[cls]}: {score:.4f}")
# src/train.py

import torch
import torch.nn as nn
from tqdm import tqdm
import os

def train_model(model, train_loader, val_loader, num_epochs, device, output_dir):
    """
    Trains the model and evaluates on validation set.

    Args:
        model (nn.Module): The segmentation model.
        train_loader (DataLoader): Training DataLoader.
        val_loader (DataLoader): Validation DataLoader.
        num_epochs (int): Number of training epochs.
        device (torch.device): Device to train on.
        output_dir (str): Directory to save model checkpoints.

    Returns:
        None
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch in train_loader:
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
                pbar.update(1)

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss}")

        # Save checkpoint
        checkpoint_path = os.path.join(output_dir, f'vnet_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        # Validate after each epoch
        evaluate_model(model, val_loader, device)

def evaluate_model(model, val_loader, device):
    """
    Evaluates the model on the validation set using Dice score.

    Args:
        model (nn.Module): The segmentation model.
        val_loader (DataLoader): Validation DataLoader.
        device (torch.device): Device to evaluate on.

    Returns:
        None
    """
    model.eval()
    dice_scores = {}
    num_classes = 4  # Liver, Right Kidney, Left Kidney, Spleen

    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            for cls in range(1, num_classes + 1):  # Assuming background is 0
                pred_inds = (preds == cls)
                target_inds = (labels == cls)
                intersection = (pred_inds & target_inds).sum().item()
                pred_sum = pred_inds.sum().item()
                target_sum = target_inds.sum().item()
                dice = (2. * intersection) / (pred_sum + target_sum + 1e-8)
                dice_scores[cls] = dice_scores.get(cls, 0) + dice

    # Calculate average Dice score for each class
    for cls in dice_scores:
        dice_scores[cls] /= len(val_loader)

    class_names = {1: 'Liver', 2: 'Right Kidney', 3: 'Left Kidney', 4: 'Spleen'}
    for cls, score in dice_scores.items():
        print(f"Validation Dice Score for {class_names[cls]}: {score:.4f}")
