import torch
import torch.nn as nn

def dice_coefficient(pred, target, num_classes=4):
    """
    Computes Dice Coefficient for each class.

    Args:
        pred (torch.Tensor): Predicted labels.
        target (torch.Tensor): Ground truth labels.
        num_classes (int): Number of classes.

    Returns:
        dice_scores (dict): Dice score for each class.
    """
    dice_scores = {}
    for cls in range(1, num_classes + 1):  # Exclude background
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).sum().item()
        pred_sum = pred_inds.sum().item()
        target_sum = target_inds.sum().item()
        dice = (2. * intersection) / (pred_sum + target_sum + 1e-8)
        dice_scores[cls] = dice
    return dice_scores

def evaluate(model, dataloader, device, num_classes=4):
    model.eval()
    dice_scores = {cls: 0.0 for cls in range(1, num_classes + 1)}
    total_batches = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            batch_dice = dice_coefficient(preds, labels, num_classes)
            for cls in dice_scores:
                dice_scores[cls] += batch_dice.get(cls, 0)
            total_batches += 1

    # Average over all batches
    avg_dice_scores = {cls: score / total_batches for cls, score in dice_scores.items()}
    return avg_dice_scores
