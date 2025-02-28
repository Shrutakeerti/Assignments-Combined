import torch
import SimpleITK as sitk
import numpy as np
import os

def load_model(model, checkpoint_path, device):
    """
    Loads the model weights from a checkpoint.

    Args:
        model (nn.Module): The segmentation model.
        checkpoint_path (str): Path to the checkpoint file.
        device (torch.device): Device to load the model on.

    Returns:
        model (nn.Module): Model with loaded weights.
    """
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict(model, image_path, device):
    """
    Performs prediction on a single CT scan.

    Args:
        model (nn.Module): The trained segmentation model.
        image_path (str): Path to the CT scan image.
        device (torch.device): Device to perform inference on.

    Returns:
        prediction (np.ndarray): Predicted segmentation mask.
    """
    image = sitk.ReadImage(image_path)
    image = sitk.GetArrayFromImage(image).astype(np.float32)
    image = (image - np.mean(image)) / np.std(image)  # Normalize
    image = np.expand_dims(image, axis=0)  # Add channel dimension
    image_tensor = torch.tensor(image).unsqueeze(0).to(device)  # Shape: [1, 1, D, H, W]

    with torch.no_grad():
        output = model(image_tensor)
        preds = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    return preds

def save_prediction(prediction, reference_image_path, output_path):
    """
    Saves the predicted segmentation mask as a NIfTI file.

    Args:
        prediction (np.ndarray): Predicted segmentation mask.
        reference_image_path (str): Path to the original CT scan for metadata.
        output_path (str): Path to save the prediction.

    Returns:
        None
    """
    reference_image = sitk.ReadImage(reference_image_path)
    prediction_image = sitk.GetImageFromArray(prediction)
    prediction_image.CopyInformation(reference_image)
    sitk.WriteImage(prediction_image, output_path)
