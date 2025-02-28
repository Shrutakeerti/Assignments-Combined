import torch
import torch.nn as nn
from monai.networks.nets import VNet

def get_vnet_model(num_classes):
    """
    Initializes and returns a VNet model from MONAI.

    Args:
        num_classes (int): Number of output classes.

    Returns:
        model (nn.Module): VNet model.
    """
    model = VNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=num_classes,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
    return model
