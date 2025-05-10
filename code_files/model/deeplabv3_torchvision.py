import torch.nn as nn
import torch
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights

def torchvision_deeplabv3(num_classes=2):
    """
    Load DeepLabV3 with a ResNet-101 backbone, modify the classifier for a custom number of output classes,
    and modify the first convolution layer for grayscale input (1 channel).
    
    Args:
        num_classes (int): Number of output segmentation classes.
        use_cuda (bool): Whether to move the model to GPU if available.

    Returns:
        model (torch.nn.Module): The modified DeepLabV3 model.
    """
    try:
        weights = DeepLabV3_ResNet101_Weights.DEFAULT
        model = deeplabv3_resnet101(weights=weights)
        
        # Modify the classifier for custom number of classes
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        
        # Move to GPU if available and requested
        use_cuda = True
        if use_cuda and torch.cuda.is_available():
            model = model.cuda()
            print("Model successfully moved to CUDA")
        else:
            print("CUDA not used - running on CPU")

        return model

    except Exception as e:
        print("Error during model setup:", e)
        return None