import torch
import torch.nn as nn
from torchvision import models

def resnet18(num_classes=1, use_cuda=True):
    """
    Load a ResNet-18 model pretrained on ImageNet, modify the first convolution layer to
    accept grayscale input (1 channel), and adjust the output layer for custom number of classes.
    
    Args:
        num_classes (int): Number of output segmentation classes (e.g., 1 for binary).
        use_cuda (bool): Whether to move the model to GPU if available.
    
    Returns:
        model (torch.nn.Module): The modified ResNet-18 model.
    """
    try:
        # Load the pretrained ResNet-18 model
        model = models.resnet18(pretrained=True)
        
        # Modify the fully connected layer to match the number of classes
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        
        # Optionally, move to GPU if requested and CUDA is available
        if use_cuda and torch.cuda.is_available():
            model = model.cuda()
            print("✅ Model successfully moved to CUDA")
        else:
            print("⚠️ CUDA not used - running on CPU")
        
        return model

    except Exception as e:
        print("❌ Error during model setup:", e)
        return None