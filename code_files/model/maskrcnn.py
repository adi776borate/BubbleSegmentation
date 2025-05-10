import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.resnet import resnet50, ResNet50_Weights

def maskrcnn(num_classes, pretrained=True):
    """
    Load Mask R-CNN model with a ResNet-50-FPN backbone and custom head for segmentation.

    Args:
        num_classes (int): Number of output classes (including background).
        pretrained (bool): Whether to load COCO-pretrained weights.

    Returns:
        torch.nn.Module: The customized Mask R-CNN model.
    """
    # Load the model with or without COCO-pretrained weights
    model = maskrcnn_resnet50_fpn(pretrained=pretrained)

    # Modify the backbone to accept grayscale input (1 channel)
    # Extract the resnet backbone from the model
    backbone = model.backbone

    # The default ResNet-50 model used in Mask R-CNN takes 3-channel images, so modify it
    # Get the first convolution layer of the backbone
    conv1 = backbone.body.conv1

    # Replace the box predictor (classification + bbox regression)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace the mask predictor (segmentation masks)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model