import segmentation_models_pytorch as smp

def fpn(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1):
    model = smp.FPN(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes
    )
    return model