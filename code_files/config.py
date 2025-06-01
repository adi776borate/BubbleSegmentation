# config.py
import os
import torch

class Config:
    # General
    IN_CHANNELS = 3
    NUM_CLASSES = 2

    IMAGE_DIR = "../Data/US_2"
    LABEL_DIR = "../Data/Label_2"
    TEST_IMAGE_DIR = "../Data/US_Test_2023April7" 
    TEST_LABEL_DIR = "../Data/Label_Test_2023April7" 

    # Training
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LEARNING_RATE = float(os.getenv("LEARNING_RATE", 3e-4)) 
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 16))             
    NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 5))           

    # Optimizer
    OPTIMIZER = os.getenv("OPTIMIZER", "Adam")
    USE_SCHEDULER = os.getenv("USE_SCHEDULER", "True").lower() == "true"

    # Model
    # Options: ResNet18CNN, DeepLabV3Plus, FPN, TorchvisionDeepLabV3, Unet++, MaskRCNN, 
    MODEL_NAME = os.getenv("MODEL_NAME", "TorchvisionDeepLabV3")
    ENCODER_NAME = os.getenv("ENCODER_NAME", "resnet34")
    ENCODER_WEIGHTS = os.getenv("ENCODER_WEIGHTS", "imagenet")
    USE_CUDA = os.getenv("USE_CUDA", "True").lower() == "true"

    # Loss
    # Options: DiceFocalLoss, DiceLoss, AsymmetricFocalTverskyLoss
    LOSS_FN = os.getenv("LOSS_FN", "DiceFocalWithPulsePriorLoss")
    LOSS_DICE_WEIGHT = os.getenv("LOSS_DICE_WEIGHT", 0.5)
    LOSS_TVERSKY_WEIGHT = os.getenv("LOSS_TVERSKY_WEIGHT", 0.5)
    LOSS_FOCAL_WEIGHT = os.getenv("LOSS_FOCAL_WEIGHT", 0.6)

    # Early stopping settings
    EARLY_STOPPING = True
    PATIENCE = 5            # Number of epochs to wait before stopping
    DELTA = 1e-4           # Minimum change in validation loss to qualify as improvement

    # Logging
    SAVE_MODEL = os.getenv("SAVE_MODEL", "True").lower() == "true"
    CHECKPOINT_DIR = "checkpoints/"
    LOG_DIR = "logs/"

    # from datetime import datetime
    # timestamp = datetime.now().strftime("%b%d_%H-%M")
    EXPERIMENT_NAME = os.getenv(
        "EXPERIMENT_NAME",
        f"{MODEL_NAME}_{LOSS_FN}_Dice{LOSS_DICE_WEIGHT}_Tversky{LOSS_TVERSKY_WEIGHT}_Focal{LOSS_FOCAL_WEIGHT}_Epochs{NUM_EPOCHS}_LR{LEARNING_RATE}"
    )

    VISUALIZE_EVERY = int(os.getenv("VISUALIZE_EVERY", 4))
    CSV_LOG_FILE = "training_log.csv"