import segmentation_models_pytorch as smp

def get_model(architecture_name, encoder_name):
    """
    Initializes a model from segmentation-models-pytorch.
    
    Args:
        architecture_name (str): 'unet', 'segformer', 'deeplabv3plus', etc.
        encoder_name (str): 'resnet50', 'efficientnet-b4', 'mit_b3', etc.
    """
    print(f"Initializing {architecture_name} with {encoder_name} backbone...")
    
    if architecture_name == 'unet':
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet", # Use pre-trained weights
            in_channels=3,              # 3-channel RGB images
            classes=1,                  # 1 output channel (binary mask)
            activation=None       # Sigmoid for binary output
        )
    elif architecture_name == 'segformer':
        # Note: SegFormer encoders are named 'mit_b0' through 'mit_b5'
        model = smp.Segformer(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation='sigmoid'
        )
    elif architecture_name == 'deeplabv3plus':
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation='sigmoid'
        )
    # Add other architectures you want to test here...
    else:
        raise ValueError(f"Architecture '{architecture_name}' not supported.")
        
    return model