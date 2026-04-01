import segmentation_models_pytorch as smp

def get_model(architecture_name, encoder_name):
    print(f"Initializing {architecture_name} with {encoder_name} backbone...")
    
    if architecture_name == 'unet':
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet", 
            in_channels=3,              
            classes=1,                  
            activation=None  # <-- MUST BE NONE
        )
    elif architecture_name == 'segformer':
        model = smp.Segformer(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None  # <-- MUST BE NONE
        )
    elif architecture_name == 'deeplabv3plus':
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None  # <-- MUST BE NONE
        )
    else:
        raise ValueError(f"Architecture '{architecture_name}' not supported.")
        
    return model