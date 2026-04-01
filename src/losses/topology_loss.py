import torch.nn as nn
from torchvision.models import vgg19

class TopologyAwareLoss(nn.Module):
    """
    Implements the Topology-Aware Loss from Mosinska et al. (2018).
    It uses a pre-trained VGG19 to compare the perceptual features of the 
    predicted mask and the ground truth mask.
    """
    def __init__(self, device='cuda', feature_layers=[0, 5, 10, 19, 28], weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(TopologyAwareLoss, self).__init__()
        # Loads VGG19 pretrained on ImageNet
        vgg = vgg19(pretrained=True).features
        self.vgg_layers = vgg.to(device).eval()
        
        # Disables gradients for VGG (it's a fixed feature extractor)
        for param in self.vgg_layers.parameters():
            param.requires_grad = False
            
        self.feature_layers = feature_layers # Indices of ReLU layers
        self.weights = weights
        self.mse = nn.MSELoss()
        self.device = device

    def forward(self, pred, target):
        # 1. Prepares Inputs for VGG
        # VGG expects 3 channels (RGB). If mask is 1 channel, repeat it.
        if pred.shape[1] == 1:
            pred_vgg = pred.repeat(1, 3, 1, 1)
            target_vgg = target.repeat(1, 3, 1, 1)
        else:
            pred_vgg = pred
            target_vgg = target
            
        # VGG expects normalized inputs [-1, 1] roughly, or [0, 1].
        # Our masks are [0, 1] (sigmoid output).
        
        loss = 0.0
        x = pred_vgg
        y = target_vgg
        
        # 2. Extracts Features & Compare
        for i, layer in enumerate(self.vgg_layers):
            x = layer(x)
            y = layer(y)
            
            if i in self.feature_layers:
                # Adds weighted MSE of feature maps
                # We find the index in our list to get the weight
                idx = self.feature_layers.index(i)
                loss += self.weights[idx] * self.mse(x, y)
                
            # Stops if we went past the last layer we care about
            if i >= max(self.feature_layers):
                break
                
        return loss