import torch.nn as nn
import torch
import torchvision
class Model(nn.Module):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Encoder - feature extraction
        pretrained = torchvision.models.resnet50(torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        
        pretrained = list(pretrained.children())
        self.pretrained = nn.Sequential(*pretrained[:-1]) 
        self.output = nn.Linear(512, 3 * 42)
        
        
    
    def forward(self, X):
        X = self.pretrained(X).squeeze(-1).squeeze(-1)
        X = self.output(X)
        return X.view(-1, 3, 6, 7)