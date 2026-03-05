import torch.nn as nn
import torch
import torchvision
class Model(nn.Module):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Encoder - feature extraction
        pretrained = torchvision.models.resnet18(torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        
        pretrained = list(pretrained.children())
        self.pretrained = nn.Sequential(*pretrained[:-1]) 
        self.output = nn.Linear(512, 27)
        
        
    
    def forward(self, X):
        X = self.pretrained(X).squeeze(-1).squeeze(-1)
        X = self.output(X)
        return X.view(-1, 3, 3, 3)
        
if __name__=="__main__":
    model = Model()
    model.eval()
    with torch.no_grad():
        X = torch.zeros((3,3,3)).unsqueeze(0)
        y = model(X)
    print(y)