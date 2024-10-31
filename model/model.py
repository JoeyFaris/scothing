from torch import nn
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

class ClothingAnalyzer(nn.Module):
    def __init__(self, model_name='resnet50', num_classes=None):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = models.resnet50(pretrained=True)
        
        if num_classes:
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, num_classes)
        
        self.model = self.model.to(self.device)
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return self.transform(image).unsqueeze(0).to(self.device)
    
    def forward(self, x):
        return self.model(x)
    
    def predict(self, image_path):
        self.eval()
        with torch.no_grad():
            image = self.preprocess_image(image_path)
            outputs = self(image)
            return outputs