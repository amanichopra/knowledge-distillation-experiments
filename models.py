import torch.nn as nn
import torch
import torchvision

# deep CNN to use as teacher
class CNNTeacher(nn.Module):
    def __init__(self, num_classes=10, dropout=0.1):
        super(CNNTeacher, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    # method also returns hidden layer rep to compute soft target loss
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        out_conv_features = torch.nn.functional.avg_pool1d(x, 2)
        x = self.classifier(x)
        return x, out_conv_features

# shallow CNN to use as teacher
class CNNStudent(nn.Module):
    def __init__(self, num_classes=10, dropout=0.1):
        super(CNNStudent, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    # method also returns hidden layer rep to compute soft target loss
    def forward(self, x):
        x = self.features(x)
        out_conv_features = torch.flatten(x, 1)
        x = self.classifier(out_conv_features)
        return x, out_conv_features

class Resnet18(torch.nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        
        resnet = torchvision.models.resnet18(pretrained=pretrained)
        
        # get all layers before last fc layer
        self.features = torch.nn.ModuleList(resnet.children())[:-1]
        self.features = torch.nn.Sequential(*self.features)
        self.fc = torch.nn.Linear(resnet.fc.in_features, 10)
    

    def forward(self, input_imgs):
        x = self.features(input_imgs)
        out_features = torch.flatten(x, 1)
        x = self.fc(out_features)
                
        return x, out_features

# modify base resnet models to enable classifying on cifar10
class Resnet34(torch.nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        
        resnet = torchvision.models.resnet34(pretrained=pretrained)
        
        # get all layers before last fc layer
        self.features = torch.nn.ModuleList(resnet.children())[:-1]
        self.features = torch.nn.Sequential(*self.features)
        self.fc = torch.nn.Linear(resnet.fc.in_features, 10)
    
    # method also returns hidden layer rep to compute soft target loss
    def forward(self, input_imgs):
        x = self.features(input_imgs)
        out_features = torch.flatten(x, 1)
        x = self.fc(out_features)
                
        return x, out_features

class Resnet50(torch.nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        
        resnet = torchvision.models.resnet50(pretrained=pretrained)
        
        # get all layers before last fc layer
        self.features = torch.nn.ModuleList(resnet.children())[:-1]
        self.features = torch.nn.Sequential(*self.features)
        self.fc = torch.nn.Linear(resnet.fc.in_features, 10)
    
    # method also returns hidden layer rep to compute soft target loss
    def forward(self, input_imgs):
        x = self.features(input_imgs)
        out_features = torch.flatten(x, 1)
        x = self.fc(out_features)
                
        return x, out_features