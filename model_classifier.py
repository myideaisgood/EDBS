import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(torch.nn.Module):

    def __init__(self, in_channels, out_features, hidden_size=64, feature_size=64):
        super(Classifier, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size
        self.feature_size = feature_size

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.MaxPool2d(2),            
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.MaxPool2d(2),  
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.MaxPool2d(2)                        
        )

        self.relu = nn.ReLU()
        self.fc = nn.Linear(5*5*hidden_size, hidden_size, bias=True)
        self.classifier = nn.Linear(hidden_size, out_features, bias=True)

    def forward(self, inputs):
        features = self.features(inputs)
        features = features.view((features.size(0), -1))
        features = self.fc(features)
        features = F.normalize(features, p=2, dim=1)

        logits = self.classifier(self.relu(features))

        return features, logits

if __name__ == '__main__':
    model = Classifier(in_channels=3, out_features=64)
    
    input = torch.rand(4, 3, 84, 84)
    f, l = model(input)