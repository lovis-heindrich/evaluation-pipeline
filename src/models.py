from torch import nn
import torch
import torch.nn.functional as F
from base_classes import PipelineClassifier

class BaseClassifier(nn.Module, PipelineClassifier):
    def __init__(self, num_channels, num_classes):
        super(BaseClassifier, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(28800, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.name = "Baseline"

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return(x)

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    def get_name(self):
        return self.name
    
    def get_prediction(self, x):
        with torch.no_grad():
            prediction = self.forward(x)
            softmax = F.softmax(prediction, 1)
            max_activation, predicted = torch.max(softmax.data, 1)
            return softmax, max_activation, predicted
    
    @staticmethod
    def get_classifier(num_channels, num_classes, num_epochs, device, dropout = 0):
        classifier = BaseClassifier(num_channels, num_classes)

class MCSampleClassifier(nn.Module, PipelineClassifier):
    def __init__(self, num_channels, num_classes, drop, device, n_samples=50):
        super(MCSampleClassifier, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.pool = nn.MaxPool2d(2,2)
        self.drop2d = nn.Dropout2d(drop)
        self.drop1d = nn.Dropout(drop)
        self.fc1 = nn.Linear(28800, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.num_samples = 50
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.device = device
        self.name = "MC Dropout"

    def forward(self, x):
        x = self.drop2d(F.relu(self.conv1(x)))
        x = self.drop2d(self.pool(F.relu(self.conv2(x))))
        x = x.view(-1, self.num_flat_features(x))
        x = self.drop1d(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return(x)

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def get_name(self):
        return self.name
    
    def get_prediction(self, x):
        with torch.no_grad():
            softmax, max_activation, predicted = self.MCSample(x)
            return softmax, max_activation, predicted

    def MCSample(self, x):
        self.train()
        with torch.no_grad():
            softmax = torch.zeros(x.shape[0], self.num_classes).to(self.device)
            for i in range(self.num_samples):
                softmax += F.softmax(self.forward(x), 1)
            softmax = softmax / self.num_samples
            max_activation, predicted = torch.max(softmax.data, 1)
            return softmax, max_activation, predicted
