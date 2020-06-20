import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class MNIST_Net(nn.Module):
    def __init__(self):
        super(MNIST_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


class CWMNISTNetwork(nn.Module):
    def __init__(self, temperature=None):
        super(CWMNISTNetwork, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, 200),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 10),
        )

        self.temperature = temperature

    def forward(self, x, mode="logits"):
        x = x - 0.5
        x = self.extractor(x)
        x = x.view(len(x), -1)
        logits = self.classifier(x)
        return logits


class RonyMNISTNetwork(nn.Module):
    def __init__(self, temperature=None):
        super(RonyMNISTNetwork, self).__init__()

        self.temperature = temperature

        num_channels = 1
        num_labels = 10

        drop = 0.5

        activ = nn.ReLU(True)

        self.feature_extractor = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(num_channels, 32, 3)),
                    ("relu1", activ),
                    ("conv2", nn.Conv2d(32, 32, 3)),
                    ("relu2", activ),
                    ("maxpool1", nn.MaxPool2d(2, 2)),
                    ("conv3", nn.Conv2d(32, 64, 3)),
                    ("relu3", activ),
                    ("conv4", nn.Conv2d(64, 64, 3)),
                    ("relu4", activ),
                    ("maxpool2", nn.MaxPool2d(2, 2)),
                ]
            )
        )

        self.classifier = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(64 * 4 * 4, 200)),
                    ("relu1", activ),
                    ("drop", nn.Dropout(drop)),
                    ("fc2", nn.Linear(200, 200)),
                    ("relu2", activ),
                    ("fc3", nn.Linear(200, num_labels)),
                ]
            )
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.classifier.fc3.weight, 0)
        nn.init.constant_(self.classifier.fc3.bias, 0)

    def forward(self, x, mode="logits"):
        features = self.feature_extractor(input)
        logits = self.classifier(features.view(-1, 64 * 4 * 4))
        return logits
