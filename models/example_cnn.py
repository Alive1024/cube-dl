import torch
from torch import nn
import torch.nn.functional as F


class ExampleCNN(nn.Module):
    def __init__(self, num_input_channels, num_classes):
        super().__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(num_input_channels, 64, kernel_size=3, stride=1, padding=1),
                                         torch.nn.MaxPool2d(stride=2, kernel_size=2),
                                         torch.nn.ReLU(),
                                         torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                                         torch.nn.MaxPool2d(stride=2, kernel_size=2),
                                         torch.nn.ReLU())
        self.dense = torch.nn.Sequential(torch.nn.Linear(7 * 7 * 128, 1024),
                                         torch.nn.ReLU(),
                                         torch.nn.Dropout(p=0.2),
                                         torch.nn.Linear(1024, num_classes))

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 7 * 7 * 128)
        x = self.dense(x)
        return F.log_softmax(x, dim=1)
