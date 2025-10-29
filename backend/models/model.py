import torch
import torch.nn as nn
import torch.nn.functional as F

class MyCNNModel(nn.Module):
  def __init__(self, input_size):
    super(MyCNNModel, self).__init__()
    self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
    self.pool = nn.MaxPool2d(2,2)
    self.act = nn.ReLU()
    self.pool2 = nn.AdaptiveAvgPool1d(1024)
    self.fc1 = nn.Linear(1024, 256)
    self.fc2 = nn.Linear(256, 10)
  def forward(self, x):
    x = self.pool(self.act(self.conv1(x)))
    x = self.pool(self.act(self.conv2(x)))
    x = torch.flatten(x, start_dim=1)
    x = self.pool2(x)
    x = self.act(self.fc1(x))
    x = self.fc2(x)
    return x