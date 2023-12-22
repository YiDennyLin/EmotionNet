import torch
from torchvision import models
from torch import nn

class SpatialAttention(nn.Module):
  """
    Spatial Attention Module
  """
  def __init__(self, kernel_size=7):
    # default set size = 7
    super(SpatialAttention, self).__init__()
    assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
    padding = 3 if kernel_size == 7 else 1

    self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    # Average along channel axis
    avg_out = torch.mean(x, dim=1, keepdim=True)
    # Max along channel axis
    max_out, _ = torch.max(x, dim=1, keepdim=True)
    # Stack channel-wise
    x = torch.cat([avg_out, max_out], dim=1)
    x = self.conv1(x)
    return self.sigmoid(x)

class EmotionNet(nn.Module):
  """
   The framework of EmotionNet
  """
  def __init__(self, num_classes=7):
    super(EmotionNet, self).__init__()
    self.resnet = models.resnet18(pretrained=True)
    # self.resnet = models.resnet50(pretrained=True)
    self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    self.sa = SpatialAttention()

  def forward(self, x):
    x = self.resnet.conv1(x)
    x = self.resnet.bn1(x)
    x = self.resnet.relu(x)
    x = self.resnet.maxpool(x)

    x = self.sa(x) * x

    x = self.resnet.layer1(x)
    # x = self.sa(x) * x  # Spatial Attention

    x = self.resnet.layer2(x)
    # x = self.sa(x) * x

    x = self.resnet.layer3(x)
    # x = self.sa(x) * x

    x = self.resnet.layer4(x)
    # x = self.sa(x) * x

    x = self.resnet.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.resnet.fc(x)

    return x
