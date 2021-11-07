import torch
import torch.nn as nn
from torchsummary import summary


class CNet(nn.Module):
    def __init__(self):
        super(CNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=16,
                      kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=3,
                      stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
                      stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.full1 = nn.Linear(3 * 3 * 64, 100)
        self.full2 = nn.Linear(100, 20)
        self.full3 = nn.Linear(20, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.full1(x.view(x.size()[0], -1))
        x = self.full2(x)
        x = self.full3(x)
        return x


if __name__ == '__main__':
    summary(CNet().cuda(), input_size=(1, 20, 20))
