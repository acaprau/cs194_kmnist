import torch
import torch.nn.functional as F

from torch import nn
from utils import flatten

IM_HEIGHT = 28
IM_WIDTH = 28


class CNNConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(CNNConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out += identity
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)

        return out


class CNNSimple(nn.Module):
    def __init__(self):
        super(CNNSimple, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(32*4*4, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.maxpool1(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpool2(out)
        out = flatten(out)
        out = self.dropout(out)
        out = self.fc1(out)

        return out


class CNNDeeper(nn.Module):
    def __init__(self):
        super(CNNDeeper, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.block1 = CNNConvBlock(in_channels=64, out_channels=64)
        self.drop1 = nn.Dropout(p=0.5)
        self.block2 = CNNConvBlock(in_channels=64, out_channels=64)
        self.drop2 = nn.Dropout(p=0.5)
        self.block3 = CNNConvBlock(in_channels=64, out_channels=128)
        self.drop3 = nn.Dropout(p=0.5)
        self.block4 = CNNConvBlock(in_channels=128, out_channels=128)
        self.drop4 = nn.Dropout(p=0.5)
        self.block5 = CNNConvBlock(in_channels=128, out_channels=128)
        self.drop5 = nn.Dropout(p=0.5)
        self.block6 = CNNConvBlock(in_channels=128, out_channels=128)
        self.drop6 = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        out_dim = IM_HEIGHT - 2 * 6
        self.fc1 = nn.Linear(128*out_dim*out_dim, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.block1(out)
        out = self.drop1(out)
        out = self.block2(out)
        out = self.drop2(out)
        out = self.block3(out)
        out = self.drop3(out)
        out = self.block4(out)
        out = self.drop4(out)
        out = self.block5(out)
        out = self.drop5(out)
        out = self.block6(out)
        out = self.drop6(out)
        out = flatten(out)
        out = self.fc1(out)

        return out
