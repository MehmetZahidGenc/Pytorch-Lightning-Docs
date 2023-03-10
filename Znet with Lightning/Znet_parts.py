import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class MaxPoolLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Zpoint(nn.Module):
    def __init__(self):
        super(Zpoint, self).__init__()
        self.maxZP = MaxPoolLayer()

    def forward(self, x, identify, is_MaxPool=True):
        # input is CHW
        diffY = identify.size()[2] - x.size()[2]
        diffX = identify.size()[3] - x.size()[3]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        result = torch.cat([identify, x], dim=1)

        if is_MaxPool:
            result = self.maxZP(result)

        return result


class BatchNormalization(nn.Module):
    def __init__(self, num_of_features):
        super(BatchNormalization, self).__init__()

        self.num_of_features = num_of_features

        self.BN = nn.Sequential(
            nn.BatchNorm2d(num_features=self.num_of_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.BN(x)


class classifierPart(nn.Module):
    def __init__(self, n_classes):
        super(classifierPart, self).__init__()

        self.n_classes = n_classes

        self.cP = nn.Sequential(
            nn.Linear(in_features=40960, out_features=512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=self.n_classes),
        )

    def forward(self, x):
        return self.cP(x)