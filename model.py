import torch as t
from torch import nn
from torchvision import models


class DeConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DeConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class CenterNet(nn.Module):

    def __init__(self, num_classes):
        super(CenterNet, self).__init__()
        backbone = models.resnet18(pretrained=True)
        self.preprocess = nn.Sequential(*list(backbone.children())[:4])
        self.d1 = backbone.layer1
        self.d2 = backbone.layer2
        self.d3 = backbone.layer3
        self.d4 = backbone.layer4
        self.u1 = DeConvBlock(in_channels=512, out_channels=256)
        self.u2 = DeConvBlock(in_channels=256, out_channels=128)
        self.u3 = DeConvBlock(in_channels=128, out_channels=64)
        self.last = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=num_classes + 4, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.preprocess(x)
        x1 = self.d1(x)
        x2 = self.d2(x1)
        x3 = self.d3(x2)
        x4 = self.d4(x3)
        x5 = self.u1(x4) + x3
        x6 = self.u2(x5) + x2
        x7 = self.u3(x6) + x1
        x8 = self.last(x7)
        return x8


if __name__ == "__main__":
    d = t.randn(2, 3, 512, 512)
    model = CenterNet(1)
    out = model(d)
    print(out.size())