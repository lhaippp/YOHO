import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import math


class FilterResponseNormalization(nn.Module):
    def __init__(self, num_features, eps=1e-6, learnable_eps=False):
        """
        Input Variables:
        ----------------
            num_features: A integer indicating the number of input feature dimensions.
            eps: A scalar constant or learnable variable.
            learnable_eps: A bool value indicating whether the eps is learnable.
        """
        super(FilterResponseNormalization, self).__init__()
        self.eps = nn.Parameter(torch.Tensor([eps]))
        if not learnable_eps:
            self.eps.requires_grad_(False)
        self.gamma = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.tau = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.reset_parameters()

    def forward(self, x):
        """
        Input Variables:
        ----------------
            x: Input tensor of shape [NxCxHxW]
        """
        nu2 = torch.pow(x, 2).mean(dim=(2, 3), keepdim=True)
        x = x * torch.rsqrt(nu2 + torch.abs(self.eps))
        return torch.max(self.gamma * x + self.beta, self.tau)

    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        nn.init.zeros_(self.tau)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 conv with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = FilterResponseNormalization
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None, norm_layer=None):
        super(Bottleneck, self).__init__()

        if norm_layer is None:
            norm_layer = FilterResponseNormalization

        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_class=1000, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = FilterResponseNormalization

        self.inplanes = 64

        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_class)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                FilterResponseNormalization(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        input = x
        x = self.conv1(x)  # 1/2  128
        x = self.bn1(x)
        x = self.relu(x)
        f1 = self.maxpool(x)  # 1/4 64

        f2 = self.layer1(f1)  # 1/4 64
        f3 = self.layer2(f2)  # 1/8 32
        f4 = self.layer3(f3)  # 1/16 16
        f5 = self.layer4(f4)  # 1/32 8

        return [x, f2, f3, f4, f5, input]


def Resnet101(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(
            model_zoo.load_url(
                "https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth",
                model_dir="model_data",
            ),
            strict=False,
        )
    del model.avgpool
    del model.fc
    return model
