import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.resnet import resnet50
from nets.vgg import VGG16
from nets.resnet101_frn import Resnet101, FilterResponseNormalization
from nets.resnet34 import resnet34


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, stride=2):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=stride)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs


class UNetConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, normal_layer=None):
        super(UNetConvBlock, self).__init__()
        if normal_layer is None:
            normal_layer = FilterResponseNormalization

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1)
        self.bn1 = normal_layer(out_planes)

        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1)
        self.bn2 = normal_layer(out_planes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class UNetUpBlock(nn.Module):
    def __init__(
        self,
        in_chans,
        out_chans,
        up_conv_in_channels=None,
        up_conv_out_channels=None,
        up_mode="upconv",
    ):
        super(UNetUpBlock, self).__init__()
        if up_conv_in_channels == None:
            up_conv_in_channels = in_chans
        if up_conv_out_channels == None:
            up_conv_out_channels = out_chans

        if up_mode == "upconv":
            self.up = nn.ConvTranspose2d(
                up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2
            )
        elif up_mode == "upsample":
            self.up = nn.Sequential(
                nn.Upsample(mode="bilinear", scale_factor=2),
                nn.Conv2d(up_conv_in_channels, up_conv_out_channels, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_chans, out_chans)

    def forward(self, bridge, x):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)

        return out


# New


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_chan,
            out_chan,
            kernel_size=ks,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class EdgeOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, upsize, *args, **kwargs):
        super(EdgeOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
        self.up = nn.UpsamplingBilinear2d(scale_factor=upsize)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        x = self.up(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=4)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)


class unetDown(nn.Module):
    def __init__(self, c1, c2, c_, scale, stride=2):
        super(unetDown, self).__init__()
        self.conv1 = nn.Conv2d(c1 + c2, c_, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c_, c_, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(c_, c_, kernel_size=1)
        self.down = nn.Conv2d(c2, c2, kernel_size=3, padding=1, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(c_ * 2)
        self.side_conv = nn.Conv2d(c1 + c2, c_, kernel_size=1)
        self.edge_out = nn.Conv2d(c_ * 2, 1, kernel_size=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=scale)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.down(inputs2)], 1)
        identity = outputs
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = torch.cat([outputs, self.side_conv(identity)], 1)
        outputs = self.bn(outputs)
        outputs = self.relu(outputs)
        edge = self.edge_out(outputs)
        edge = self.up(edge)
        return edge, outputs


class edgeOut(nn.Module):
    def __init__(self, in_ch, mid_ch, n_classes, upsize, depth):
        super(edgeOut, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.relu = nn.ReLU(inplace=True)
        self.side_conv = nn.Conv2d(in_ch, mid_ch, kernel_size=1)
        self.conv_out = nn.Conv2d(mid_ch, n_classes, kernel_size=1, bias=False)
        self.up = nn.ConvTranspose2d(
            n_classes, n_classes, kernel_size=2 * depth, stride=upsize
        )
        self.mul = upsize

    def forward(self, x):
        ipt = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + self.side_conv(ipt)
        out = self.relu(out)
        out = self.conv_out(out)
        out = self.up(out, output_size=x.size() * self.mul)
        return out


class Combine(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Combine, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, out_ch, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

        self.norm_layer1 = nn.GroupNorm(4, 64)
        self.norm_layer2 = nn.GroupNorm(4, 64)

    def forward(self, x):
        attn = self.relu(self.norm_layer1(self.conv1(x)))
        attn = self.relu(self.norm_layer2(self.conv2(attn)))
        attn = F.softmax(self.conv3(attn), dim=1)

        return ((x * attn).sum(1)).unsqueeze(1)


class Unet(nn.Module):
    def __init__(
        self, num_classes=21, pretrained=False, backbone="resnet34", mode="train"
    ):
        super(Unet, self).__init__()
        if backbone == "vgg":
            self.vgg = VGG16(pretrained=pretrained)
            in_filters = [192, 384, 768, 1024]
        elif backbone == "resnet50":
            self.resnet50 = resnet50(pretrained=pretrained)
            in_filters = [192, 512, 1024, 3072]
        elif backbone == "resnet34":
            self.resnet34 = resnet34(pretrained=pretrained)
            in_filters = [128, 192, 384, 768]
        elif backbone == "resnet101":
            self.resnet101 = Resnet101(pretrained=pretrained)
            in_filters = [67, 192, 512, 1024, 2048]
        else:
            raise ValueError(
                "Unsupported backbone - `{}`, Use vgg, resnet50.".format(backbone)
            )

        if backbone == "resnet101":
            out_filters = [64, 128, 256, 512, 1024]
        elif backbone == "resnet34":
            out_filters = [32, 64, 128, 256]
        else:
            out_filters = [64, 128, 256, 512]

        # upsampling
        if backbone == "resnet101":
            self.up_concat4 = UNetUpBlock(in_filters[4], out_filters[4])
            self.up_concat3 = UNetUpBlock(in_filters[3], out_filters[3])
            self.up_concat2 = UNetUpBlock(in_filters[2], out_filters[2])
            self.up_concat1 = UNetUpBlock(
                in_filters[1],
                out_filters[1],
                up_conv_in_channels=256,
                up_conv_out_channels=128,
            )
            self.up_concat0 = UNetUpBlock(
                in_filters[0],
                out_filters[0],
                up_conv_in_channels=128,
                up_conv_out_channels=64,
            )
        else:
            # 64,64,512
            self.up_concat4 = unetUp(in_filters[3], out_filters[3], stride=1)
            # 128,128,256
            self.up_concat3 = unetUp(in_filters[2], out_filters[2], stride=2)
            # 256,256,128
            self.up_concat2 = unetUp(in_filters[1], out_filters[1])
            # 512,512,64
            self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        if backbone == "resnet50":
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.final = nn.Conv2d(out_filters[0], 1, 1)
        if backbone == "resnet34":
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(out_filters[0], 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 16, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.final = nn.Conv2d(16, 1, 1)

        # New Edge Extraction section
        self.conv_out_sp2 = EdgeOutput(64, 32, 1, 2)
        self.conv_out_sp4 = EdgeOutput(64, 32, 1, 4)
        self.conv_out_sp8 = EdgeOutput(128, 64, 1, 8)
        self.final_sp = Combine(5, 5)

        self.backbone = backbone

        self.mode = mode

        self.edge_out_sp1 = EdgeOutput(64, 32, 1, 2, 2)
        self.edge_out_sp2 = EdgeOutput(64, 32, 1, 4, 4)
        self.edge_out_sp3 = EdgeOutput(128, 64, 1, 8, 8)
        self.edge_out_sp4 = EdgeOutput(256, 128, 1, 16, 16)
        self.edge_out_sp5 = EdgeOutput(512, 256, 1, 16, 32)

        self.laplacian_kernel = (
            torch.tensor([-1, -1, -1, -1, 8, -1, -1, -1, -1], dtype=torch.float32)
            .reshape(1, 1, 3, 3)
            .requires_grad_(False)
            .type(torch.cuda.FloatTensor)
        )
        self.convlp = nn.Conv2d(1, 1, 3, padding=1)
        self.convlp.weight.data = self.laplacian_kernel

        self.convu = nn.Conv2d(1, 16, kernel_size=1)
        self.convf = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.convd = nn.Conv2d(16, 1, kernel_size=1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(16)

    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet34":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet34.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet50.forward(inputs)
        elif self.backbone == "resnet101":
            [feat1, feat2, feat3, feat4, feat5, ipt] = self.resnet101.forward(inputs)
        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)

        # New Edge Recognition branch for enhanced recognition of lesion edges
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        feat_out_sp2 = self.edge_out_sp1(feat1)
        feat_out_sp4 = self.edge_out_sp2(feat2)
        feat_out_sp8 = self.edge_out_sp3(feat3)
        feat_out_sp16 = self.edge_out_sp4(feat4)
        feat_out_sp32 = self.edge_out_sp5(feat5)
        if self.up_conv is not None:
            up1 = self.up_conv(up1)
        fuse = torch.cat(
            (feat_out_sp2, feat_out_sp4, feat_out_sp8, feat_out_sp16, feat_out_sp32),
            dim=1,
        )
        feat_out_sp = self.final_sp(fuse)
        final = self.final(up1)
        boundary = self.convlp(final)
        boundary = self.convu(boundary)
        boundary = self.bn1(boundary)
        boundary = self.relu(boundary)
        boundary = self.convf(boundary)
        boundary = self.bn2(boundary)
        boundary = self.relu(boundary)
        boundary = self.convd(boundary)
        return final, feat_out_sp, boundary

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet50.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet34":
            for param in self.resnet34.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet101":
            for param in self.resnet101.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet50.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet34":
            for param in self.resnet34.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet101":
            for param in self.resnet101.parameters():
                param.requires_grad = True

    def crop(self, pic, th, tw):
        h, w = pic.shape[2], pic.shape[3]
        x1 = 28
        y1 = 28
        return pic[:, :, y1 : y1 + th, x1 : x1 + tw]
