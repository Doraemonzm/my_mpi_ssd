# import math
# import torch.nn as nn
# import torch.utils.model_zoo as model_zoo
from ssd.modeling import registry
#
# # __all__ = ['ResNet', 'resnet_base']
#
# model_urls = {
#     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#     'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#     'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#     'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
# }
#
#
# class ConvBnReluLayer(nn.Module):
#
#     def __init__(self, inplanes, planes, kernel_size, padding, stride, bias=False):
#         super(ConvBnReluLayer, self).__init__()
#         self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, padding=padding,
#                               stride=stride, bias=bias)
#         self.bn = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         out = self.conv(x)
#         out = self.bn(out)
#         out = self.relu(out)
#         return out
#
#
# class ExtraLayers(nn.Module):
#
#     def __init__(self, inplanes):
#         super(ExtraLayers, self).__init__()
#         self.convbnrelu1_1 = ConvBnReluLayer(inplanes, 256, kernel_size=1, padding=0, stride=1)
#         self.convbnrelu1_2 = ConvBnReluLayer(256, 512, kernel_size=3, padding=1, stride=2)
#         self.convbnrelu2_1 = ConvBnReluLayer(512, 256, kernel_size=1, padding=0, stride=1)
#         self.convbnrelu2_2 = ConvBnReluLayer(256, 512, kernel_size=3, padding=1, stride=2)
#         self.convbnrelu3_1 = ConvBnReluLayer(512, 256, kernel_size=1, padding=0, stride=1)
#         self.convbnrelu3_2 = ConvBnReluLayer(256, 512, kernel_size=3, padding=1, stride=2)
#         self.avgpool = nn.AvgPool2d(3, stride=1)
#
#     def forward(self, x):
#         out1_1 = self.convbnrelu1_1(x)
#         out1_2 = self.convbnrelu1_2(out1_1)
#         out2_1 = self.convbnrelu2_1(out1_2)
#         out2_2 = self.convbnrelu2_2(out2_1)
#         out3_1 = self.convbnrelu3_1(out2_2)
#         out3_2 = self.convbnrelu3_2(out3_1)
#         out = self.avgpool(out3_2)
#         return out1_2, out2_2, out3_2, out
#
#
# def conv3x3(in_planes, out_planes, stride=1, dilation=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, dilation=dilation,
#                      padding=1, bias=False)
#
#
# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride, dilation)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out
#
#
# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=dilation,
#                                padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * 4)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out
#
#
# class ResNetBase(nn.Module):
#
#     def __init__(self, block, layers, width=1, num_classes=1000):
#         self.inplanes = 64
#         widths = [int(round(ch * width)) for ch in [64, 128, 256, 512]]
#         super(ResNetBase, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#                                bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, widths[0], layers[0])
#         self.layer2 = self._make_layer(block, widths[1], layers[1], stride=2)
#         self.layer3 = self._make_layer(block, widths[2], layers[2], stride=2)
#         # change stride = 2, dilation = 1 in ResNet to stride = 1, dilation = 2 for the final _make_layer
#         self.layer4 = self._make_layer(block, widths[3], layers[3], stride=1, dilation=2)
#         # remove the final avgpool and fc layers
#         # self.avgpool = nn.AvgPool2d(7, stride=1)
#         # self.fc = nn.Linear(widths[3] * block.expansion, num_classes)
#         # add extra layers
#         self.extra_layers = ExtraLayers(self.inplanes)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, dilation=dilation, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, dilation, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         out38x38 = x
#         x = self.layer3(x)
#         x = self.layer4(x)
#         out19x19 = x
#
#         out10x10, out5x5, out3x3, out1x1 = self.extra_layers(x)
#         # x = self.avgpool(x)
#         # x = x.view(x.size(0), -1)
#         # x = self.fc(x)
#
#         return out38x38, out19x19, out10x10, out5x5, out3x3, out1x1
#
# #
# # def resnet_base(depth, width=1, pretrained=False, **kwargs):
# #     """Constructs a ResNet base network model for SSD.
# #     Args:
# #         depth (int): choose 18, 34, 50, 101, 152
# #         width (float): widen factor for intermediate layers of resnet
# #         pretrained (bool): If True, returns a model pre-trained on ImageNet
# #     """
# #     if (depth not in [18, 34, 50, 101, 152]):
# #         raise ValueError('Choose 18, 34, 50, 101 or 152 for depth')
# #     if ((width != 1) and pretrained):
# #         raise ValueError('Does not support pretrained models with width>1')
# #
# #     name_dict = {18: 'resnet18', 34: 'resnet34', 50: 'resnet50', 101: 'resnet101', 152: 'resnet152'}
# #     layers_dict = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3],
# #                    101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}
# #     block_dict = {18: BasicBlock, 34: BasicBlock, 50: Bottleneck, 101: Bottleneck, 152: Bottleneck}
# #     model = ResNetBase(block_dict[depth], layers_dict[depth], width, **kwargs)
# #     if ((width == 1) and pretrained):
# #         model.load_state_dict(model_zoo.load_url(model_urls[name_dict[depth]]))
# #     return model
#
#
# @registry.BACKBONES.register('resnetbase')
# def resnetbase(cfg, pretrained=True,**kwargs):
#     name_dict = {18: 'resnet18', 34: 'resnet34', 50: 'resnet50', 101: 'resnet101', 152: 'resnet152'}
#     layers_dict = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3],
#                    101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}
#     block_dict = {18: BasicBlock, 34: BasicBlock, 50: Bottleneck, 101: Bottleneck, 152: Bottleneck}
#     model = ResNetBase(block_dict[50], layers_dict[50], 1, **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
#     return model


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
# from models.model_helper import weights_init
from ssd.utils.model_zoo import load_state_dict_from_url


model_urls = {
    'resnet': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',

}
def weights_init(m):
    for key in m.state_dict():
        if key.split('.')[-1] == 'weight':
            if 'conv' in key:
                init.kaiming_normal(m.state_dict()[key], mode='fan_out')
            if 'bn' in key:
                m.state_dict()[key][...] = 1
        elif key.split('.')[-1] == 'bias':
            m.state_dict()[key][...] = 0

def add_extras(size, in_channel, batch_norm=False):
    # Extra layers added to resnet for feature scaling
    layers = []
    layers += [nn.Conv2d(in_channel, 256, kernel_size=1, stride=1)]
    layers += [nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)]
    layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
    layers += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)]
    if size == '300':
        layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)]
    else:
        layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)]
        layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)]

    return layers


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SSDResnet(nn.Module):
    def __init__(self, block, num_blocks, size):
        super(SSDResnet, self).__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Bottom-up layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.inchannel = block.expansion * 512
        self.extras = nn.ModuleList(add_extras(str(size), self.inchannel))
        self.smooth1 = nn.Conv2d(
            self.inchannel, 512, kernel_size=3, stride=1, padding=1)
        self._init_modules()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_modules(self):
        self.extras.apply(weights_init)
        self.smooth1.apply(weights_init)

    def forward(self, x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        x = c5
        c5_ = self.smooth1(c5)
        sources = [c3, c4, c5_]
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)
        return sources


def SSDResnet18(size, channel_size='48'):
    return SSDResnet(BasicBlock, [2, 2, 2, 2], size)


def SSDResnet34(size, channel_size='48'):
    return SSDResnet(BasicBlock, [3, 4, 6, 3], size)


def SSDResnet50(size, channel_size='48'):
    return SSDResnet(Bottleneck, [3, 4, 6, 3], size)


def SSDResnet101(size, channel_size='48'):
    return SSDResnet(Bottleneck, [3, 4, 23, 3], size)


def SSDResnet152(size, channel_size='48'):
    return SSDResnet(Bottleneck, [3, 8, 36, 3], size)


# if __name__ == "__main__":
#     import os
#     os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#     model3 = SSDResnet18(size=300)
#     with torch.no_grad():
#         model3.eval()
#         x = torch.randn(1, 3, 300, 300)
#         model3.cuda()
#         model3(x.cuda())
#         import time
#         st = time.time()
#         for i in range(1):
#             model3(x.cuda())
#         print(time.time() - st)
        # print(model3(x))


@registry.BACKBONES.register('resnet')
def resnet(cfg, pretrained=True):
    # model = VGG(cfg)
    model = SSDResnet50(size=300)
    weights = torch.load('/home/wzm/test/my_SSD/weights/resnet50-19c8e357.pth')
    if pretrained:
        # model.init_from_pretrain(load_state_dict_from_url(model_urls['vgg']))
        # model.load_state_dict(load_state_dict_from_url(model_urls['resnet']), strict=False)
        model.load_state_dict(weights, strict=False)

    return model