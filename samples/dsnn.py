import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from spikingjelly.clock_driven import layer
    from spikingjelly.clock_driven import neuron as cext_neuron
except:
    from spikingjelly.activation_based import layer
    from spikingjelly.activation_based import neuron as cext_neuron

__all__ = ['DSNN', 'dsnn18', 'dsnn34', 'dsnn50', 'dsnn101',
           'dsnn152']

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def convpxp(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    if stride != 1:
        return nn.Sequential(
            nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride,
                        groups=in_planes, padding=1, bias=False),
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)
        )
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, connect_f=None):
        super(BasicBlock, self).__init__()
        self.connect_f = connect_f
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('SpikingBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in SpikingBasicBlock")

        self.conv1 = layer.SeqToANNContainer(
            conv3x3(inplanes, planes, stride),
            norm_layer(planes)
        )
        self.sn1 = cext_neuron.MultiStepIFNode(detach_reset=True)

        self.conv2 = layer.SeqToANNContainer(
            conv3x3(planes, planes),
            norm_layer(planes)
        )
        self.downsample = downsample
        if self.downsample is not None:
            self.pool = layer.SeqToANNContainer(
                convpxp(inplanes, planes * self.expansion, stride),
                norm_layer(planes * self.expansion),
                nn.ReLU(),
            )
        self.stride = stride
        self.sn2 = cext_neuron.MultiStepIFNode(detach_reset=True)

    def forward(self, x):
        identity = x[0]
        x_acc = x[1]

        out = self.sn1(self.conv1(identity))

        out = self.sn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(identity)
            x_acc = self.pool(x_acc)

        x_acc = x_acc + out

        if self.connect_f == 'AND':
            out = out * identity
        elif self.connect_f == 'IAND':
            out = identity * (1. - out)
        elif self.connect_f == 'OR':
            out = out + identity - (identity * out)
        elif self.connect_f == 'XOR':
            out = identity * (1. - out) + out * (1. - identity)
        else:
            raise NotImplementedError(self.connect_f)

        return [out, x_acc]

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, connect_f=None):
        super(Bottleneck, self).__init__()
        self.connect_f = connect_f
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = layer.SeqToANNContainer(
            conv1x1(inplanes, width),
            norm_layer(width)
        )
        self.sn1 = cext_neuron.MultiStepIFNode(detach_reset=True)

        self.conv2 = layer.SeqToANNContainer(
            conv3x3(width, width, stride, groups, dilation),
            norm_layer(width)
        )
        self.sn2 = cext_neuron.MultiStepIFNode(detach_reset=True)

        self.conv3 = layer.SeqToANNContainer(
            conv1x1(width, planes * self.expansion),
            norm_layer(planes * self.expansion)
        )
        self.downsample = downsample
        if self.downsample is not None:
            self.pool = layer.SeqToANNContainer(
                    convpxp(inplanes, planes * self.expansion, stride),
                    norm_layer(planes * self.expansion),
                    nn.ReLU()
                )
        self.stride = stride
        self.sn3 = cext_neuron.MultiStepIFNode(detach_reset=True)

    def forward(self, x):
        identity = x[0]
        x_acc = x[1]

        out = self.sn1(self.conv1(identity))

        out = self.sn2(self.conv2(out))

        out = self.sn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(identity)
            x_acc = self.pool(x_acc)

        x_acc = x_acc + out

        if self.connect_f == 'AND':
            out = out * identity
        elif self.connect_f == 'IAND':
            out = identity * (1. - out)
        elif self.connect_f == 'OR':
            out = out + identity - (identity * out)
        elif self.connect_f == 'XOR':
            out = identity * (1. - out) + out * (1. - identity)
        else:
            raise NotImplementedError(self.connect_f)

        return [out, x_acc]


def zero_init_blocks(net: nn.Module, connect_f: str):
    for m in net.modules():
        if isinstance(m, Bottleneck):
            nn.init.constant_(m.conv3.module[1].weight, 0)
        elif isinstance(m, BasicBlock):
            nn.init.constant_(m.conv2.module[1].weight, 0)

class DSNN(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, T=4, connect_f=None):
        super(DSNN, self).__init__()
        self.T = T
        self.connect_f = connect_f
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)


        self.sn1 = cext_neuron.MultiStepIFNode(detach_reset=True)
        self.maxpool = layer.SeqToANNContainer(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.layer1 = self._make_layer(block, 64, layers[0], connect_f=connect_f)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], connect_f=connect_f)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], connect_f=connect_f)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], connect_f=connect_f)
        self.avgpool = layer.SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc_acc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            zero_init_blocks(self, connect_f)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, connect_f=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                layer.SeqToANNContainer(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                ),
                cext_neuron.MultiStepIFNode(detach_reset=True)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, connect_f))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, connect_f=connect_f))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x.unsqueeze_(0)
        x = x.repeat(self.T, 1, 1, 1, 1)
        x = self.sn1(x)
        x = self.maxpool(x)
        x = [x, x]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feat = self.layer4(x)

        x = self.avgpool(feat[0])
        x = torch.flatten(x, 2)
        x = self.fc(x.mean(dim=0))

        x_acc = self.avgpool(feat[1])
        x_acc = torch.flatten(x_acc, 2)
        x_acc = self.fc(x_acc.mean(dim=0))

        return x, x_acc

    def forward(self, x):
        return self._forward_impl(x)


def _d_snn(block, layers, **kwargs):
    model = DSNN(block, layers, **kwargs)
    return model


def dsnn18(**kwargs):
    return _d_snn(BasicBlock, [2, 2, 2, 2], **kwargs)


def dsnn34(**kwargs):
    return _d_snn(BasicBlock, [3, 4, 6, 3], **kwargs)


def dsnn50(**kwargs):
    return _d_snn(Bottleneck, [3, 4, 6, 3], **kwargs)


def dsnn101(**kwargs):
    return _d_snn(Bottleneck, [3, 4, 23, 3], **kwargs)


def dsnn152(**kwargs):
    return _d_snn(Bottleneck, [3, 8, 36, 3], **kwargs)