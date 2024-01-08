'''
FPN in PyTorch.
See the paper "Feature Pyramid Networks for Object Detection" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torch.autograd import Variable


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, block, num_blocks, out_channels=2):
        super(FPN, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Bottom-up layers
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # resnet101 encoder
        self.model = torchvision.models.resnet101(pretrained=True) 

        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers left
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

        # Lateral layers right
        self.latlayer4 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.latlayer5 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.latlayer6 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.latlayer7 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)

        # Prediction shrink layer
        self.shrink = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)

        # Final feature layer
        self.finalconv1 = nn.Conv2d(256, 32, 3, padding=1)
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalconv3 = nn.Conv2d(32, out_channels, 3, padding=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.

        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.

        Returns:
          (Variable) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]

        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        # # Bottom-up
        # c1 = F.relu(self.bn1(self.conv1(x)))
        # c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        # c2 = self.layer1(c1)
        # c3 = self.layer2(c2)
        # c4 = self.layer3(c3)
        # c5 = self.layer4(c4)
        # print(c5.shape)

        # print(self.model)
        c1 = self.model.relu(self.model.bn1(self.model.conv1(x)))
        c1 = self.model.maxpool(c1)
        c2 = self.model.layer1(c1)
        c3 = self.model.layer2(c2)
        c4 = self.model.layer3(c3)
        c5 = self.model.layer4(c4)

        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        # r5 = self.latlayer4(p5)
        # r4 = self.latlayer5(p4)
        # r3 = self.latlayer6(p3)
        # r2 = self.latlayer7(p2)

        # _,_,H,W = r2.size()
        # m5 = F.interpolate(r5, size=(H,W), mode='bilinear', align_corners=True)
        # m4 = F.interpolate(r4, size=(H,W), mode='bilinear', align_corners=True)
        # m3 = F.interpolate(r3, size=(H,W), mode='bilinear', align_corners=True)
        # m2 = r2

        # cat = torch.cat([m5, m4, m3, m2], dim=1)

        # pre = self.shrink(cat)
        # _,_,H1,W1 = x.size()
        # upsam = F.interpolate(pre, size=(H1,W1), mode='bilinear', align_corners=True)
        # f2 = self.finalconv1(upsam)
        # feature = self.finalconv2(f2)
        # ans = self.finalconv3(feature)
        # return feature, ans

        _,_,H1,W1 = x.size()
        upsam = F.interpolate(p2, size=(H1,W1), mode='bilinear', align_corners=True)
        f2 = F.relu(self.finalconv1(upsam))
        feature = self.finalconv2(f2)
        ans = self.finalconv3(feature)
        return feature, ans

    def _initialize_weights(self, *stages):
        for modules in stages:
            for module in modules.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()


def FPN101(channels):
    # return FPN(Bottleneck, [2,2,2,2])  # resnet-18 structure,different layer
    return FPN(Bottleneck, [3,4,23,3], channels)  # resnet-101 structure


def test():
    net = FPN101()
    fms = net(Variable(torch.randn(1,3,600,900)))
    for fm in fms:
        print(fm.size())

if __name__ == "__main__":
    test()
