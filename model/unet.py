import torch
import torch.nn as nn
from torch.nn import functional as F
from vgg import VGG16


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs


class Unet(nn.Module):
    def __init__(self, num_classes=21, in_channels=3, pretrained=False):
        super(Unet, self).__init__()
        self.vgg = VGG16(pretrained=pretrained,in_channels=in_channels)
        in_filters = [192, 384, 768, 1024]
        out_filters = [64, 128, 256, 512]
        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        self.conv1_4 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.conv2_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.conv1_3 = nn.Conv2d(768, 256, kernel_size=3, padding=1)
        self.conv2_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.conv1_2 = nn.Conv2d(384, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.conv1_1 = nn.Conv2d(192, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # final conv (without any concat)
        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

    def forward(self, inputs):
        feat1 = self.vgg.features[  :4 ](inputs)
        feat2 = self.vgg.features[4 :9 ](feat1)
        feat3 = self.vgg.features[9 :16](feat2)
        feat4 = self.vgg.features[16:23](feat3)
        feat5 = self.vgg.features[23:-1](feat4)

        # up4 = self.up_concat4(feat4, feat5)
        # up3 = self.up_concat3(feat3, up4)
        # up2 = self.up_concat2(feat2, up3)
        # up1 = self.up_concat1(feat1, up2)

        _,_,H,W = feat4.size()
        up4 = torch.cat([feat4, F.interpolate(feat5, size=(H,W), mode='bilinear', align_corners=True)], 1)
        up4 = self.conv1_4(up4)
        up4 = self.conv2_4(up4)

        _,_,H,W = feat3.size()
        up3 = torch.cat([feat3, F.interpolate(up4, size=(H,W), mode='bilinear', align_corners=True)], 1)
        up3 = self.conv1_3(up3)
        up3 = self.conv2_3(up3)

        _,_,H,W = feat2.size()
        up2 = torch.cat([feat2, F.interpolate(up3, size=(H,W), mode='bilinear', align_corners=True)], 1)
        up2 = self.conv1_2(up2)
        up2 = self.conv2_2(up2)

        _,_,H,W = feat1.size()
        up1 = torch.cat([feat1, F.interpolate(up2, size=(H,W), mode='bilinear', align_corners=True)], 1)
        up1 = self.conv1_1(up1)
        up1 = self.conv2_1(up1)

        final = self.final(up1)
        
        return final

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

