from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision
from torchvision.models import densenet
from attention import *


deepsupervision = True

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class DoubleConvTiny(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConvTiny, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(in_ch, out_ch, 1),
        #     nn.BatchNorm2d(out_ch),
        #     nn.ReLU(inplace=True)
        # )
        # self.combine = nn.Sequential(
        #     nn.Conv2d(out_ch*2, out_ch, 1)
        # )

    def forward(self, input):
        return self.conv(input)
        # output_conv = self.conv(input)
        # output_conv1 = self.conv1(input)
        # output = self.combine(torch.cat([output_conv, output_conv1], 1))
        # return output

class DilationConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DilationConv, self).__init__()
        self.dilation1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, dilation=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )
        self.dilation2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=2, dilation=2),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )
        self.dilation5 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=5, dilation=5),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )
        self.dilation_cat = nn.Sequential(
            nn.Conv2d(in_ch*3, out_ch, 1, 1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.channel_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, 1, padding=0),
            nn.BatchNorm2d(out_ch)
        )
        self.shorcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, 1),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, input):
        # dila1 = self.dilation1(input)
        # dila2 = self.dilation2(input)
        # dila5 = self.dilation5(input)
        # feature_cat = torch.cat([dila1, dila2, dila5], dim=1)
        # result = self.dilation_cat(feature_cat)
        
        # dila1 = self.dilation1(input)
        # dila2 = self.dilation2(dila1)
        # dila5 = self.dilation5(dila2)
        # result = self.channel_conv(dila5)

        conv = self.conv(input)
        dila2 = self.dilation2(conv)
        dila5 = self.dilation5(dila2)

        # dila2 = self.dilation2(input)
        # dila2 = self.dilation5(dila2)

        result = self.channel_conv(dila5)
        shorcut = self.shorcut(input)
        result = nn.ReLU(inplace=True)(result + shorcut)
        return result

class SelfAttentionConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SelfAttentionConv, self).__init__()
        self.conv = nn.Sequential(
            Self_Attn(in_ch),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class NestedUNet(nn.Module):
    def __init__(self, args,in_channel,out_channel):
        super().__init__()

        self.args = args

        # nb_filter = [32, 64, 128, 256, 512]  # 原始纯DoubleConv
        # nb_filter = [64, 64, 128, 256, 512]  # resnet18,34
        # nb_filter = [64, 256, 512, 1024, 2048]  # resnet50,101,152
        # nb_filter = [64, 192, 288, 768, 2048]  # inception 5层
        # nb_filter = [192, 288, 768, 2048]  # inception 4层
        nb_filter = [64, 128, 256, 512, 512]  # vgg
        # nb_filter = [64, 128, 256, 512, 1024]  # densenet

        self.pool = nn.MaxPool2d(2, 2)
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # densenet = models.densenet121(pretrained=True)
        # resnet = models.resnet101(pretrained=True)
        # resnet = models.resnet50(pretrained=True)
        # resnet = models.resnet34(pretrained=True)
        # resnet = models.resnet18(pretrained=True)
        # inception = models.inception_v3(pretrained=True)
        vgg16 = models.vgg16(pretrained=True)
        # vgg16_bn = models.vgg16_bn(pretrained=True)
        # vgg13 = models.vgg13(pretrained=True)
        # vgg11 = models.vgg11(pretrained=True)

        # ---------- densenet ----------
        # self.conv0_0 = nn.Sequential(
        #     densenet.features[0],
        #     densenet.features[1],
        #     densenet.features[2],
        #     densenet.features[3]
        # )
        # self.conv1_0 = nn.Sequential(
        #     densenet.features[4],
        #     densenet.features[5]
        # )
        # self.conv2_0 = nn.Sequential(
        #     densenet.features[6],
        #     densenet.features[7]
        # )
        # self.conv3_0 = nn.Sequential(
        #     densenet.features[8],
        #     densenet.features[9]
        # )
        # self.conv4_0 = nn.Sequential(
        #     densenet.features[10],
        #     densenet.features[11]
        # )
        # ---------- densenet ----------

        # ---------- resnet ----------
        # self.conv0_0 = nn.Sequential(
        #     resnet.conv1,
        #     resnet.bn1,
        #     resnet.relu,
        #     resnet.maxpool
        # )

        # # self.conv0_0 = nn.Sequential(
        # #     nn.Conv2d(3,nb_filter[0],7,1,3, bias=False),
        # #     nn.BatchNorm2d(nb_filter[0]),
        # #     nn.ReLU(inplace=True),
        # #     nn.Conv2d(nb_filter[0],nb_filter[0],3,padding=1),  # 这里的卷积要不要?
        # #     nn.BatchNorm2d(nb_filter[0]),
        # #     nn.ReLU(inplace=True)
        # # )
        # self.conv1_0 = resnet.layer1
        # self.conv2_0 = resnet.layer2
        # self.conv3_0 = resnet.layer3
        # self.conv4_0 = resnet.layer4
        # ---------- resnet ----------

        # ---------- inception ----------
        # ----- 4 layer -----
        # self.conv0_0 = nn.Sequential(
        #     inception.Conv2d_1a_3x3,
        #     inception.Conv2d_2a_3x3,
        #     inception.Conv2d_2b_3x3,
        #     inception.maxpool1,
        #     inception.Conv2d_3b_1x1,
        #     inception.Conv2d_4a_3x3,
        #     inception.maxpool2
        # )
        # self.conv1_0 = nn.Sequential(
        #     inception.Mixed_5b,
        #     inception.Mixed_5c,
        #     inception.Mixed_5d
        # )
        # self.conv2_0 = nn.Sequential(
        #     inception.Mixed_6a,
        #     inception.Mixed_6b,
        #     inception.Mixed_6c,
        #     inception.Mixed_6d,
        #     inception.Mixed_6e
        # )
        # self.conv3_0 = nn.Sequential(
        #     inception.Mixed_7a,
        #     inception.Mixed_7b,
        #     inception.Mixed_7c
        # )
        # ----- 4 layer -----
        # ----- 5 layer -----
        # self.conv0_0 = nn.Sequential(
        #     inception.Conv2d_1a_3x3,
        #     inception.Conv2d_2a_3x3,
        #     inception.Conv2d_2b_3x3,
        #     inception.maxpool1
        # )
        # self.conv1_0 = nn.Sequential(
        #     inception.Conv2d_3b_1x1,
        #     inception.Conv2d_4a_3x3,
        #     inception.maxpool2
        # )
        # self.conv2_0 = nn.Sequential(
        #     inception.Mixed_5b,
        #     inception.Mixed_5c,
        #     inception.Mixed_5d
        # )
        # self.conv3_0 = nn.Sequential(
        #     inception.Mixed_6a,
        #     inception.Mixed_6b,
        #     inception.Mixed_6c,
        #     inception.Mixed_6d,
        #     inception.Mixed_6e
        # )
        # self.conv4_0 = nn.Sequential(
        #     inception.Mixed_7a,
        #     inception.Mixed_7b,
        #     inception.Mixed_7c
        # )
        # ----- 5 layer -----
        # ---------- inception ----------

        # ---------- vgg16 ----------
        self.conv0_0 = nn.Sequential(
            vgg16.features[0],
            vgg16.features[1],
            vgg16.features[2],
            vgg16.features[3],
            vgg16.features[4]
        )
        self.conv1_0 = nn.Sequential(
            vgg16.features[5],
            vgg16.features[6],
            vgg16.features[7],
            vgg16.features[8],
            vgg16.features[9]
        )
        self.conv2_0 = nn.Sequential(
            vgg16.features[10],
            vgg16.features[11],
            vgg16.features[12],
            vgg16.features[13],
            vgg16.features[14],
            vgg16.features[15],
            vgg16.features[16]
        )
        self.conv3_0 = nn.Sequential(
            vgg16.features[17],
            vgg16.features[18],
            vgg16.features[19],
            vgg16.features[20],
            vgg16.features[21],
            vgg16.features[22],
            vgg16.features[23]
        )
        self.conv4_0 = nn.Sequential(
            vgg16.features[24],
            vgg16.features[25],
            vgg16.features[26],
            vgg16.features[27],
            vgg16.features[28],
            vgg16.features[29],
            vgg16.features[30]
        )
        # ---------- vgg16 ----------

        # ---------- vgg16_bn ----------
        # self.conv0_0 = nn.Sequential(
        #     vgg16_bn.features[0],
        #     vgg16_bn.features[1],
        #     vgg16_bn.features[2],
        #     vgg16_bn.features[3],
        #     vgg16_bn.features[4],
        #     vgg16_bn.features[5],
        #     vgg16_bn.features[6]
        # )
        # self.conv1_0 = nn.Sequential(
        #     vgg16_bn.features[7],
        #     vgg16_bn.features[8],
        #     vgg16_bn.features[9],
        #     vgg16_bn.features[10],
        #     vgg16_bn.features[11],
        #     vgg16_bn.features[12],
        #     vgg16_bn.features[13]
        # )
        # self.conv2_0 = nn.Sequential(
        #     vgg16_bn.features[14],
        #     vgg16_bn.features[15],
        #     vgg16_bn.features[16],
        #     vgg16_bn.features[17],
        #     vgg16_bn.features[18],
        #     vgg16_bn.features[19],
        #     vgg16_bn.features[20],
        #     vgg16_bn.features[21],
        #     vgg16_bn.features[22],
        #     vgg16_bn.features[23]
        # )
        # self.conv3_0 = nn.Sequential(
        #     vgg16_bn.features[24],
        #     vgg16_bn.features[25],
        #     vgg16_bn.features[26],
        #     vgg16_bn.features[27],
        #     vgg16_bn.features[28],
        #     vgg16_bn.features[29],
        #     vgg16_bn.features[30],
        #     vgg16_bn.features[31],
        #     vgg16_bn.features[32],
        #     vgg16_bn.features[33]
        # )
        # self.conv4_0 = nn.Sequential(
        #     vgg16_bn.features[34],
        #     vgg16_bn.features[35],
        #     vgg16_bn.features[36],
        #     vgg16_bn.features[37],
        #     vgg16_bn.features[38],
        #     vgg16_bn.features[39],
        #     vgg16_bn.features[40],
        #     vgg16_bn.features[41],
        #     vgg16_bn.features[42],
        #     vgg16_bn.features[43]
        # )
        # ---------- vgg16_bn ----------

        # ---------- vgg13 ----------
        # self.conv0_0 = nn.Sequential(
        #     vgg13.features[0],
        #     vgg13.features[1],
        #     vgg13.features[2],
        #     vgg13.features[3],
        #     vgg13.features[4]
        # )
        # self.conv1_0 = nn.Sequential(
        #     vgg13.features[5],
        #     vgg13.features[6],
        #     vgg13.features[7],
        #     vgg13.features[8],
        #     vgg13.features[9]
        # )
        # self.conv2_0 = nn.Sequential(
        #     vgg13.features[10],
        #     vgg13.features[11],
        #     vgg13.features[12],
        #     vgg13.features[13],
        #     vgg13.features[14]
        # )
        # self.conv3_0 = nn.Sequential(
        #     vgg13.features[15],
        #     vgg13.features[16],
        #     vgg13.features[17],
        #     vgg13.features[18],
        #     vgg13.features[19]
        # )
        # self.conv4_0 = nn.Sequential(
        #     vgg13.features[20],
        #     vgg13.features[21],
        #     vgg13.features[22],
        #     vgg13.features[23],
        #     vgg13.features[24]
        # )
        # ---------- vgg13 ----------

        # ---------- vgg11 ----------
        # self.conv0_0 = nn.Sequential(
        #     vgg11.features[0],
        #     vgg11.features[1],
        #     vgg11.features[2]
        # )
        # self.conv1_0 = nn.Sequential(
        #     vgg11.features[3],
        #     vgg11.features[4],
        #     vgg11.features[5]
        # )
        # self.conv2_0 = nn.Sequential(
        #     vgg11.features[6],
        #     vgg11.features[7],
        #     vgg11.features[8],
        #     vgg11.features[9],
        #     vgg11.features[10]
        # )
        # self.conv3_0 = nn.Sequential(
        #     vgg11.features[11],
        #     vgg11.features[12],
        #     vgg11.features[13],
        #     vgg11.features[14],
        #     vgg11.features[15]
        # )
        # self.conv4_0 = nn.Sequential(
        #     vgg11.features[16],
        #     vgg11.features[17],
        #     vgg11.features[18],
        #     vgg11.features[19],
        #     vgg11.features[20]
        # )
        # ---------- vgg11 ----------

        # ---------- dilation conv(HDC) ----------
        # self.conv1_0 = DilationConv(nb_filter[0], nb_filter[1])
        # self.conv2_0 = DilationConv(nb_filter[1], nb_filter[2])
        # self.conv3_0 = DilationConv(nb_filter[2], nb_filter[3])
        # self.conv4_0 = DilationConv(nb_filter[3], nb_filter[4])
        # ---------- dilation conv(HDC) ----------

        # ---------- 原始纯DoubleConv ----------
        # self.conv0_0 = DoubleConv(in_channel, nb_filter[0])
        # # self.conv0_0 = nn.Sequential(
        # #     nn.Conv2d(3,32,7,1,3, bias=False),
        # #     # nn.BatchNorm2d(nb_filter[0], eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        # #     nn.BatchNorm2d(nb_filter[0]),
        # #     nn.ReLU(inplace=True),
        # #     nn.Conv2d(nb_filter[0],nb_filter[0],3,padding=1),
        # #     nn.BatchNorm2d(nb_filter[0]),
        # #     nn.ReLU(inplace=True)
        # # )
        # self.conv1_0 = DoubleConv(nb_filter[0], nb_filter[1])
        # self.conv2_0 = DoubleConv(nb_filter[1], nb_filter[2])
        # self.conv3_0 = DoubleConv(nb_filter[2], nb_filter[3])
        # self.conv4_0 = DoubleConv(nb_filter[3], nb_filter[4])
        # ---------- 原始纯DoubleConv ----------

        self.conv0_1 = DoubleConvTiny(nb_filter[0]+nb_filter[1], nb_filter[0])
        self.conv1_1 = DoubleConvTiny(nb_filter[1]+nb_filter[2], nb_filter[1])
        self.conv2_1 = DoubleConvTiny(nb_filter[2]+nb_filter[3], nb_filter[2])
        self.conv3_1 = DoubleConvTiny(nb_filter[3]+nb_filter[4], nb_filter[3])

        self.conv0_2 = DoubleConvTiny(nb_filter[0]*2+nb_filter[1], nb_filter[0])
        self.conv1_2 = DoubleConvTiny(nb_filter[1]*2+nb_filter[2], nb_filter[1])
        self.conv2_2 = DoubleConvTiny(nb_filter[2]*2+nb_filter[3], nb_filter[2])

        self.conv0_3 = DoubleConvTiny(nb_filter[0]*3+nb_filter[1], nb_filter[0])
        self.conv1_3 = DoubleConvTiny(nb_filter[1]*3+nb_filter[2], nb_filter[1])

        self.conv0_4 = DoubleConvTiny(nb_filter[0]*4+nb_filter[1], nb_filter[0])
        self.sigmoid = nn.Sigmoid()
        if deepsupervision:
            self.refine1 = nn.Conv2d(nb_filter[0], nb_filter[0], 3, 1, 1)
            self.refine2 = nn.Conv2d(nb_filter[0], nb_filter[0], 3, 1, 1)
            self.refine3 = nn.Conv2d(nb_filter[0], nb_filter[0], 3, 1, 1)
            self.refine4 = nn.Conv2d(nb_filter[0], nb_filter[0], 3, 1, 1)
            self.final1 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
        # self.attention1 = Self_Attn(nb_filter[1])
        # self.attention2 = Self_Attn(nb_filter[2])
        # self.attention3 = Self_Attn(nb_filter[3])
        # self.attention4 = Self_Attn(nb_filter[4])


    def forward(self, input):
        # ----- backbone -----
        x0_0 = self.conv0_0(input)

        x1_0 = self.conv1_0(x0_0)  # resnet, inception, vgg
        _,_,H,W = x0_0.size()
        x0_1 = self.conv0_1(torch.cat([x0_0, F.interpolate(x1_0, size=(H,W), mode='bilinear', align_corners=True)], 1))

        x2_0 = self.conv2_0(x1_0)
        _,_,H,W = x1_0.size()
        x1_1 = self.conv1_1(torch.cat([x1_0, F.interpolate(x2_0, size=(H,W), mode='bilinear', align_corners=True)], 1))
        _,_,H,W = x0_0.size()
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, F.interpolate(x1_1, size=(H,W), mode='bilinear', align_corners=True)], 1))

        # save_x0_2 = x0_2

        x3_0 = self.conv3_0(x2_0)
        _,_,H,W = x2_0.size()
        x2_1 = self.conv2_1(torch.cat([x2_0, F.interpolate(x3_0, size=(H,W), mode='bilinear', align_corners=True)], 1))
        _,_,H,W = x1_0.size()
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, F.interpolate(x2_1, size=(H,W), mode='bilinear', align_corners=True)], 1))
        _,_,H,W = x0_0.size()
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, F.interpolate(x1_2, size=(H,W), mode='bilinear', align_corners=True)], 1))

        x4_0 = self.conv4_0(x3_0)
        _,_,H,W = x3_0.size()
        x3_1 = self.conv3_1(torch.cat([x3_0, F.interpolate(x4_0, size=(H,W), mode='bilinear', align_corners=True)], 1))
        _,_,H,W = x2_0.size()
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, F.interpolate(x3_1, size=(H,W), mode='bilinear', align_corners=True)], 1))
        _,_,H,W = x1_0.size()
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, F.interpolate(x2_2, size=(H,W), mode='bilinear', align_corners=True)], 1))
        _,_,H,W = x0_0.size()
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, F.interpolate(x1_3, size=(H,W), mode='bilinear', align_corners=True)], 1))
        # ----- backbone -----

        # ----- oirigin DoubleConv -----
        # x0_0 = self.conv0_0(input)
        # x0_0 = self.pool(x0_0)
        # x1_0 = self.conv1_0(x0_0)  # origin DoubleConv, vgg(效果差)
        # _,_,H,W = x0_0.size()
        # x0_1 = self.conv0_1(torch.cat([x0_0, F.interpolate(x1_0, size=(H,W), mode='bilinear', align_corners=True)], 1))

        # x2_0 = self.conv2_0(self.pool(x1_0))  # origin DoubleConv
        # x2_0 = self.conv2_0(x1_0)
        # _,_,H,W = x1_0.size()
        # x1_1 = self.conv1_1(torch.cat([x1_0, F.interpolate(x2_0, size=(H,W), mode='bilinear', align_corners=True)], 1))
        # _,_,H,W = x0_0.size()
        # x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, F.interpolate(x1_1, size=(H,W), mode='bilinear', align_corners=True)], 1))

        # x3_0 = self.conv3_0(self.pool(x2_0))  # origin DoubleConv
        # x3_0 = self.conv3_0(x2_0)
        # _,_,H,W = x2_0.size()
        # x2_1 = self.conv2_1(torch.cat([x2_0, F.interpolate(x3_0, size=(H,W), mode='bilinear', align_corners=True)], 1))
        # _,_,H,W = x1_0.size()
        # x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, F.interpolate(x2_1, size=(H,W), mode='bilinear', align_corners=True)], 1))
        # _,_,H,W = x0_0.size()
        # x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, F.interpolate(x1_2, size=(H,W), mode='bilinear', align_corners=True)], 1))

        # x4_0 = self.conv4_0(self.pool(x3_0))  # origin DoubleConv
        # x4_0 = self.conv4_0(x3_0)
        # _,_,H,W = x3_0.size()
        # x3_1 = self.conv3_1(torch.cat([x3_0, F.interpolate(x4_0, size=(H,W), mode='bilinear', align_corners=True)], 1))
        # _,_,H,W = x2_0.size()
        # x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, F.interpolate(x3_1, size=(H,W), mode='bilinear', align_corners=True)], 1))
        # _,_,H,W = x1_0.size()
        # x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, F.interpolate(x2_2, size=(H,W), mode='bilinear', align_corners=True)], 1))
        # _,_,H,W = x0_0.size()
        # x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, F.interpolate(x1_3, size=(H,W), mode='bilinear', align_corners=True)], 1))
        # ----- oirigin DoubleConv -----

        # out_feature = x0_4

        if deepsupervision:
            _,_,H1,W1 = x0_0.size()
            _,_,H2,W2 = input.size()
            if H1 != H2 or W1 != W2:
                x0_1 = F.interpolate(x0_1, size=(H2,W2), mode='bilinear', align_corners=True)
                # x0_1_save = x0_1
                x0_1 = self.refine1(x0_1)
                x0_2 = F.interpolate(x0_2, size=(H2,W2), mode='bilinear', align_corners=True)
                # x0_2_save = x0_2
                x0_2 = self.refine2(x0_2)
                x0_3 = F.interpolate(x0_3, size=(H2,W2), mode='bilinear', align_corners=True)
                # x0_3_save = x0_3
                x0_3 = self.refine3(x0_3)
                # out_feature = x0_3
                x0_4 = F.interpolate(x0_4, size=(H2,W2), mode='bilinear', align_corners=True)
                # x0_4_save = x0_4
                x0_4 = self.refine4(x0_4)
                # out_feature = x0_4
            out_low_feature1 = F.interpolate(x0_0, size=(H2,W2), mode='bilinear', align_corners=True)
            # out_low_feature2 = F.interpolate(x1_0, size=(H2,W2), mode='bilinear', align_corners=True)
            # out_deep_feature = F.interpolate(x1_3, size=(H2,W2), mode='bilinear', align_corners=True)
            # out_deep_feature1 = F.interpolate(x4_0, size=(H2,W2), mode='bilinear', align_corners=True)
            # out_low_feature1 = F.interpolate(x4_0, size=(H2,W2), mode='bilinear', align_corners=True)
            # out_low_feature2 = F.interpolate(x0_1_save, size=(H2,W2), mode='bilinear', align_corners=True)
            # out_feature = torch.cat([x0_3,x0_4], 1)
            # out_feature = torch.cat([out_low_feature,x0_4], 1)  # 伪标注图可视化效果还不错(可能卷过头了)
            # out_feature = torch.cat([out_low_feature,x0_3], 1)  # 伪标注图可视化效果还不错
            out_feature = torch.cat([out_low_feature1,x0_1,x0_2,x0_3,x0_4], 1)
            # out_feature = torch.cat([out_low_feature,out_deep_feature,x0_4], 1)
            # out_feature = torch.cat([out_low_feature,out_deep_feature], 1)
            # out_feature = torch.cat([out_low_feature,out_deep_feature1], 1)
            # out_feature = torch.cat([out_low_feature,x0_1], 1)
            # out_feature = torch.cat([out_low_feature,x0_2], 1)
            # out_feature = torch.cat([out_low_feature,out_low_feature1], 1)
            # out_feature = torch.cat([x0_3_save,x0_4_save], 1)
            # out_feature = torch.cat([out_low_feature,x0_4_save], 1)
            # out_feature = torch.cat([x0_1,x0_4], 1)  # 伪标柱图可视化效果也还不错,主要问题是用了深监督里的两个支作为判断特征,可能会完全收到前期学习的影响?
            # out_feature = torch.cat([out_low_feature2,x0_4_save], 1)
            # out_feature = out_low_feature
            # out_feature = out_low_feature2
            # out_feature = x0_4
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            # return [output1, output2, output3], x3_0, x0_3
            # return [output1, output2, output3, output4], x4_0, x0_4, x1_0
            # return [output1, output2, output3, output4], save_x0_2, out_feature, x0_0
            return [output1, output2, output3, output4], None, out_feature, None  # 尝试再节省下显存
            # return [output1, output2, output3, output4], x4_0, spatialattention, x0_4
        else:
            output = self.final(x0_4)
            return output, x4_0, x0_4
            # return output, x4_0, spatialattention, x0_4
