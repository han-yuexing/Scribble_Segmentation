# ----- version1 -----
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
# from modeling.aspp import build_aspp
# from modeling.decoder import build_decoder
# from modeling.backbone import build_backbone

# class DeepLab(nn.Module):
#     def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
#                  sync_bn=True, freeze_bn=False):
#         super(DeepLab, self).__init__()
#         if backbone == 'drn':
#             output_stride = 8

#         

#         self.backbone = build_backbone(backbone, output_stride, BatchNorm)
#         self.aspp = build_aspp(backbone, output_stride, BatchNorm, momentum=0.0003)
#         self.decoder = build_decoder(num_classes, backbone, BatchNorm)

#         self.freeze_bn = freeze_bn

#     def forward(self, input):
#         x, low_level_feat = self.backbone(input)
#         x = self.aspp(x)
#         x = self.decoder(x, low_level_feat)
#         x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

#         return x

#     def freeze_bn(self):
#         for m in self.modules():
#             if isinstance(m, SynchronizedBatchNorm2d):
#                 m.eval()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.eval()

#     def get_1x_lr_params(self):
#         modules = [self.backbone]
#         for i in range(len(modules)):
#             for m in modules[i].named_modules():
#                 if self.freeze_bn:
#                     if isinstance(m[1], nn.Conv2d):
#                         for p in m[1].parameters():
#                             if p.requires_grad:
#                                 yield p
#                 else:
#                     if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
#                             or isinstance(m[1], nn.BatchNorm2d):
#                         for p in m[1].parameters():
#                             if p.requires_grad:
#                                 yield p

#     def get_10x_lr_params(self):
#         modules = [self.aspp, self.decoder]
#         for i in range(len(modules)):
#             for m in modules[i].named_modules():
#                 if self.freeze_bn:
#                     if isinstance(m[1], nn.Conv2d):
#                         for p in m[1].parameters():
#                             if p.requires_grad:
#                                 yield p
#                 else:
#                     if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
#                             or isinstance(m[1], nn.BatchNorm2d):
#                         for p in m[1].parameters():
#                             if p.requires_grad:
#                                 yield p

# if __name__ == "__main__":
#     model = DeepLab(backbone='xception', output_stride=16)
#     model.eval()
#     input = torch.rand(1, 3, 513, 513)
#     output = model(input)
#     print(output.size())
# ----- version1 -----

# ----- version2 -----
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp import build_aspp
from modeling.backbone import build_backbone

class deeplabv3plus(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21, sync_bn=True, freeze_bn=False):
        super(deeplabv3plus, self).__init__()
        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d
        
        MODEL_OUTPUT_STRIDE = 16
        MODEL_ASPP_OUTDIM = 256
        MODEL_SHORTCUT_DIM = 48
        MODEL_SHORTCUT_KERNEL = 1
        MODEL_NUM_CLASSES = 21
        TRAIN_BN_MOM = 0.0003

        self.backbone = None		
        self.backbone_layers = None
        input_channel = 2048		
        self.aspp = build_aspp(backbone, output_stride, BatchNorm, momentum=TRAIN_BN_MOM)
        # self.dropout1 = nn.Dropout(0.5)
        # self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
        # self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=MODEL_OUTPUT_STRIDE//4)

        indim = 256
        self.shortcut_conv = nn.Sequential(
                nn.Conv2d(indim, MODEL_SHORTCUT_DIM, MODEL_SHORTCUT_KERNEL, 1, padding=MODEL_SHORTCUT_KERNEL//2,bias=True),
                SynchronizedBatchNorm2d(MODEL_SHORTCUT_DIM, momentum=TRAIN_BN_MOM),
                nn.ReLU(inplace=True),		
        )		
        self.cat_conv = nn.Sequential(
                nn.Conv2d(MODEL_ASPP_OUTDIM+MODEL_SHORTCUT_DIM, MODEL_ASPP_OUTDIM, 3, 1, padding=1,bias=True),
                SynchronizedBatchNorm2d(MODEL_ASPP_OUTDIM, momentum=TRAIN_BN_MOM),
                nn.ReLU(inplace=True),
                # nn.Dropout(0.5),
                nn.Conv2d(MODEL_ASPP_OUTDIM, MODEL_ASPP_OUTDIM, 3, 1, padding=1,bias=True),
                SynchronizedBatchNorm2d(MODEL_ASPP_OUTDIM, momentum=TRAIN_BN_MOM),
                nn.ReLU(inplace=True),
                # nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(MODEL_ASPP_OUTDIM, MODEL_NUM_CLASSES, 1, 1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SynchronizedBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.backbone_layers = self.backbone.get_layers()

    def forward(self, x):
        x_bottom = self.backbone(x)
        layers = self.backbone.get_layers()
        feature_aspp = self.aspp(layers[-1])
        # feature_aspp = self.dropout1(feature_aspp)
        feature_aspp = F.interpolate(feature_aspp, layers[0].size()[2:], mode='bilinear', align_corners=True)
        # feature_aspp = self.upsample_sub(feature_aspp)

        feature_shallow = self.shortcut_conv(layers[0])
        feature_cat = torch.cat([feature_aspp,feature_shallow],1)
        feature = self.cat_conv(feature_cat) 
        result = self.cls_conv(feature)
        result = F.interpolate(result, x.size()[2:], mode='bilinear', align_corners=True)
        # result = self.upsample4(result)
        return result
# ----- version2 -----