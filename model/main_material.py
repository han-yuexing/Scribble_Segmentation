import os
import csv
import math
import torch
import random
import mat4py
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Function
from torch.autograd import Variable
from dataset import *
from metrics import *
from unetpp import NestedUNet
from unet import *
from resnet34unet import *
from fpn import FPN101 as resnet101_fpn
from deeplabv3plus import *
import torch.backends.cudnn as cudnn
from dataloader import DeeplabDataset, deeplab_dataset_collate
from modeling.sync_batchnorm.replicate import patch_replication_callback
from attention import *
from torch import nn
from sklearn.preprocessing import MinMaxScaler
# import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian

from comparison.unet3plus import UNet_3Plus_DeepSup

plt.switch_backend('agg')


def getArgs():
    parse = argparse.ArgumentParser()
    parse.add_argument("--action", type=str, help="train/test/train&test", default="train&test")
    parse.add_argument("--epoch", type=int, default=120)  # 180,160,150,120,100,70
    parse.add_argument('--arch', '-a', metavar='ARCH', default='unet++',
                       help='resnet34unet/unet++/fpn/deeplabv3+/vgg16unet/unet3+')  # resnet34unet内部包含反卷积,不能处理边长分别为奇偶的情况
    parse.add_argument("--batch_size", type=int, default=1)
    parse.add_argument('--dataset', default='ceramic',
                       help='dataset name:TiAl/TiAl_full/TiAl_cam/micrograph/ceramic/pascal/pascal_tiny/mix/choosevoc')  # 原本名字叫pascalvoc2012和pascalvoc2012_tiny,后来改简单了,如果有模型加在不了,可能就是这边名字改了的原因
    parse.add_argument("--log_dir", default='result/log', help="log dir")
    parse.add_argument("--classes", default=2, help="number of output classes")
    parse.add_argument("--gpu_ids", default="0", help="gpu ids")
    parse.add_argument("--decay_lr_ratio", default=1, help="gamma for learning rate decay")  # 0.9
    parse.add_argument("--decay_lr_step", default=40, help="epoch for learning rate decay")  # 50
    parse.add_argument("--lr", default=0.003, help="learning rate")  # 分段的学习率已经写在adjust_lr函数里面了
    parse.add_argument("--attention_decay_lr_ratio", default=1, help="gamma for learning rate decay in self attention")  # 0.9
    parse.add_argument("--attention_decay_lr_step", default=40, help="epoch for learning rate decay in self attention")  # 40
    parse.add_argument("--attention_lr", default=0.01, help="learning rate for modified self attention")
    parse.add_argument("--T", default=0.5, help="hyperparameter for sharpen in self attention")
    parse.add_argument("--aux", default=True, help="whether to use auxiliary branch or not")
    args = parse.parse_args()
    return args

def getLog(args):
    dirname = os.path.join(args.log_dir,args.arch,str(args.batch_size),str(args.dataset),str(args.epoch))
    filename = dirname +'/log.log'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    logging.basicConfig(
            filename=filename,
            level=logging.DEBUG,
            format='%(asctime)s:%(levelname)s:%(message)s'
        )
    return logging

def getModel(args):
    if args.arch == 'unet++':
        model = NestedUNet(args,3,args.classes).to(device)
    elif args.arch == 'resnet34unet':  # resnet34unet内部包含反卷积,不能处理边长分别为奇偶的情况
        model = resnet34_unet(args.classes).to(device)
    elif args.arch == 'fpn':
        model = resnet101_fpn(args.classes).to(device)
    elif args.arch == 'deeplabv3+':
        model = deeplabv3plus(backbone='xception', num_classes=args.classes).to(device)
        # model = DeepLab(backbone='xception', num_classes=args.classes).to(device)
        # model = DeepLab(backbone='resnet', num_classes=args.classes).to(device)
    elif args.arch == 'vgg16unet':
        model = Unet(num_classes=21, in_channels=3, pretrained=True).to(device)
    elif args.arch == 'unet3+':
        model = UNet_3Plus_DeepSup(3,3)
    return model

def getSelfAttnModel(args):
    #TODO:attention_dim以及输入维度测试(attention_dim应小于输入维度)
    # model = Map_Self_Attention(64,128)  # attention_dim=128:https://github.com/googleinterns/wss/blob/b01acaf0317a48ed8b4a1a239a9a90bcfb0844e8/core/train_utils_core.py#L134
    # model = Custom_Self_Attention(64,32,args.classes)
    # model = Custom_Self_Attention(128,32,args.classes)
    model = Custom_Self_Attention(64*5,32,args.classes)
    # model = Custom_Self_Attention(32*5,32,args.classes)
    # model = Custom_Self_Attention(64+128,32,args.classes)
    # model = Custom_Self_Attention(64+512,32,args.classes)
    # model = Custom_Self_Attention(64+128+64,32,args.classes)
    # model = Custom_Self_Attention(64+512,32,args.classes)
    # 同一个相可能会有多个方向,由于卷积核本身无法实现对不同方向的相同相获取同种特征,因此需要依赖于映射到的特征空间将不同方向的相归类在一起,由于考虑到本身64维特征已足够区分不同相,甚至还能区分出方向,因此本质上要想区分不同相并不需要那么高的维度,因此这里选了向下降至32维
    return model

def getDataset(args):
    train_dataloaders, val_dataloaders ,test_dataloaders= None,None,None
    if args.dataset == 'TiAl':
        train_dataset = TiAlDataset(r'train', classes=args.classes, aug=True)
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_dataset = TiAlDataset(r"val", classes=args.classes)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        # val_dataloaders = None
        test_dataset = TiAlDataset(r"test", classes=args.classes)
        test_dataloaders = DataLoader(test_dataset, batch_size=1)
    if args.dataset == 'TiAl_full':
        train_dataset = TiAlDatasetFull(r'train', classes=args.classes, aug=True)
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_dataset = TiAlDatasetFull(r"val", classes=args.classes)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        # val_dataloaders = None
        test_dataset = TiAlDatasetFull(r"test", classes=args.classes)
        test_dataloaders = DataLoader(test_dataset, batch_size=1)
    if args.dataset == 'TiAl_cam':
        train_dataset = TiAlDatasetCAM(r'train', classes=args.classes, aug=True)
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_dataset = TiAlDatasetCAM(r"val", classes=args.classes)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        # val_dataloaders = None
        test_dataset = TiAlDatasetCAM(r"test", classes=args.classes)
        test_dataloaders = DataLoader(test_dataset, batch_size=1)
    if args.dataset == 'ceramic':
        train_dataset = CeramicDataset(r'train', classes=args.classes, aug=True)
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_dataset = CeramicDataset(r"val", classes=args.classes)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        # val_dataloaders = None
        test_dataset = CeramicDataset(r"test", classes=args.classes)
        test_dataloaders = DataLoader(test_dataset, batch_size=1)
    if args.dataset == 'mix':
        train_dataset = MixDataset(r'train', classes=args.classes, aug=True)
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_dataset = MixDataset(r"val", classes=args.classes)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        # val_dataloaders = None
        test_dataset = MixDataset(r"test", classes=args.classes)
        test_dataloaders = DataLoader(test_dataset, batch_size=1)
    if args.dataset == 'micrograph':
        train_dataset = micrographDataset(r'train', classes=args.classes, aug=True)
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_dataset = micrographDataset(r"val", classes=args.classes)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        # val_dataloaders = None
        test_dataset = micrographDataset(r"test", classes=args.classes)
        test_dataloaders = DataLoader(test_dataset, batch_size=1)
    if args.dataset == 'pascal':
        # train_dataset = pascalDataset(r'train', aug=True)
        train_dataset = pascalDataset(r'train', classes=args.classes) 
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_dataset = pascalDataset(r"val", classes=args.classes)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        test_dataset = pascalDataset(r"test", classes=args.classes)
        test_dataloaders = DataLoader(test_dataset, batch_size=1)
    if args.dataset == 'pascal_tiny':
        train_dataset = pascalDatasetTiny(r'train', classes=args.classes)
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_dataset = pascalDatasetTiny(r"val", classes=args.classes)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        test_dataset = pascalDatasetTiny(r"test", classes=args.classes)
        test_dataloaders = DataLoader(test_dataset, batch_size=1)
    if args.dataset == 'choosevoc':
        train_dataset = ChooseVocDataset(r'train', classes=args.classes, aug=True)
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_dataset = ChooseVocDataset(r"val", classes=args.classes)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        # val_dataloaders = None
        test_dataset = ChooseVocDataset(r"test", classes=args.classes)
        test_dataloaders = DataLoader(test_dataset, batch_size=1)
    return train_dataloaders,val_dataloaders,test_dataloaders
    

def val(model, val_dataloaders, args, best_miou,epoch):
    model= model.eval()
    with torch.no_grad():
        i=0   #验证集中第i张图
        miou_list = []
        predict = []
        num = len(val_dataloaders)  #验证集图片的总数
        for x, mask, _, _ in val_dataloaders:
            x = x.to(device)
            if args.arch == 'fpn':
                _, y = model(x)
            else:
                output, _, feature, _ = model(x)
                # output = model(x)  # unet3+
                
                predict.append(output[-1])
                y = output[-1]
                
            img_y = y.max(1)[1].squeeze().cpu().data.numpy()  #输入损失函数之前要把预测图变成numpy格式，且为了跟训练图对应，要额外加多一维表示batchsize
            current_miou, current_iou = get_miou(mask[0],img_y, args.classes)
            miou_list.append(current_miou)
            # miou_list.append(get_miou(mask[0],img_y, args.classes))  #获取当前预测图的miou，并加到总miou中
            if i < num:
                i += 1   #处理验证集下一张图
        aver_miou = np.array(miou_list).mean()
        print('Each image Miou list', miou_list)  # XY-11,XZ-2,XZ-6
        print('Miou=%f' % (aver_miou))
        logging.info('Miou=%f' % (aver_miou))
        if aver_miou > best_miou:
            print("new best miou=%f" % (aver_miou))
            logging.info("new best miou=%f" % (aver_miou))
        torch.save(model.state_dict(), r'../saved_model/'+str(args.arch)+'_'+str(args.batch_size)+'_'+str(args.dataset)+'_'+str(args.epoch)+'_'+str(args.lr)+'.pth')
        for i in range(len(predict)):
            predict_tmp = predict[i][0]
            predict_tmp = predict_tmp.unsqueeze(dim=0)
            pre_label = predict_tmp.max(1)[1].squeeze().cpu().data.numpy()
            pre_label = np.asarray(pre_label, dtype=np.uint8)
            # pre_label *= 100
            pre_label *= 60
            cv2.imwrite(args.dataset + "_" + str(i) + ".png", pre_label)
            if epoch == 99 or epoch == 120:
                cv2.imwrite(args.dataset + "_" + str(i) + "_" + str(epoch) + ".png", pre_label)
        return aver_miou

def train(model, attention_model, criterion, loss_fn, loss_sup, optimizer, optimizer_attn, train_dataloader, val_dataloader, lr_scheduler, lr_scheduler_attn, devices_list, args):
    num_epochs = args.epoch
    max_itr = num_epochs * len(train_dataloader)
    loss_list = []
    miou_list = []

    fmap_block = list()
    grad_block = list()
    def backward_hook(module, grad_in, grad_out):
        if len(grad_block) == 0:
            grad_block.append(grad_out[0].detach())
        else:
            grad_block[0] = grad_out[0].detach()


    def forward_hook(module, input, output):
        if len(fmap_block) == 0:
            fmap_block.append(output)
        else:
            fmap_block[0] = output
    
    # -----train by epoch-----
    # best_pred = 0.0
    best_miou = 0.0
    itr = 0

    last_loss = 100  # 用来验证loss突然上升的问题
    save_99miou = 0
    save_99bestmiou = 0
    vis_index1 = None
    
    # cus_loss = Custom_loss()

    saved_class_feature = None
    saved_class_label = None
    for epoch in range(num_epochs + 1):
        itr += 1
        model = model.train()
        # attention_model = attention_model.train()
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch, num_epochs))
        logging.info('Epoch {}/{}'.format(epoch, num_epochs))
        dt_size = len(train_dataloader.dataset)
        epoch_loss = 0
        step = 0
        batch_index = 0
        # last_output_cls = None
        # last_output_prob = None
        # now_lr = adjust_lr(optimizer, epoch, num_epochs + 1, args)
        for x, aug_x, y, mask, path1, path2 in train_dataloader:
        # for x, aug_x, x_small, x_large, y, y_small, y_large, path1, path2 in train_dataloader:
            # image_0 = x.squeeze(0).permute(1,2,0).numpy()
            # image_1 = aug_x.squeeze(0).permute(1,2,0).numpy()
            # image_2 = y.squeeze(0).numpy()
            # image_3 = mask.squeeze(0).numpy()
            # print(np.unique(image_3))

            # cv2.imwrite("testrotate0.png", image_0 * 255)
            # # cv2.imwrite("testrotate1.png", image_1)
            # cv2.imwrite("testrotate2.png", image_2)
            # cv2.imwrite("testrotate3.png", image_3)

            # print("saved!")
            # exit(0)

            now_lr = adjust_lr(optimizer, itr, max_itr, args, model)
            print("now_lr:", now_lr)

            # x = x.squeeze()  # 同批加flip两张图需要这个--
            # y = y.squeeze()

            step += 1
            # itr += 1  # TODO:最早是在这里用的itr,然后就导致adjust_lr里面要除图的个数
            inputs = x.to(device)
            # inputs_small = x_small.to(device)
            # inputs_large = x_large.to(device)
            # aug_inputs = aug_x.to(device)
            labels = y.to(device)
            mask = mask.squeeze(0)
            mask = mask.to(device)
            # labels_small = y_small.to(device)
            # labels_large = y_large.to(device)
            # lr_scheduler(optimizer, batch_index, epoch, best_pred)
            batch_index += 1

            # model.module.conv0_4.register_forward_hook(forward_hook)
            # model.module.conv0_4.register_backward_hook(backward_hook)
            # model.module.final4.register_forward_hook(forward_hook)
            # model.module.final4.register_backward_hook(backward_hook)

            # zero the parameter gradients
            optimizer.zero_grad()
            labels = labels.squeeze(dim=1).long()
            '''
            if args.arch == 'fpn':
                feature, output = model(inputs)
                # aug_feature, aug_output = model(aug_inputs)
                # -----划线部分像素点精确监督-----  (后续考虑膨胀或者动态修改等)
                loss = criterion(output, labels)
                # aug_loss = criterion(aug_output, labels)
                # -----划线部分像素点精确监督-----
                # -----度量无标注像素点数据增强前后相似性-----  (找一个合理的相似性度量方案)

                # -----度量无标注像素点数据增强前后相似性-----
                # 这里应该至少还要个啥同类内方差小,多类中心距离大这种,先跑起来再说吧
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            else:
            '''
            # output, atten = model(inputs)

            # output_small, atten_small = model(inputs_small)

            # output_large, atten_large = model(inputs_large)

            # atten_input = []
            # _,_,H,W = atten.size()
            # atten_input.append(F.interpolate(atten_small, size=(H,W), mode='bilinear', align_corners=True))
            # atten_input.append(F.interpolate(atten_large, size=(H,W), mode='bilinear', align_corners=True))
            # atten_input.append(atten)
            # atten_input_train = torch.cat(atten_input, axis=1)
            # atten_output_train = attention_model(atten_input_train,True)
            # softmax_layer = nn.Softmax()
            # attention_scales_weights = softmax_layer(atten_output_train)

            # _,_,H,W = output[3].shape
            # score_attention_output_small = torch.mul(F.interpolate(output_small[3], size=(H,W), mode='bilinear', align_corners=True), F.interpolate(attention_scales_weights[:,0].unsqueeze(0), size=(H,W), mode='bilinear', align_corners=True))
            # score_attention_output_large = torch.mul(F.interpolate(output_large[3], size=(H,W), mode='bilinear', align_corners=True), F.interpolate(attention_scales_weights[:,1].unsqueeze(0), size=(H,W), mode='bilinear', align_corners=True))
            # score_attention_output = torch.mul(output[3], F.interpolate(attention_scales_weights[:,2].unsqueeze(0), size=(H,W), mode='bilinear', align_corners=True))
            # attention_fusion_train_output = score_attention_output_small + score_attention_output_large + score_attention_output
            # multiscale_loss = criterion(attention_fusion_train_output, labels)

            # aug_output = model(aug_inputs)
        
            # atten_output = attention_model(feature1)

            # for child in model.module.conv0_0.children():
            #     print(child.weight)
            #     break

            # if epoch >= 80:
            #     for child in model.module.conv4_0.children():
            #         print(child.weight)
            #         break

            output, final_feature, feature, low_feature = model(inputs)
            # output = model(inputs)  # unet3+

            output_map = F.softmax(output[-1],dim=1).squeeze(0)
            output_map1 = torch.max(output_map,0)[0]
            print(np.unique(output_map1.flatten().detach().cpu().numpy()))

            # output_1 = F.softmax(output[-1],dim=1)
            # vis = output_1.detach().squeeze(0)
            # vis_value = torch.max(vis,0)[0]
            # threshold = torch.max(vis_value.flatten(),0)[0]
            # print("output threshold:", threshold)

            
            # ---------- self-attention for pseudo label version 3 ----------
            if epoch >= 50 and args.aux:  # 50

                # for l in range(64,128):
                #     fea = feature.detach().cpu().numpy()[0][l]
                #     Min = np.min(fea)
                #     Max = np.max(fea)
                #     fea = (fea-Min)/(Max-Min)
                #     cv2.imwrite("TiAl_feature_" + str(l) + ".png", fea*255)
                #     # exit(0)

                adjust_lr1(optimizer_attn, itr, max_itr, args, attention_model)
                _, height, width = labels.shape

                # if epoch >= 80:
                #     print(attention_model.module.embedding_conv_1.weight)

                if 50 <= epoch < 100:  # 50~100
                    time = 10  # 10
                else:
                    time = 10  # 4
                for k in range(time):
                    
                    # adjust_lr1(optimizer_attn, itr, max_itr, args, attention_model)
                    # if epoch >= 80:
                    #     print(attention_model.module.embedding_conv_1.weight)

                    for_crf_img = []
                    optimizer_attn.zero_grad()
                    class_index = torch.where(labels != 255, 1, 0).flatten()
                    class_index = torch.nonzero(class_index)
                    margin_length, _ = class_index.shape
                    # truth_map = torch.zeros((margin_length, margin_length)).long().cuda()
                    temp_class_index_take = torch.take(labels, class_index)
                    # for i in range(args.classes):
                    #     temp_truth_margin = torch.where(temp_class_index_take == i, 1, 0)
                    #     temp_truth_map = torch.mul(temp_truth_margin, temp_truth_margin.permute(1,0)).long()
                    #     truth_map = torch.where(temp_truth_map == 1, 1, truth_map)
                    _, channel, width, height  = feature.shape
                    feature_squeeze = feature.squeeze(0)
                    class_feature = torch.take(feature_squeeze[0,:,:], class_index).unsqueeze(0).squeeze(-1)
                    for j in range(1,channel):
                        temp_class_feature = torch.take(feature_squeeze[j,:,:], class_index).unsqueeze(0).squeeze(-1)
                        class_feature = torch.cat((class_feature, temp_class_feature), dim=0)
                    class_feature = class_feature.unsqueeze(0).unsqueeze(-1)
                    feature_squeeze = feature_squeeze.unsqueeze(0)
                    if saved_class_feature == None or saved_class_label == None:  # 第一个反正本身就是跟自己关联,也不用放在最后赋值了
                        saved_class_feature = class_feature.detach()
                        saved_class_label = temp_class_index_take.squeeze(1)
                    # attention_map, supervision_map, memory_map = attention_model(feature_squeeze, class_feature, temp_class_index_take.squeeze(1), saved_class_feature, saved_class_label)
                    attention_map, memory_map = attention_model(feature_squeeze, class_feature, temp_class_index_take.squeeze(1), saved_class_feature, saved_class_label)
                    # 处理细化旋转的扩充边界
                    # print(np.unique(memory_map.detach().cpu().numpy()))
                    # attention_loss = criterion(attention_map,labels)
                    # attention_loss = loss_fn(attention_map,labels)
                    attention_loss = criterion(attention_map,labels)
                    # supervision_loss = loss_sup(supervision_map, truth_map.float())
                    # memory_margin_length = len(saved_class_label)  # 20211130:这行和下面一行好像没用上?先注释了,貌似可以省空间
                    # memory_truth_map = torch.zeros((memory_margin_length, margin_length)).long().cuda()  # 20211130:这行好像没用上?先注释了,貌似可以省空间
                    memory_loss = None
                    for i in range(args.classes):
                        temp_memory_truth_margin = torch.where(saved_class_label == i, 1, 0).unsqueeze(1)
                        total_number = temp_memory_truth_margin.shape[0]
                        count_number = len(torch.nonzero(temp_memory_truth_margin))
                        if count_number == 0:
                            continue
                        temp_memory_truth_map = torch.mul(temp_memory_truth_margin, torch.where(temp_class_index_take == i, 1, 0).permute(1,0)).long()
                        # 上面的是用来算ground truth的
                        temp_index_map = torch.mul(temp_memory_truth_margin, torch.ones_like(temp_class_index_take.permute(1,0))).long()
                        zeros_map = torch.zeros_like(memory_map).float()
                        memory_map_1 = torch.where(temp_index_map == 1, memory_map, zeros_map)
                        # 这上面几个是为了去除其余类的输出数值
                        if memory_loss == None:
                            # memory_loss = loss_sup(memory_map_1, temp_memory_truth_map.float())
                            memory_loss = loss_sup(memory_map_1, temp_memory_truth_map.float()) / count_number * total_number
                            # print(loss_sup(memory_map_1, temp_memory_truth_map.float()) / count_number * total_number)
                        else:
                            # memory_loss += loss_sup(memory_map_1, temp_memory_truth_map.float())
                            memory_loss += loss_sup(memory_map_1, temp_memory_truth_map.float()) / count_number * total_number
                            # print(loss_sup(memory_map_1, temp_memory_truth_map.float()) / count_number * total_number)
                    memory_loss /= args.classes  # TODO:2021/12/9修改,之前这里写的是定值3,跑的陶瓷那个可能有影响,后面跑结果图的时候再补充下
                    if k == time - 1:  # 除了第一个之外,要把现在的和之前的关联有助于充分解决方向问题
                        saved_class_feature = class_feature.detach()
                        saved_class_label = temp_class_index_take.squeeze(1)
                    print("attention_loss", attention_loss)
                    # print("supervision_loss", supervision_loss)
                    print("memory_loss", memory_loss)
                    # supervision_loss = memory_loss + supervision_loss
                    supervision_loss = memory_loss
                    # attention_supervision_ratio = 0.5  # 此处比例需要实验确定,0.5
                    # 这里的attention_loss_sum应该是监督生成伪标注图,除了使用的特征影响外,这里的损失对伪标注图质量有直接影响
                    # attention_loss_sum = attention_loss * 1 + (supervision_loss * attention_supervision_ratio)
                    # attention_loss_sum = attention_loss * 0.5 + supervision_loss * 1
                    attention_loss_sum = attention_loss * 1 + supervision_loss * 1  # 感觉这个可能最好
                    # attention_loss_sum = attention_loss * 1 + supervision_loss * 0  # 跨batch监督项消融
                    # with torch.no_grad():
                    #     attention_map = F.softmax(attention_map,dim=1)
                    #     attention_map = attention_map**(1/0.5)
                    #     attention_map = attention_map / attention_map.sum(dim=1, keepdim=True)
                    #     attention_map = torch.log(attention_map)
                    
                    # print(attention_map.permute(0,2,3,1))

                    # ----- sharpen -----
                    attention_map = F.softmax(attention_map,dim=1)
                    # attention_map = attention_map**(1/args.T)
                    # attention_map = attention_map / attention_map.sum(dim=1, keepdim=True)
                    # ----- sharpen -----
                    
                    # print(attention_map.permute(0,2,3,1))

                    # # ----- sharpen -----
                    # attention_map = F.softmax(attention_map,dim=1)
                    # print(attention_map.permute(0,2,3,1))
                    # attention_map = attention_map**(1/args.T)
                    # attention_map = attention_map / attention_map.sum(dim=1, keepdim=True)
                    # print(attention_map.permute(0,2,3,1))
                    # # ----- sharpen -----
                    # attention_map = torch.log(attention_map)
                    # attention_loss = loss_fn(attention_map,labels)
                    # print(F.softmax(attention_map,dim=1).squeeze(0).permute(1,2,0))            
                    # ----- paint result -----
                    # vis = F.softmax(attention_map,dim=1).squeeze(0).cpu()
                    vis = attention_map.detach().squeeze(0)
                    vis_index = torch.max(vis,0)[1]
                    vis_value = torch.max(vis,0)[0]
                    # print(np.unique(vis_value.flatten().detach().cpu().numpy()))
                    threshold = torch.max(vis_value.flatten(),0)[0]
                    print("threshold:", threshold)
                    # threshold_index = int(len(vis_index.flatten()) * 0.8)  # 0.75,0.8
                    # threshold1 = torch.sort(vis_value.flatten(),descending=True)[0][threshold_index]

                    # 叠加掩膜遮盖,处理旋转带来的扩充边缘
                    vis_index = torch.where(mask==255, 255, vis_index)

                    for i in range(args.classes):
                        temp_zeros = torch.zeros_like(vis_value).float()
                        temp_vis_value = torch.where(vis_index == i, vis_value, temp_zeros)
                        if len(torch.nonzero(temp_vis_value)) != 0:
                            temp_ratio = torch.sum(temp_vis_value) / len(torch.nonzero(temp_vis_value))
                        else:
                            temp_ratio = 0
                        print("class ratio:", temp_ratio)
                        # temp_threshold_index = int(len(torch.nonzero(temp_vis_value)) * 0.6)  # 0.75,0.8
                        # temp_threshold = torch.sort(temp_vis_value.flatten(),descending=True)[0][temp_threshold_index]
                        temp_threshold = temp_ratio

                        # temp_threshold_index = int(len(torch.nonzero(temp_vis_value)) * 0.9)  # 0.75,0.8
                        # temp_threshold = torch.sort(temp_vis_value.flatten(),descending=True)[0][temp_threshold_index]

                        # temp_threshold = threshold1
                        temp_ones = torch.ones_like(vis_value).float()
                        temp_vis_value = torch.where(vis_index != i, temp_ones, temp_vis_value)
                        vis_index = torch.where(temp_vis_value < temp_threshold, 255, vis_index)

                    # threshold = max(0.9, 0.9+(threshold-0.9)*0.8)
                    # threshold = 0
                    # print("threshold:", threshold)
                    # vis_index = torch.where(vis_value>threshold, vis_index, 255)
                    
                    save_vis = vis_index.cpu().numpy()
                    
                    # ----- 统计伪标注数量 -----
                    unique, count = np.unique(save_vis,return_counts=True)
                    data_count = dict(zip(unique,count))
                    if 255 in unique:
                        sum_count = 0
                        for i in range(args.classes):
                            if i in unique:
                                sum_count += data_count[i]
                        ratio = sum_count/(height*width)
                    else:
                        ratio = 1
                    print("pseudo label number:", data_count)
                    print("ratio:", ratio)
                    # ----- 统计伪标注数量 -----
                    cv2.imwrite("attentionmap_" + str(step) + ".png", save_vis)
                    for j in range(args.classes):
                        save_vis_temp = np.where(save_vis==j, save_vis, 255)
                        for_crf_img.append(np.where(save_vis==j,1,0))
                        cv2.imwrite("attentionmap_" + str(step) + "_" + str(j) + ".png", save_vis_temp)
                    # ----- paint result -----
                    attention_loss_sum.backward()

                    # ----- 打印梯度 -----
                    # for name, parms in attention_model.module.named_parameters():
                    #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '-->grad_value:', parms.grad)
                    # ----- 打印梯度 -----

                    optimizer_attn.step()

                
                # ----- densecrf -----
                # for j in range(len(for_crf_img)):
                #     # n_labels = args.classes
                #     n_labels = 2

                #     input_img = inputs.squeeze(0).permute(1,2,0).detach().cpu().numpy()

                #     d = dcrf.DenseCRF(input_img.shape[1] * input_img.shape[0], n_labels)
        
                #     # input_label = torch.max(vis,0)[1].detach().cpu().numpy()
                #     input_label = for_crf_img[j]

                #     # 得到一元势（负对数概率）
                #     U = unary_from_labels(input_label, n_labels, gt_prob=0.5, zero_unsure=None)  
                #     #U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=HAS_UNK)## 如果有不确定区域，用这一行代码替换上一行
                #     d.setUnaryEnergy(U)
                
                #     # 这将创建与颜色无关的功能，然后将它们添加到CRF中
                #     feats = create_pairwise_gaussian(sdims=(3, 3), shape=input_img.shape[:2])
                #     d.addPairwiseEnergy(feats, compat=3,kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
                
                #     # 这将创建与颜色相关的功能，然后将它们添加到CRF中
                #     feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13), img=input_img, chdim=2)
                #     d.addPairwiseEnergy(feats, compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

                #     Q = d.inference(1)
        
                #     # 找出每个像素最可能的类
                #     CRF_output = np.argmax(Q, axis=0).reshape((input_img.shape[0], input_img.shape[1]))
                #     # print(CRF_output.shape)
                #     # print(np.unique(CRF_output))
                #     # exit(0)
                #     cv2.imwrite("CRF_result_" + str(j) + ".png", CRF_output * 100)
                # ----- densecrf -----
                

                # loss_attention = criterion(output[-1], vis_index.unsqueeze(0))
                loss_attention = loss_fn(output[-1], vis_index.unsqueeze(0), attention_map.detach(), device)
                print("loss_attention:", loss_attention)
                # if epoch > 90 and k == time - 1 and vis_index.shape[0] == 480 and vis_index.shape[1] == 640:
                #     save_pseudo = vis_index.detach().cpu().data.numpy()
                #     cv2.imwrite("pseudo_label.png", save_pseudo)
                #     exit(0)
                loss_attention_ratio = 3  # TODO:这边可能加个超参数效果更好点;1,2,3,4;1<2=4<3,不过实际差距不大
                loss_attention = loss_attention * loss_attention_ratio
            # ---------- self-attention for pseudo label version 3 ----------
            


            # if 70 < epoch <= 99:
            #     ft_height, ft_width = low_feature.shape[2], low_feature.shape[3]
            #     if ft_width / ft_height == 640/480 or ft_width / ft_height == 480/640:
            #         for l in range(64):
            #             fea = final_feature.detach().cpu().numpy()[0][l]
            #             Min = np.min(fea)
            #             Max = np.max(fea)
            #             fea = (fea-Min)/(Max-Min) * 255
            #             cv2.imwrite("TiAl_finalfeature_" + str(l) + ".png", fea)

            #         for l in range(64):
            #             fea = low_feature.detach().cpu().numpy()[0][l]
            #             Min = np.min(fea)
            #             Max = np.max(fea)
            #             fea = (fea-Min)/(Max-Min) * 255
            #             cv2.imwrite("TiAl_lowfeature_" + str(l) + ".png", fea)
            
            #         exit(0)

            '''
            # ---------- 分布监督 ----------
            output_label = output[-1].max(1)[1]
            output_label_stack = torch.stack([output_label]*64,dim=1)  # 1,64,640,480
            mask_label_stack = torch.stack([labels]*64,dim=1)
            # cls0_map = torch.where(output_label == 0, output_label, 255)
            # cls1_map = torch.where(output_label == 1, output_label, 255)
            # cls2_map = torch.where(output_label == 2, output_label, 255)
            # mask_cls0_map = torch.where(labels == 0, labels, 255)
            # mask_cls1_map = torch.where(labels == 1, labels, 255)
            # mask_cls2_map = torch.where(labels == 2, labels, 255)
            feature0_map = torch.where(output_label_stack == 0, feature, torch.zeros_like(feature))
            feature1_map = torch.where(output_label_stack == 1, feature, torch.zeros_like(feature))
            feature2_map = torch.where(output_label_stack == 2, feature, torch.zeros_like(feature))
            mask_feature0_map = torch.where(mask_label_stack == 0, feature, torch.zeros_like(feature))
            mask_feature1_map = torch.where(mask_label_stack == 1, feature, torch.zeros_like(feature))
            mask_feature2_map = torch.where(mask_label_stack == 2, feature, torch.zeros_like(feature))
            cls0_num = torch.count_nonzero(torch.where(output_label == 0, 1, 0))
            cls1_num = torch.count_nonzero(torch.where(output_label == 1, 1, 0))
            cls2_num = torch.count_nonzero(torch.where(output_label == 2, 1, 0))
            mask_cls0_num = torch.count_nonzero(torch.where(labels == 0, 1, 0))
            mask_cls1_num = torch.count_nonzero(torch.where(labels == 1, 1, 0))
            mask_cls2_num = torch.count_nonzero(torch.where(labels == 2, 1, 0))
            b,c,H,W = feature.size()
            feature0_view = feature0_map.view(b,c,-1)
            feature0_viewsum = torch.sum(feature0_view, -1)
            feature1_view = feature1_map.view(b,c,-1)
            feature1_viewsum = torch.sum(feature1_view, -1)
            feature2_view = feature2_map.view(b,c,-1)
            feature2_viewsum = torch.sum(feature2_view, -1)
            feature0_mean = feature0_viewsum/cls0_num
            feature1_mean = feature1_viewsum/cls1_num
            feature2_mean = feature2_viewsum/cls2_num
            # print(feature0_mean)
            # print(feature1_mean)
            # print(feature2_mean)
            mask_feature0_view = mask_feature0_map.view(b,c,-1)
            mask_feature0_viewsum = torch.sum(mask_feature0_view, -1)
            mask_feature1_view = mask_feature1_map.view(b,c,-1)
            mask_feature1_viewsum = torch.sum(mask_feature1_view, -1)
            mask_feature2_view = mask_feature2_map.view(b,c,-1)
            mask_feature2_viewsum = torch.sum(mask_feature2_view, -1)
            mask_feature0_mean = mask_feature0_viewsum/mask_cls0_num
            mask_feature1_mean = mask_feature1_viewsum/mask_cls1_num
            mask_feature2_mean = mask_feature2_viewsum/mask_cls2_num
            # print('-' * 10)
            # print(feature0_mean)
            # print(mask_feature0_mean)
            # print('-' * 10)
            # print(feature1_mean)
            # print(mask_feature1_mean)
            # print('-' * 10)
            # print(feature2_mean)
            # print(mask_feature2_mean)
            # print('-' * 10)
            # cls0_distance = torch.dist(feature0_mean, mask_feature0_mean, p=2)
            # cls1_distance = torch.dist(feature1_mean, mask_feature1_mean, p=2)
            # cls2_distance = torch.dist(feature2_mean, mask_feature2_mean, p=2)
            cls0_distance = F.kl_div(F.log_softmax(feature0_view,dim=-1), F.softmax(mask_feature0_view,dim=-1), reduction='mean')
            cls1_distance = F.kl_div(F.log_softmax(feature1_view,dim=-1), F.softmax(mask_feature1_view,dim=-1), reduction='mean')
            cls2_distance = F.kl_div(F.log_softmax(feature2_view,dim=-1), F.softmax(mask_feature2_view,dim=-1), reduction='mean')
            # print(cls0_num/(cls0_num + cls1_num + cls2_num))
            # print(cls1_num/(cls0_num + cls1_num + cls2_num))
            # print(cls2_num/(cls0_num + cls1_num + cls2_num))
            # print(cls0_distance * (cls0_num/(cls0_num + cls1_num + cls2_num)))
            # print(cls1_distance * (cls1_num/(cls0_num + cls1_num + cls2_num)))
            # print(cls2_distance * (cls2_num/(cls0_num + cls1_num + cls2_num)))
            # distribution_loss = (cls0_distance * (cls0_num/(cls0_num + cls1_num + cls2_num)) + cls1_distance * (cls0_num/(cls0_num + cls1_num + cls2_num))) / 2
            # distribution_loss = (cls0_distance * (cls0_num/(cls0_num + cls1_num + cls2_num)) + cls1_distance * (cls0_num/(cls0_num + cls1_num + cls2_num)) + cls2_distance * (cls0_num/(cls0_num + cls1_num + cls2_num))) / 3
            distribution_loss = (cls0_distance + cls1_distance + cls2_distance) / 3
            # ---------- 分布监督 ----------
            '''
            
            # ----- 这段是老的实现,当初的思想是取最显著的固定比例的像素点作为标注 -----
            # supratio = 0.9
            # output_label = F.softmax(output[-1]).max(1)[0].squeeze(0).cpu().detach().numpy()
            # h, w = output_label.shape
            # output_label = output_label.flatten()
            # output_label_index = np.argsort(output_label)
            # threshold = int(h * w * (1 - supratio))
            # output_label_prob = np.where(output_label_index <= threshold, output_label, 0)
            # output_label_prob = output_label_prob.reshape(h,w)
            # output_label_prob = torch.Tensor(output_label_prob).to(device).unsqueeze(0)
            # last_output_prob = torch.where(labels != 255, torch.zeros_like(labels).float(), output_label_prob)
            # output_cls = torch.where(last_output_prob != 0,output[-1].max(1)[1], 255)
            # ----- 这段是老的实现,当初的思想是取最显著的固定比例的像素点作为标注 -----

            vis = F.softmax(output[-1]).squeeze(0).cpu().detach().numpy()
            for i in range(args.classes):
                vis_img = vis[i] * 255
                cv2.imwrite(args.dataset + str(i) + ".png", vis_img)
            
            # if epoch < 80:  # 单独使用loss_attention时使用
            if True:
                loss1 = criterion(output[0], labels)
                loss2 = criterion(output[1], labels)
                loss3 = criterion(output[2], labels)
                loss4 = criterion(output[3], labels)
                # loss5 = criterion(output[4], labels)  # unet3+
                # ratio1, ratio2, ratio3, ratio4 = 0.7, 0.9, 1.1, 1.3
                # ratio1, ratio2, ratio3, ratio4 = 0.8, 0.9, 1, 1.1
                # ratio1, ratio2, ratio3, ratio4 = 1.3, 1.1, 0.9, 0.7
                ratio1, ratio2, ratio3, ratio4 = 1, 1, 1, 1
                # ratio1, ratio2, ratio3, ratio4, ratio5 = 1, 1, 1, 1, 1  # unet3+
                # ratio1, ratio2, ratio3 = 1, 1, 1
                # print("loss_aux:", loss_aux)
                # loss = (ratio2*loss2 + ratio3*loss3 + ratio4*loss4) / 3  # 最早在unet++的第一层套的vgg11删掉了一个池化层,所以当初在考虑要不要加第一层的监督
                loss = (ratio1*loss1 + ratio2*loss2 + ratio3*loss3 + ratio4*loss4) / 4  # unet++的第一层包含vgg11的池化层时,使用第一层进行deepsupervision比不用效果更好
                # loss = (ratio1*loss1 + ratio2*loss2 + ratio3*loss3 + ratio4*loss4 + ratio5*loss5) / 5  # unet3+
                print("loss1:%f,loss2:%f,loss3:%f,loss4:%f" % (loss1, loss2, loss3, loss4))
                print("loss:", loss)
                # if loss/last_loss > 50 and epoch >= 80:
                #     print("abnormal!!!")
                #     break_flag = True
                #     break
                # last_loss = loss
            
            
            if epoch >= 100 and args.aux:  # 100
                # TODO: multi-task learning
                # loss = loss_attention
                # loss = loss * 0 + loss_attention * 1
                # loss = loss * 0.5 + loss_attention * 1
                # loss += (loss_attention * (epoch - 30)/40)  
                # loss = loss + loss_attention  # baseline
                # loss = 0.5 * loss + 0.5 * loss_attention
                # loss = (7/10 * loss) + (loss_attention * 3/10)  # 7/10;3/10
                # loss = (loss**2/(loss+loss_attention)) + (loss_attention**2/(loss+loss_attention))
                # loss = (loss*loss_attention/(loss+loss_attention)) + (loss*loss_attention/(loss+loss_attention))
                # loss = 2 * (loss * loss_attention / (loss * 0.3 + loss_attention * 0.7))
                # loss = loss * loss_attention / (loss * 0.3 + loss_attention * 0.7)
                # loss = loss * loss_attention / (loss * 0.3 + loss_attention * 0.7) + loss
                # loss = loss * loss_attention / (loss + loss_attention) + loss
                # loss = (loss ** 1.4 * loss_attention ** 0.6 + 1e-10) ** 0.5
                #TODO:确定比例超参数,大了之后貌似会导致结果跳得比较厉害,小得话步数要的比较多
                # loss = loss_attention ** 2 / (loss + loss_attention) + loss  # loss + loss_attention - loss*loss_attention/(loss + loss_attention)
                loss = loss + loss_attention - loss*loss_attention/(loss + loss_attention)
                # loss = loss + loss_attention - loss*loss_attention/(loss * 2 + loss_attention)  # 好像是更优化loss_attention?
                # loss = loss + loss_attention - loss*loss_attention/(loss + loss_attention * 2)  # 按理来说应该是更优化loss,但出来的好像还是loss_attention更小

            
            # last_output_prob = torch.where(labels != 255, torch.zeros_like(labels).float(), output_label_prob)
            # last_output_cls = torch.where(last_output_prob != 0,output[-1].max(1)[1], 255)

            # loss = (ratio1*loss1 + ratio2*loss2 + ratio3*loss3 + ratio4*loss4) / 4 + multiscale_loss
            # mask = torch.where(labels == 255, torch.ones_like(labels), torch.zeros_like(labels))
            # output_1 = output.max(1)[1]
            # output_1 = output_1 * mask
            # aug_output_1 = aug_output.max(1)[1]
            # aug_output_1 = aug_output_1 * mask
            # aug_loss = nn.MSELoss()(output_1.float(), aug_output_1.float())
            # print(loss)
            # print(aug_loss)
            # total_loss = loss + (epoch+1) / 10 * aug_loss
            # print(total_loss)
            # print("total loss:", total_loss)
            # loss = criterion(output, aug_output, labels, epoch)  # my_loss

            loss.backward()
            optimizer.step()
            # print("now_lr:", optimizer.param_groups[0]['lr'])

            epoch_loss += loss.item()

            # ----- paint CAM -----
            # grads_val = grad_block[0].cpu().data.numpy().squeeze()
            # fmap = fmap_block[0].cpu().data.numpy().squeeze()
            # cam = gen_cam(fmap, grads_val)
            # for i in range(len(cam)):
            #     img_show = np.float32(cam[i]) * 255
            #     cv2.imwrite("CAM" + str(i + 1) + ".png", img_show)
            # img_show = np.float32(cam) * 255
            # cv2.imwrite("CAM_class0.png", img_show)
            # ----- paint CAM -----

            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // train_dataloader.batch_size + 1, loss.item()))
            logging.info("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // train_dataloader.batch_size + 1, loss.item()))

        lr_scheduler.step()
        loss_list.append(epoch_loss)

        if epoch % 10 == 0 or epoch >= 99:
            miou = val(model,val_dataloader,args,best_miou,epoch)
        else:
            miou = 0
        best_miou = max(best_miou, miou)
        miou_list.append(miou)
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
        logging.info("epoch %d loss:%0.3f" % (epoch, epoch_loss))
        if epoch == 99:
            save_99miou = miou
            save_99bestmiou = best_miou

        # if epoch == 99:
        #     torch.save(model.module, "./epoch99.pkl")
        #     exit(0)

    print('=' * 10)
    print("epoch 99 miou:%f" % (save_99miou))
    print("epoch 99 best miou:%f" % (save_99bestmiou))
    print("total best miou:%f" % (best_miou))

    draw_x = np.arange(0, len(loss_list))
    plt.plot(draw_x, loss_list)
    plt.xlabel('epoch')
    plt.ylabel('train loss')
    plt.savefig("../result/train_loss.png")
    plt.close()

    plt.plot(draw_x, miou_list)
    plt.xlabel('epoch')
    plt.ylabel('val miou')
    plt.savefig("../result/val_miou.png")
    plt.close()
    # -----train by epoch-----
    
    return model

def test(model, test_dataloaders, save_predict=False):
    logging.info('final test........')
    print('final test........')
    if save_predict ==True:
        dir = os.path.join(r'../saved_predict',str(args.arch),str(args.batch_size),str(args.epoch),str(args.dataset))
        if not os.path.exists(dir):
            os.makedirs(dir)
        else:
            print('dir already exist!')
    # model.load_state_dict(torch.load(r'../saved_model/'+str(args.arch)+'_'+str(args.batch_size)+'_'+str(args.dataset)+'_'+str(args.epoch)+'.pth', map_location='cpu'))  # 载入训练好的模型
    model.load_state_dict(torch.load(r'../saved_model/'+str(args.arch)+'_'+str(args.batch_size)+'_'+str(args.dataset)+'_'+str(args.epoch)+'_'+str(args.lr)+'.pth', map_location='cpu'))  # 载入训练好的模型
    model.eval()

    with torch.no_grad():
        i=0
        num = len(test_dataloaders)  #验证集图片的总数
        for pic, _, name, _ in test_dataloaders:
            pic = pic.to(device)
            if args.arch == 'fpn':
                _, predict = model(pic)
            else:
                output, atten, _, _ = model(pic)
                # output = model(pic)  # unet3+

                # _,_,H,W = output[3].shape
                # score_attention_output_small = torch.mul(F.interpolate(output_small[3], size=(H,W), mode='bilinear', align_corners=True), F.interpolate(attention_scales_weights[:,0].unsqueeze(0), size=(H,W), mode='bilinear', align_corners=True))
                # score_attention_output_large = torch.mul(F.interpolate(output_large[3], size=(H,W), mode='bilinear', align_corners=True), F.interpolate(attention_scales_weights[:,1].unsqueeze(0), size=(H,W), mode='bilinear', align_corners=True))
                # score_attention_output = torch.mul(output[3], F.interpolate(attention_scales_weights[:,2].unsqueeze(0), size=(H,W), mode='bilinear', align_corners=True))
                # attention_fusion_val_output = score_attention_output_small + score_attention_output_large + score_attention_output

                # predict = attention_fusion_val_output

                predict = output[-1]

            pre_label = predict.max(1)[1].squeeze().cpu().data.numpy()
            pre_label = np.asarray(pre_label, dtype=np.uint8)
            pre_label *= 60
            cv2.imwrite(args.dataset + "_" + str(i) + ".png", pre_label)
            i += 1
            '''
            pre = cm[pre_label]
            pre = Image.fromarray(pre.astype("uint8"), mode='RGB')
            if i < 20 and save_predict:
                # print(name[0][-15:-4])
                # pre.save("./saved_test_sample/" + name[0][-15:-4] +  "_result.png")
                pass
            i += 1
            '''

def adjust_lr(optimizer, itr, max_itr, args, model):
    
    # ----- 这段最初好像是poly的策略? -----
	# now_lr = args.lr * (1 - itr/(max_itr+1)) ** 0.9
	# optimizer.param_groups[0]['lr'] = now_lr
	# optimizer.param_groups[1]['lr'] = 10*now_lr
	# return now_lr
    # ----- 这段最初好像是poly的策略? -----

    # ----- 180epoch,学习率设置 -----
    # itr /= 2  # 两张图用  TODO:最早itr写在每个batch里面就要除
    itr -= 1  # TODO:新修改的itr位置,减一就行
    # if itr <= 30:
    #     now_lr = 0.001
    # elif 30 < itr <= 80:
    #     now_lr = 0.003
    # elif 80 < itr <= 130:
    #     now_lr = 0.003 * 0.9
    # else:
    #     now_lr = 0.003 * 0.9 * 0.9

    # if itr <= 30:
    #     now_lr = 0.001
    # elif 30 < itr <= 80:
    #     now_lr = 0.003
    # elif 80 < itr <= 120:
    #     now_lr = 0.003 * 0.9
    # else:
    #     now_lr = 0.003 * 0.9 * 0.9
    if itr < 50:  # 50
        # now_lr = 0.007
        now_lr = 0.003  # 两张图用
        # now_lr = 0.005  # 一张图用
        # now_lr = 0.003  # 一张图全监督
        # now_lr = 0.001  # 两张图全监督
        # now_lr = 0.07  # 验证伪标注图
        refine_lr = now_lr
        # backbone_lr = 0.0003
    elif 50 <= itr < 100:  # 50~100
        # now_lr = 0.003  # 一张图用(差不大多,可能还没用下面的那个好)
        now_lr = 0.001  # 一张图用,两张图用
        # now_lr = 0.001  # 一张图全监督
        # now_lr = 0.0007  # 两张图全监督
        # now_lr = 0.03  # 验证伪标注图
        refine_lr = now_lr
        # backbone_lr = 0.0003
    elif 100 <= itr < 150:  # 80~120,60~120
        # now_lr = 0.003 * 0.9
        # now_lr = 0.03
        # now_lr = 0.001 * 0.5
        now_lr = 0
        # ----- backbone最后一层冻掉 -----
        for child in model.module.conv4_0.children():
            for par in child.parameters():
                par.requires_grad = False
        # ----- backbone最后一层冻掉 -----
        # for child in model.module.conv1_1.children():
        #     for par in child.parameters():
        #         par.requires_grad = False
        # for child in model.module.conv1_2.children():
        #     for par in child.parameters():
        #         par.requires_grad = False
        # for child in model.module.conv1_3.children():
        #     for par in child.parameters():
        #         par.requires_grad = False
        # for child in model.module.conv2_1.children():
        #     for par in child.parameters():
        #         par.requires_grad = False
        # for child in model.module.conv2_2.children():
        #     for par in child.parameters():
        #         par.requires_grad = False
        # for child in model.module.conv3_1.children():
        #     for par in child.parameters():
        #         par.requires_grad = False
        # ----- 输出层前几个深监督 -----
        # for child in model.module.conv0_1.children():
        #     for par in child.parameters():
        #         par.requires_grad = False
        # for child in model.module.conv0_2.children():
        #     for par in child.parameters():
        #         par.requires_grad = False
        # for child in model.module.conv0_3.children():
        #     for par in child.parameters():
        #         par.requires_grad = False
        # ----- 输出层前几个深监督 -----
        refine_lr = 0.0001  # 两张图用
        # refine_lr = 0.0003  # 一张图用
        # now_lr = 0.007  # 验证伪标注图
        # now_lr = 0.0007
        # backbone_lr = 0.0003 * 0.2
    else:
        # refine_lr = 0.0001  # 2021/12/09修改,原本使用
        refine_lr = 0.0003
        now_lr = 0
        # backbone_lr = 0
    optimizer.param_groups[0]['lr'] = now_lr  # 这样应该是只修改了base_params
    # optimizer.param_groups[1]['lr'] = backbone_lr
    # optimizer.param_groups[2]['lr'] = backbone_lr
    # optimizer.param_groups[3]['lr'] = backbone_lr
    # optimizer.param_groups[4]['lr'] = backbone_lr
    # optimizer.param_groups[5]['lr'] = backbone_lr
    # optimizer.param_groups[1]['lr'] = now_lr  # no backbone
    # optimizer.param_groups[2]['lr'] = now_lr  # no backbone
    # optimizer.param_groups[3]['lr'] = now_lr  # no backbone
    # optimizer.param_groups[4]['lr'] = now_lr  # no backbone
    optimizer.param_groups[5]['lr'] = now_lr
    # if itr < 80:
    #     optimizer.param_groups[1]['lr'] = now_lr
    #     optimizer.param_groups[2]['lr'] = now_lr
    #     optimizer.param_groups[3]['lr'] = now_lr
    #     optimizer.param_groups[4]['lr'] = now_lr
    #     optimizer.param_groups[5]['lr'] = now_lr
    optimizer.param_groups[6]['lr'] = refine_lr
    optimizer.param_groups[7]['lr'] = refine_lr
    optimizer.param_groups[8]['lr'] = refine_lr
    optimizer.param_groups[9]['lr'] = refine_lr
    optimizer.param_groups[10]['lr'] = refine_lr
    optimizer.param_groups[11]['lr'] = refine_lr
    optimizer.param_groups[12]['lr'] = refine_lr
    optimizer.param_groups[13]['lr'] = refine_lr
    optimizer.param_groups[14]['lr'] = refine_lr
    optimizer.param_groups[15]['lr'] = refine_lr
    optimizer.param_groups[16]['lr'] = refine_lr
    optimizer.param_groups[17]['lr'] = refine_lr
    # if itr >= 80:  # 60
    #     ratio = 1  # 0.5,1
    #     # optimizer.param_groups[1]['lr'] = now_lr * ratio
    #     # optimizer.param_groups[2]['lr'] = now_lr * ratio
    #     # optimizer.param_groups[3]['lr'] = now_lr * ratio
    #     # optimizer.param_groups[4]['lr'] = now_lr * ratio
    #     # optimizer.param_groups[5]['lr'] = now_lr * ratio
    #     optimizer.param_groups[6]['lr'] = now_lr * ratio
    #     optimizer.param_groups[7]['lr'] = now_lr * ratio
    #     optimizer.param_groups[8]['lr'] = now_lr * ratio
    #     optimizer.param_groups[9]['lr'] = now_lr * ratio
    #     optimizer.param_groups[10]['lr'] = now_lr * ratio
    #     optimizer.param_groups[11]['lr'] = now_lr * ratio
    #     optimizer.param_groups[12]['lr'] = now_lr * ratio
    #     optimizer.param_groups[13]['lr'] = now_lr * ratio
    #     optimizer.param_groups[14]['lr'] = now_lr * ratio
    #     optimizer.param_groups[15]['lr'] = now_lr * ratio
    #     # optimizer.param_groups[16]['lr'] = now_lr * ratio
    #     # optimizer.param_groups[17]['lr'] = now_lr * ratio
    #     # optimizer.param_groups[18]['lr'] = now_lr * ratio
    #     # optimizer.param_groups[19]['lr'] = now_lr * ratio
    #     # optimizer.param_groups[6]['lr'] = 0.0003
    #     # optimizer.param_groups[7]['lr'] = 0.0003
    #     # optimizer.param_groups[8]['lr'] = 0.0003
    #     # optimizer.param_groups[9]['lr'] = 0.0003
    #     # optimizer.param_groups[10]['lr'] = 0.0003
    #     # optimizer.param_groups[11]['lr'] = 0.0003
    #     # optimizer.param_groups[12]['lr'] = 0.0003
    #     # optimizer.param_groups[13]['lr'] = 0.0003
    #     # optimizer.param_groups[14]['lr'] = 0.0003
    #     # optimizer.param_groups[15]['lr'] = 0.0003
    # ----- 180epoch,学习率设置 -----

    return now_lr
    

def adjust_lr1(optimizer, itr, max_itr, args, model):
    
    itr -= 1
    # itr /= 2  # 两张图用  TODO:最早itr写在每个batch里面就要除
    if 50 <= itr < 100:  # 40~60
        # now_lr = 0.003
        # now_lr = 0.03 * (150 - itr) / 100
        now_lr = 0.03
        # now_lr = 0.03  # 0.007
    elif 100 <= itr < 200:  # 60~120
        # now_lr = 0.03 * 0.9
        # now_lr = 0.01 * 0.9
        # now_lr = 0.007
        # now_lr = 0.03
        now_lr = 0.01
        # now_lr = 0.001  # *
        # now_lr = 0
        # now_lr = 0.003
        # for par in model.module.embedding_conv_1.parameters():
        #     par.requires_grad = False
        # for par in model.module.embedding_conv_2.parameters():
        #     par.requires_grad = False
        # for par in model.module.combine_channel1.parameters():
        #     par.requires_grad = False
        # for par in model.module.combine_channel2.parameters():
        #     par.requires_grad = False
        # now_lr = 0.01
    else:
        # now_lr = 0.007  # 2021/12/09修改,原本使用
        now_lr = 0.01
    optimizer.param_groups[0]['lr'] = now_lr
    return now_lr
    

def get_params(model, key):
	for m in model.named_modules():
		if key == '1x':
			if 'backbone' in m[0] and isinstance(m[1], nn.Conv2d):
				for p in m[1].parameters():
					yield p
		elif key == '10x':
			if 'backbone' not in m[0] and isinstance(m[1], nn.Conv2d):
				for p in m[1].parameters():
					yield p

def gen_cam(feature_map, grads):
    """
    依据梯度和特征图，生成cam
    :param feature_map: np.array， in [C, H, W]
    :param grads: np.array， in [C, H, W]
    :return: np.array, [H, W]
    """
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # cam shape (H, W)

    weights = np.mean(grads, axis=(1, 2))  #

    # cam_list = []

    for i, w in enumerate(weights):
        # print(w)
        # print(feature_map[i,:,:])
        # cam = w * feature_map[i, :, :]
        cam += w * feature_map[i, :, :]

        # cam = np.maximum(cam, 0)
        # # cam = cv2.resize(cam, (64, 64))
        # cam -= np.min(cam)
        # cam /= np.max(cam)
        # cam_list.append(cam)
    cam = np.maximum(cam, 0)
    cam -= np.min(cam)
    cam /= np.max(cam)
    # return cam_list
    return cam

if __name__ == "__main__":
    args = getArgs()
    logging = getLog(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ------a strategy for calculating learning rate------
    # lrs = {
    #     'pascalvoc2012': 0.007,  # lr:0.007;epoch:50;batch-size:16;momentum:0.9;weight-decay=5e-4;nesterov:False
    #     'pascalvoc2012_tiny': 0.007,
    # }
    # learning_rate = lrs[args.dataset] / (4 * len(args.gpu_ids)) * args.batch_size
    # ------a strategy for calculating learning rate------
    print('**************************')
    print('models:%s,\nepoch:%s,\nbatch size:%s,\ndataset:%s,\nlearning rate:%.5f' % \
          (args.arch, args.epoch, args.batch_size,args.dataset, args.lr))
    logging.info('\n=======\nmodels:%s,\nepoch:%s,\nbatch size:%s,\ndataset:%s,\nlearning rate:%s\n========' % \
          (args.arch, args.epoch, args.batch_size,args.dataset,args.lr))
    print('**************************')
    model = getModel(args)
    attention_model = getSelfAttnModel(args)
    devices_list = args.gpu_ids.split(',')
    devices_list = [num for num in range(len(devices_list))]
    model = torch.nn.DataParallel(model,device_ids=devices_list)
    attention_model = torch.nn.DataParallel(attention_model,device_ids=devices_list)
    # attention_model = FAM_Module(512,3)
    # attention_model = torch.nn.DataParallel(attention_model,device_ids=devices_list)
    # attention_model = SpatialAttention()
    # attention_model = torch.nn.DataParallel(attention_model,device_ids=devices_list)
    # patch_replication_callback(model)  # 不确定是不是需要这个
    train_dataloaders,val_dataloaders,test_dataloaders = getDataset(args)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    # ----- attention loss -----
    # loss_sup = torch.nn.SmoothL1Loss()
    # loss_sup = torch.nn.L1Loss()
    loss_sup = torch.nn.MSELoss()
    # loss_fn = torch.nn.L1Loss()
    # loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)
    # loss_fn = torch.nn.NLLLoss(ignore_index=255)
    # loss_fn = FocalLoss()
    # loss_fn = FocalLoss(args.classes)
    loss_fn = WeightedCrossEntropyLoss(ignore_index=255)
    # ----- attention loss -----
    # optimizer = optim.SGD(model.module.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # ---- deeplabv3+ xception -----
    # optimizer = optim.SGD(
	# 	params = [
	# 		{'params': get_params(model.module,key='1x'), 'lr': 0.007},
	# 		{'params': get_params(model.module,key='10x'), 'lr': 10*0.007}
    #         # {'params': get_params(parameter_source,key='backbone'), 'lr': cfg.TRAIN_LR},
	# 		# {'params': get_params(parameter_source,key='cls'),      'lr': 10*cfg.TRAIN_LR},
	# 		# {'params': get_params(parameter_source,key='others'),   'lr': cfg.TRAIN_LR}
	# 	],
	# 	momentum=0.9,
    #     weight_decay=4e-5
	# )
    # itr = 0 * len(train_dataloaders)
	# max_itr = 46 * len(train_dataloaders)
    # ---- deeplabv3+ xception -----
    # optimizer = optim.Adam(model.module.parameters(),lr=args.lr)  # 整个模型使用统一学习率
    # ----- 模型backbone使用不同学习率 -----
    
    conv0_0_params = list(map(id, model.module.conv0_0.parameters()))
    conv1_0_params = list(map(id, model.module.conv1_0.parameters()))
    conv2_0_params = list(map(id, model.module.conv2_0.parameters()))
    conv3_0_params = list(map(id, model.module.conv3_0.parameters()))
    conv4_0_params = list(map(id, model.module.conv4_0.parameters()))
    conv0_1_params = list(map(id, model.module.conv0_1.parameters()))
    conv1_1_params = list(map(id, model.module.conv1_1.parameters()))
    conv2_1_params = list(map(id, model.module.conv2_1.parameters()))
    conv3_1_params = list(map(id, model.module.conv3_1.parameters()))
    conv0_2_params = list(map(id, model.module.conv0_2.parameters()))
    conv1_2_params = list(map(id, model.module.conv1_2.parameters()))
    conv2_2_params = list(map(id, model.module.conv2_2.parameters()))
    conv0_3_params = list(map(id, model.module.conv0_3.parameters()))
    conv1_3_params = list(map(id, model.module.conv1_3.parameters()))
    conv0_4_params = list(map(id, model.module.conv0_4.parameters()))
    refine1_params = list(map(id, model.module.refine1.parameters()))
    refine2_params = list(map(id, model.module.refine2.parameters()))
    refine3_params = list(map(id, model.module.refine3.parameters()))
    refine4_params = list(map(id, model.module.refine4.parameters()))
    final1_params = list(map(id, model.module.final1.parameters()))
    final2_params = list(map(id, model.module.final2.parameters()))
    final3_params = list(map(id, model.module.final3.parameters()))
    final4_params = list(map(id, model.module.final4.parameters()))
    # base_params = filter(lambda p: id(p) not in conv0_0_params + conv1_0_params + conv2_0_params + conv3_0_params + conv4_0_params + conv0_1_params + conv1_1_params + conv2_1_params + conv3_1_params + conv0_2_params + conv1_2_params + conv2_2_params + conv0_3_params + conv1_3_params + conv0_4_params + refine1_params + refine2_params + refine3_params + refine4_params, model.module.parameters())
    # base_params = filter(lambda p: id(p) not in conv0_0_params + conv1_0_params + conv2_0_params + conv3_0_params + conv4_0_params + conv0_1_params + conv1_1_params + conv2_1_params + conv3_1_params + conv0_2_params + conv1_2_params + conv2_2_params + conv0_3_params + conv1_3_params + conv0_4_params, model.module.parameters())
    base_params = filter(lambda p: id(p) not in conv0_0_params + conv1_0_params + conv2_0_params + conv3_0_params + conv4_0_params + conv0_1_params + conv0_2_params + conv0_3_params + conv0_4_params + refine1_params + refine2_params + refine3_params + refine4_params + final1_params + final2_params + final3_params + final4_params, model.module.parameters())
    params = [
        {'params': base_params},
        {'params': model.module.conv0_0.parameters(), 'lr': 0},  # 0.0003
        {'params': model.module.conv1_0.parameters(), 'lr': 0},  # 0.0003
        {'params': model.module.conv2_0.parameters(), 'lr': 0},  # 0.0003
        {'params': model.module.conv3_0.parameters(), 'lr': 0},  # 0.0003
        {'params': model.module.conv4_0.parameters(), 'lr': 0.0003},  # 0.0003
        {'params': model.module.conv0_1.parameters()},
        # {'params': model.module.conv1_1.parameters()},
        # {'params': model.module.conv2_1.parameters()},
        # {'params': model.module.conv3_1.parameters()},
        {'params': model.module.conv0_2.parameters()},
        # {'params': model.module.conv1_2.parameters()},
        # {'params': model.module.conv2_2.parameters()},
        {'params': model.module.conv0_3.parameters()},
        # {'params': model.module.conv1_3.parameters()},
        {'params': model.module.conv0_4.parameters()},
        {'params': model.module.refine1.parameters()},
        {'params': model.module.refine2.parameters()},
        {'params': model.module.refine3.parameters()},
        {'params': model.module.refine4.parameters()},
        {'params': model.module.final1.parameters()},
        {'params': model.module.final2.parameters()},
        {'params': model.module.final3.parameters()},
        {'params': model.module.final4.parameters()}
    ]
    for child in model.module.conv0_0.children():
        for par in child.parameters():
            par.requires_grad = False
    for child in model.module.conv1_0.children():
        for par in child.parameters():
            par.requires_grad = False
    for child in model.module.conv2_0.children():
        for par in child.parameters():
            par.requires_grad = False
    for child in model.module.conv3_0.children():
        for par in child.parameters():
            par.requires_grad = False
    optimizer = optim.Adam(params,lr=args.lr)
    
    # optimizer = optim.Adam(model.module.parameters(),lr=0.005)  # unet3+full:0.002;unet3+scribble:0.005;unet++full:0.005;unet++scribble:0.005
    # ----- 模型backbone使用不同学习率 -----
    optimizer_attn = optim.Adam(attention_model.module.parameters(),lr=args.attention_lr)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=args.decay_lr_step,gamma=args.decay_lr_ratio)  # 这里的衰减那些设置实际上已经没用了,unet++的lr修改都在adjust_lr函数里面实现了
    lr_scheduler_attn = optim.lr_scheduler.StepLR(optimizer_attn,step_size=args.attention_decay_lr_step,gamma=args.attention_decay_lr_ratio)
    # lr_scheduler = LR_Scheduler('poly', args.lr, 50, len(train_dataloaders))
    # lr_scheduler = None
    # load color map
    colormap = []
    cmap_file = "../data/pascalvoc2012augmented/cmap.mat"
    cmap = mat4py.loadmat(cmap_file)
    colors = cmap["cmap"]
    for color in colors:
        colormap.append([int(255*color[0]), int(255*color[1]), int(255*color[2])])
    cm = np.array(colormap).astype('uint8')
    # if args.dataset == "pascal_tiny" or args.dataset == "pascal":
    #     cudnn.benchmark = True  # 固定尺寸输入使用
    if 'train' in args.action:
        train(model, attention_model, criterion, loss_fn, loss_sup, optimizer, optimizer_attn, train_dataloaders,val_dataloaders, lr_scheduler, lr_scheduler_attn, devices_list, args)
    if 'test' in args.action:
        test(model, test_dataloaders, save_predict=True)
