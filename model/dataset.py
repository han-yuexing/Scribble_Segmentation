import torch.utils.data as data
import PIL.Image as Image
from sklearn.model_selection import train_test_split
import os
import random
import numpy as np
from skimage.io import imread
import cv2
from glob import glob
import imageio
from torchvision.transforms import transforms
import torch
from utils import *
import warnings
from skimage.segmentation import slic,mark_boundaries
import copy

warnings.filterwarnings('ignore')



class MSRCDataset(data.Dataset):
    def __init__(self, state, classes, aug=True):
        self.state = state
        self.train_root = r"../data/msrc/cow_general_train"  # 牛
        # self.train_root = r"../data/msrc/cow_general_train_full"  # 牛
        # self.train_root = r"../data/msrc/sheep_general_train"  # 羊
        # self.train_root = r"../data/msrc/sheep_general_train_full"  # 羊
        # self.train_root = r"../data/msrc/mix_train"  # 羊&牛
        # self.val_root = r"../data/msrc/cow_general_test"  # 牛
        self.val_root = r"../data/msrc/cow_general_test_more"  # 牛
        # self.val_root = r"../data/msrc/sheep_general_test"  # 羊
        # self.val_root = r"../data/msrc/mix_test"  # 羊&牛
        self.test_root = self.val_root
        self.pics,self.masks = self.getDataPath()
        self.classes = classes

    def getDataPath(self):
        assert self.state =='train' or self.state == 'val' or self.state =='test'
        if self.state == 'train':
            root = self.train_root
        if self.state == 'val':
            root = self.val_root
        if self.state == 'test':
            root = self.test_root

        pics = []
        masks = []
        data_list = os.listdir(root)
        masks_list = [name for name in data_list if "label" in name]
        pics_list = [name for name in data_list if name not in masks_list]
        assert len(masks_list) == len(pics_list)
        masks_list.sort()
        pics_list.sort()
        for i in range(len(masks_list)):
            img = os.path.join(root, pics_list[i])
            mask = os.path.join(root, masks_list[i])
            pics.append(img)
            masks.append(mask)
        return pics,masks

    def __getitem__(self, index):
        x_path = self.pics[index]
        y_path = self.masks[index]
        origin_x = cv2.imread(x_path)
        origin_y = cv2.imread(y_path, cv2.COLOR_BGR2GRAY)

        if self.state == 'train':
            origin_x = origin_x / 255
            # flip
            # origin_x, _, origin_y = random_mirror(origin_x, origin_x, origin_y)
            # # resize
            # origin_x, _, origin_y = random_scale(origin_x, origin_x, origin_y)
            # crop
            # origin_x, _, origin_y = random_crop(origin_x, origin_x, origin_y)
            # rotate
            # origin_x, _, origin_y = random_rotate(origin_x, origin_x, origin_y)
            # origin_x, _, origin_y = random_crop(origin_x, origin_x, origin_y)
            # 随机颜色抖动(针对voc这类场景图像)
            # origin_x = random_color_jitter(origin_x)
            # 随机旋转,细化旋转角度
            origin_x, _, origin_y, mask = random_rotate_custom(origin_x, origin_x, origin_y, 18, True)
            origin_x, mask, origin_y = random_mirror(origin_x, mask, origin_y)
            # flip_x, flip_y = flip(origin_x, origin_y)
        
        if self.state == 'train':
            origin_x1 = origin_x * 255
            H,W,_ = origin_x1.shape
            segment_num = H*W//1000  # 1000;500
            # ----- randaugment -----
            randaugmentops = rand_augment_ops()
            randaugment = RandAugment(randaugmentops)
            origin_x_1 = Image.fromarray(cv2.cvtColor(origin_x1.astype(np.uint8),cv2.COLOR_BGR2RGB))
            mixed_x = randaugment(origin_x_1)
            mixed_x = np.array(mixed_x)
            mixed_x = mixed_x[:,:,(2,1,0)]
            mixed_x = mixed_x.astype(np.float32) / 255
            # ----- randaugment -----
            # ----- 超像素获取目标区域 -----
            # expand_origin_y = origin_y
            # segments = slic(origin_x1[:,:,::-1].astype(np.uint8), n_segments=segment_num, compactness=50)  # io.imread的输入通道跟cv2相反
            # for i in range(segment_num):
            #     for j in range(3):
            #         area = np.where(segments==i,j,-1)
            #         overlap = np.where(area==origin_y,1,0)
            #         count = np.bincount(overlap.flatten())
            #         if len(count) > 1:
            #             expand_origin_y = np.where(segments==i,j,expand_origin_y)
            # cv2.imwrite("expand_y.png", expand_origin_y)
            # exit(0)
            # ----- 超像素获取目标区域 -----
        
        # origin_x_small, origin_y_small = fixed_scale(origin_x, origin_y, 0.5)
        # origin_x_large, origin_y_large = fixed_scale(origin_x, origin_y, 0.75)

        trans = transforms.ToTensor()
        img_x = trans(origin_x).float()
        # img_x_small = trans(origin_x_small).float()
        # img_x_large = trans(origin_x_large).float()
        if self.state == 'train':
            img_x1 = trans(mixed_x).float()
            # img_x_flip = trans(flip_x).float()
            # img_y_flip = torch.Tensor(flip_y)
            # expand_y = torch.Tensor(expand_origin_y)
            mask = torch.Tensor(mask)
        img_y = torch.Tensor(origin_y)
        # img_y_small = torch.Tensor(origin_y_small)
        # img_y_large = torch.Tensor(origin_y_large)

        if self.state == 'train':
            return img_x,img_x1,img_y,mask,x_path,y_path
            # return img_x,img_x1,expand_y,x_path,y_path

            # return torch.stack([img_x, img_x_flip],dim=0),img_x1,torch.stack([img_y,img_y_flip],dim=0),x_path,y_path
            # return img_x,img_x1,img_x_small,img_x_large,img_y,img_y_small,img_y_large,x_path,y_path
        else:
            return img_x,img_y,x_path,y_path
            # return img_x,img_x_small,img_x_large,img_y,img_y_small,img_y_large,x_path,y_path

    def __len__(self):
        return len(self.pics)

class ChooseVocDataset(data.Dataset):
    def __init__(self, state, classes, aug=True):
        self.state = state
        # self.train_root = r"../data/choosevoc/train1"  # 羊
        # self.train_root = r"../data/choosevoc/train1_full"  # 羊(全监督)
        self.train_root = r"../data/choosevoc/train2"  # 飞机
        # self.train_root = r"../data/choosevoc/train2_full"  # 飞机(全监督)
        # self.train_root = r"../data/choosevoc/train3"  # 飞机(非天空)
        # self.train_root = r"../data/choosevoc/train4"  # 飞机(地面)
        # self.train_root = r"../data/choosevoc/train4_full"  # 飞机(地面全监督)
        # self.train_root = r"../data/choosevoc/train4_5"  # 飞机(地面)
        # self.train_root = r"../data/choosevoc/train5"  # 羊(大部分草地)
        # self.train_root = r"../data/choosevoc/cat_train"  # 猫
        # self.train_root = r"../data/choosevoc/cow_train"  # 牛
        # self.val_root = r"../data/choosevoc/test1"  # 羊(37张)
        # self.val_root = r"../data/choosevoc/test2"  # 飞机(22张)
        self.val_root = r"../data/choosevoc/test2_10"  # 飞机(14张)
        # self.val_root = r"../data/choosevoc/test3"  # 空文件夹
        # self.val_root = r"../data/choosevoc/test4"  # 飞机(地面)
        # self.val_root = r"../data/choosevoc/test5"  # 羊(大部分草地)-19
        # self.val_root = r"../data/choosevoc/test5_easy"  # 羊(单独)
        # self.val_root = r"../data/choosevoc/cat_test"  # 猫
        # self.val_root = r"../data/choosevoc/cow_test"  # 牛
        self.test_root = self.val_root
        self.pics,self.masks = self.getDataPath()
        self.classes = classes

    def getDataPath(self):
        assert self.state =='train' or self.state == 'val' or self.state =='test'
        if self.state == 'train':
            root = self.train_root
        if self.state == 'val':
            root = self.val_root
        if self.state == 'test':
            root = self.test_root

        pics = []
        masks = []
        data_list = os.listdir(root)
        masks_list = [name for name in data_list if "label" in name]
        pics_list = [name for name in data_list if name not in masks_list]
        assert len(masks_list) == len(pics_list)
        masks_list.sort()
        pics_list.sort()
        for i in range(len(masks_list)):
            img = os.path.join(root, pics_list[i])
            mask = os.path.join(root, masks_list[i])
            pics.append(img)
            masks.append(mask)
        return pics,masks

    def __getitem__(self, index):
        x_path = self.pics[index]
        y_path = self.masks[index]
        origin_x = cv2.imread(x_path)
        origin_y = cv2.imread(y_path, cv2.COLOR_BGR2GRAY)

        if self.state == 'train':
            origin_x = origin_x / 255
            # flip
            # origin_x, _, origin_y = random_mirror(origin_x, origin_x, origin_y)
            # # resize
            # origin_x, _, origin_y = random_scale(origin_x, origin_x, origin_y)
            # crop
            # origin_x, _, origin_y = random_crop(origin_x, origin_x, origin_y)
            # rotate
            # origin_x, _, origin_y = random_rotate(origin_x, origin_x, origin_y)
            # origin_x, _, origin_y = random_crop(origin_x, origin_x, origin_y)
            # 随机颜色抖动(针对voc这类场景图像)
            # origin_x = random_color_jitter(origin_x)
            # 随机旋转,细化旋转角度
            origin_x, _, origin_y, mask = random_rotate_custom(origin_x, origin_x, origin_y, 18, True)
            # flip_x, flip_y = flip(origin_x, origin_y)
            origin_x, mask, origin_y = random_mirror(origin_x, mask, origin_y)
        
        if self.state == 'train':
            origin_x1 = origin_x * 255
            H,W,_ = origin_x1.shape
            segment_num = H*W//1000  # 1000;500
            # ----- randaugment -----
            randaugmentops = rand_augment_ops()
            randaugment = RandAugment(randaugmentops)
            origin_x_1 = Image.fromarray(cv2.cvtColor(origin_x1.astype(np.uint8),cv2.COLOR_BGR2RGB))
            mixed_x = randaugment(origin_x_1)
            mixed_x = np.array(mixed_x)
            mixed_x = mixed_x[:,:,(2,1,0)]
            mixed_x = mixed_x.astype(np.float32) / 255
            # ----- randaugment -----
            # ----- 超像素获取目标区域 -----
            # expand_origin_y = origin_y
            # segments = slic(origin_x1[:,:,::-1].astype(np.uint8), n_segments=segment_num, compactness=50)  # io.imread的输入通道跟cv2相反
            # for i in range(segment_num):
            #     for j in range(3):
            #         area = np.where(segments==i,j,-1)
            #         overlap = np.where(area==origin_y,1,0)
            #         count = np.bincount(overlap.flatten())
            #         if len(count) > 1:
            #             expand_origin_y = np.where(segments==i,j,expand_origin_y)
            # cv2.imwrite("expand_y.png", expand_origin_y)
            # exit(0)
            # ----- 超像素获取目标区域 -----
        
        # origin_x_small, origin_y_small = fixed_scale(origin_x, origin_y, 0.5)
        # origin_x_large, origin_y_large = fixed_scale(origin_x, origin_y, 0.75)

        trans = transforms.ToTensor()
        img_x = trans(origin_x).float()
        # img_x_small = trans(origin_x_small).float()
        # img_x_large = trans(origin_x_large).float()
        if self.state == 'train':
            img_x1 = trans(mixed_x).float()
            # img_x_flip = trans(flip_x).float()
            # img_y_flip = torch.Tensor(flip_y)
            # expand_y = torch.Tensor(expand_origin_y)
            mask = torch.Tensor(mask)
        img_y = torch.Tensor(origin_y)
        # img_y_small = torch.Tensor(origin_y_small)
        # img_y_large = torch.Tensor(origin_y_large)

        if self.state == 'train':
            return img_x,img_x1,img_y,mask,x_path,y_path
            # return img_x,img_x1,expand_y,x_path,y_path

            # return torch.stack([img_x, img_x_flip],dim=0),img_x1,torch.stack([img_y,img_y_flip],dim=0),x_path,y_path
            # return img_x,img_x1,img_x_small,img_x_large,img_y,img_y_small,img_y_large,x_path,y_path
        else:
            return img_x,img_y,x_path,y_path
            # return img_x,img_x_small,img_x_large,img_y,img_y_small,img_y_large,x_path,y_path

    def __len__(self):
        return len(self.pics)


class TiAlDataset(data.Dataset):
    def __init__(self, state, classes, aug=True):
        self.state = state
        self.train_root = r"../data/TiAl/train"
        # self.train_root = r"../data/TiAl/train_two"
        # self.train_root = r"../data/TiAl/train_coarse"
        # self.train_root = r"../data/TiAl/train_full_1"
        # self.train_root = r"../data/TiAl/train_full"
        self.val_root = r"../data/TiAl/test_comparison"
        # self.val_root = r"../data/TiAl/test_more"
        self.test_root = self.val_root
        self.pics,self.masks = self.getDataPath()
        self.classes = classes

    def getDataPath(self):
        assert self.state =='train' or self.state == 'val' or self.state =='test'
        if self.state == 'train':
            root = self.train_root
        if self.state == 'val':
            root = self.val_root
        if self.state == 'test':
            root = self.test_root

        pics = []
        masks = []
        data_list = os.listdir(root)
        masks_list = [name for name in data_list if name[-10:-4] == "_label"]
        pics_list = [name for name in data_list if name not in masks_list]
        assert len(masks_list) == len(pics_list)
        masks_list.sort()
        pics_list.sort()
        for i in range(len(masks_list)):
            img = os.path.join(root, pics_list[i])
            mask = os.path.join(root, masks_list[i])
            pics.append(img)
            masks.append(mask)
        return pics,masks

    def __getitem__(self, index):
        x_path = self.pics[index]
        y_path = self.masks[index]
        origin_x = cv2.imread(x_path)
        origin_y = cv2.imread(y_path, cv2.COLOR_BGR2GRAY)

        if self.state == 'train':
            origin_x = origin_x / 255
            # flip
            # origin_x, _, origin_y = random_mirror(origin_x, origin_x, origin_y)
            # # resize
            # origin_x, _, origin_y = random_scale(origin_x, origin_x, origin_y)
            # # crop
            # origin_x, _, origin_y = random_crop(origin_x, origin_x, origin_y)
            # rotate
            # origin_x, _, origin_y = random_rotate(origin_x, origin_x, origin_y)
            # origin_x, _, origin_y = random_crop(origin_x, origin_x, origin_y)
            # 随机旋转,细化旋转角度
            origin_x, _, origin_y, mask = random_rotate_custom(origin_x, origin_x, origin_y, 18, True)
            # flip_x, flip_y = flip(origin_x, origin_y)

            origin_x, mask, origin_y = random_mirror(origin_x, mask, origin_y)
        
        if self.state == 'train':
            origin_x1 = origin_x * 255
            H,W,_ = origin_x1.shape
            segment_num = H*W//1000  # 1000;500
            # ----- randaugment -----
            randaugmentops = rand_augment_ops()
            randaugment = RandAugment(randaugmentops)
            origin_x_1 = Image.fromarray(cv2.cvtColor(origin_x1.astype(np.uint8),cv2.COLOR_BGR2RGB))
            mixed_x = randaugment(origin_x_1)
            mixed_x = np.array(mixed_x)
            mixed_x = mixed_x[:,:,(2,1,0)]
            mixed_x = mixed_x.astype(np.float32) / 255
            # ----- randaugment -----
            # ----- 超像素获取目标区域 -----
            # expand_origin_y = origin_y
            # segments = slic(origin_x1[:,:,::-1].astype(np.uint8), n_segments=segment_num, compactness=50)  # io.imread的输入通道跟cv2相反
            # for i in range(segment_num):
            #     for j in range(3):
            #         area = np.where(segments==i,j,-1)
            #         overlap = np.where(area==origin_y,1,0)
            #         count = np.bincount(overlap.flatten())
            #         if len(count) > 1:
            #             expand_origin_y = np.where(segments==i,j,expand_origin_y)
            # cv2.imwrite("expand_y.png", expand_origin_y)
            # exit(0)
            # ----- 超像素获取目标区域 -----
        
        # origin_x_small, origin_y_small = fixed_scale(origin_x, origin_y, 0.5)
        # origin_x_large, origin_y_large = fixed_scale(origin_x, origin_y, 0.75)

        trans = transforms.ToTensor()
        img_x = trans(origin_x).float()
        # img_x_small = trans(origin_x_small).float()
        # img_x_large = trans(origin_x_large).float()
        if self.state == 'train':
            img_x1 = trans(mixed_x).float()
            # img_x_flip = trans(flip_x).float()
            # img_y_flip = torch.Tensor(flip_y)
            # expand_y = torch.Tensor(expand_origin_y)
            mask = torch.Tensor(mask)
        img_y = torch.Tensor(origin_y)
        # img_y_small = torch.Tensor(origin_y_small)
        # img_y_large = torch.Tensor(origin_y_large)

        if self.state == 'train':
            return img_x,img_x1,img_y,mask,x_path,y_path
            # return img_x,img_x1,expand_y,x_path,y_path

            # return torch.stack([img_x, img_x_flip],dim=0),img_x1,torch.stack([img_y,img_y_flip],dim=0),x_path,y_path
            # return img_x,img_x1,img_x_small,img_x_large,img_y,img_y_small,img_y_large,x_path,y_path
        else:
            return img_x,img_y,x_path,y_path
            # return img_x,img_x_small,img_x_large,img_y,img_y_small,img_y_large,x_path,y_path

    def __len__(self):
        return len(self.pics)


class TiAlDatasetFull(data.Dataset):
    def __init__(self, state, classes, aug=True):
        self.state = state
        self.train_root = r"../data/TiAl/train_full"
        self.val_root = r"../data/TiAl/test_comparison"
        self.test_root = self.val_root
        self.pics,self.masks = self.getDataPath()
        self.classes = classes

    def getDataPath(self):
        assert self.state =='train' or self.state == 'val' or self.state =='test'
        if self.state == 'train':
            root = self.train_root
        if self.state == 'val':
            root = self.val_root
        if self.state == 'test':
            root = self.test_root

        pics = []
        masks = []
        data_list = os.listdir(root)
        masks_list = [name for name in data_list if name[-10:-4] == "_label"]
        pics_list = [name for name in data_list if name not in masks_list]
        assert len(masks_list) == len(pics_list)
        masks_list.sort()
        pics_list.sort()
        for i in range(len(masks_list)):
            img = os.path.join(root, pics_list[i])
            mask = os.path.join(root, masks_list[i])
            pics.append(img)
            masks.append(mask)
        return pics,masks

    def __getitem__(self, index):
        x_path = self.pics[index]
        y_path = self.masks[index]
        origin_x = cv2.imread(x_path)
        origin_y = cv2.imread(y_path, cv2.COLOR_BGR2GRAY)

        if self.state == 'train':
            origin_x = origin_x / 255
            # flip
            origin_x, _, origin_y = random_mirror(origin_x, origin_x, origin_y)
            # # resize
            # origin_x, _, origin_y = random_scale(origin_x, origin_x, origin_y)
            # # crop
            # origin_x, _, origin_y = random_crop(origin_x, origin_x, origin_y)
            # rotate
            origin_x, _, origin_y = random_rotate(origin_x, origin_x, origin_y)

        if self.state == 'train':
            origin_x1 = origin_x * 255
            randaugmentops = rand_augment_ops()
            randaugment = RandAugment(randaugmentops)
            origin_x1 = Image.fromarray(cv2.cvtColor(origin_x1.astype(np.uint8),cv2.COLOR_BGR2RGB))
            mixed_x = randaugment(origin_x1)
            mixed_x = np.array(mixed_x)
            mixed_x = mixed_x[:,:,(2,1,0)]
            mixed_x = mixed_x.astype(np.float32) / 255

        trans = transforms.ToTensor()
        img_x = trans(origin_x).float()
        if self.state == 'train':
            img_x1 = trans(mixed_x).float()
        img_y = torch.Tensor(origin_y)

        if self.state == 'train':
            return img_x,img_x1,img_y,x_path,y_path
        else:
            return img_x,img_y,x_path,y_path

    def __len__(self):
        return len(self.pics)


class TiAlDatasetCAM(data.Dataset):
    def __init__(self, state, classes, aug=True):
        self.state = state
        self.train_root = r"../data/TiAl/train"
        # self.train_root = r"../data/TiAl/train_coarse"
        self.val_root = r"../data/TiAl/test_comparison"
        self.test_root = self.val_root
        self.pics,self.masks = self.getDataPath()
        self.classes = classes

    def getDataPath(self):
        assert self.state =='train' or self.state == 'val' or self.state =='test'
        if self.state == 'train':
            root = self.train_root
        if self.state == 'val':
            root = self.val_root
        if self.state == 'test':
            root = self.test_root

        pics = []
        masks = []
        data_list = os.listdir(root)
        masks_list = [name for name in data_list if name[-10:-4] == "_label"]
        pics_list = [name for name in data_list if name not in masks_list]
        assert len(masks_list) == len(pics_list)
        masks_list.sort()
        pics_list.sort()
        for i in range(len(masks_list)):
            img = os.path.join(root, pics_list[i])
            mask = os.path.join(root, masks_list[i])
            pics.append(img)
            masks.append(mask)
        return pics,masks

    def __getitem__(self, index):
        x_path = self.pics[index]
        y_path = self.masks[index]
        origin_x = cv2.imread(x_path)
        origin_y = cv2.imread(y_path, cv2.COLOR_BGR2GRAY)
        origin_y = np.where(origin_y==0,origin_y,255)
        
        if self.state == 'train':
            origin_x = origin_x / 255
            # flip
            origin_x, _, origin_y = random_mirror(origin_x, origin_x, origin_y)
            # # resize
            # origin_x, _, origin_y = random_scale(origin_x, origin_x, origin_y)
            # # crop
            # origin_x, _, origin_y = random_crop(origin_x, origin_x, origin_y)
            # rotate
            origin_x, _, origin_y = random_rotate(origin_x, origin_x, origin_y)
            # origin_x, _, origin_y = random_crop(origin_x, origin_x, origin_y)

            # flip_x, flip_y = flip(origin_x, origin_y)
        
        if self.state == 'train':
            origin_x1 = origin_x * 255
            H,W,_ = origin_x1.shape
            segment_num = H*W//1000  # 1000;500
            # ----- randaugment -----
            randaugmentops = rand_augment_ops()
            randaugment = RandAugment(randaugmentops)
            origin_x_1 = Image.fromarray(cv2.cvtColor(origin_x1.astype(np.uint8),cv2.COLOR_BGR2RGB))
            mixed_x = randaugment(origin_x_1)
            mixed_x = np.array(mixed_x)
            mixed_x = mixed_x[:,:,(2,1,0)]
            mixed_x = mixed_x.astype(np.float32) / 255
            # ----- randaugment -----
            # ----- 超像素获取目标区域 -----
            # expand_origin_y = origin_y
            # segments = slic(origin_x1[:,:,::-1].astype(np.uint8), n_segments=segment_num, compactness=50)  # io.imread的输入通道跟cv2相反
            # for i in range(segment_num):
            #     for j in range(3):
            #         area = np.where(segments==i,j,-1)
            #         overlap = np.where(area==origin_y,1,0)
            #         count = np.bincount(overlap.flatten())
            #         if len(count) > 1:
            #             expand_origin_y = np.where(segments==i,j,expand_origin_y)
            # cv2.imwrite("expand_y.png", expand_origin_y)
            # exit(0)
            # ----- 超像素获取目标区域 -----
        
        # origin_x_small, origin_y_small = fixed_scale(origin_x, origin_y, 0.5)
        # origin_x_large, origin_y_large = fixed_scale(origin_x, origin_y, 0.75)

        trans = transforms.ToTensor()
        img_x = trans(origin_x).float()
        # img_x_small = trans(origin_x_small).float()
        # img_x_large = trans(origin_x_large).float()
        if self.state == 'train':
            img_x1 = trans(mixed_x).float()
            # img_x_flip = trans(flip_x).float()
            # img_y_flip = torch.Tensor(flip_y)
            # expand_y = torch.Tensor(expand_origin_y)
        img_y = torch.Tensor(origin_y)
        # img_y_small = torch.Tensor(origin_y_small)
        # img_y_large = torch.Tensor(origin_y_large)

        if self.state == 'train':
            return img_x,img_x1,img_y,x_path,y_path
            # return img_x,img_x1,expand_y,x_path,y_path

            # return torch.stack([img_x, img_x_flip],dim=0),img_x1,torch.stack([img_y,img_y_flip],dim=0),x_path,y_path
            # return img_x,img_x1,img_x_small,img_x_large,img_y,img_y_small,img_y_large,x_path,y_path
        else:
            return img_x,img_y,x_path,y_path
            # return img_x,img_x_small,img_x_large,img_y,img_y_small,img_y_large,x_path,y_path

    def __len__(self):
        return len(self.pics)


class micrographDataset(data.Dataset):
    def __init__(self, state, classes, aug=True):
        self.state = state
        self.train_root = r"../data/micrograph/train_full"
        self.val_root = r"../data/micrograph/test"
        self.test_root = self.val_root
        self.pics,self.masks = self.getDataPath()
        self.classes = classes

    def getDataPath(self):
        assert self.state =='train' or self.state == 'val' or self.state =='test'
        if self.state == 'train':
            root = self.train_root
        if self.state == 'val':
            root = self.val_root
        if self.state == 'test':
            root = self.test_root

        pics = []
        masks = []
        data_list = os.listdir(root)
        masks_list = [name for name in data_list if name[-10:-4] == "_label"]
        pics_list = [name for name in data_list if name not in masks_list]
        assert len(masks_list) == len(pics_list)
        masks_list.sort()
        pics_list.sort()
        for i in range(len(masks_list)):
            img = os.path.join(root, pics_list[i])
            mask = os.path.join(root, masks_list[i])
            pics.append(img)
            masks.append(mask)
        return pics,masks

    def __getitem__(self, index):
        x_path = self.pics[index]
        y_path = self.masks[index]
        origin_x = cv2.imread(x_path)
        origin_y = cv2.imread(y_path, cv2.COLOR_BGR2GRAY)

        if self.state == 'train':
            origin_x = origin_x / 255
            # flip
            # origin_x, _, origin_y = random_mirror(origin_x, origin_x, origin_y)
            # # resize
            # origin_x, _, origin_y = random_scale(origin_x, origin_x, origin_y)
            # # crop
            # origin_x, _, origin_y = random_crop(origin_x, origin_x, origin_y)
            # rotate
            # origin_x, _, origin_y = random_rotate(origin_x, origin_x, origin_y)

        if self.state == 'train':
            origin_x1 = origin_x * 255
            randaugmentops = rand_augment_ops()
            randaugment = RandAugment(randaugmentops)
            origin_x1 = Image.fromarray(cv2.cvtColor(origin_x1.astype(np.uint8),cv2.COLOR_BGR2RGB))
            mixed_x = randaugment(origin_x1)
            mixed_x = np.array(mixed_x)
            mixed_x = mixed_x[:,:,(2,1,0)]
            mixed_x = mixed_x.astype(np.float32) / 255

        trans = transforms.ToTensor()
        img_x = trans(origin_x).float()
        if self.state == 'train':
            img_x1 = trans(mixed_x).float()
        img_y = torch.Tensor(origin_y)
        if self.state == 'train':
            return img_x,img_x1,img_y,x_path,y_path
        else:
            return img_x,img_y,x_path,y_path

    def __len__(self):
        return len(self.pics)


class CeramicDataset(data.Dataset):
    def __init__(self, state, classes, aug=True):
        self.state = state
        self.train_root = r"../data/ceramic/train"
        # self.train_root = r"../data/ceramic/train_one"
        # self.train_root = r"../data/ceramic/train_full"
        # self.train_root = r"../data/ceramic/train_full_one"
        self.val_root = r"../data/ceramic/test"
        # -----
        # self.train_root = r"../data/ceramic/train1"
        # self.val_root = r"../data/ceramic/test1"
        # -----
        self.test_root = self.val_root
        self.pics,self.masks = self.getDataPath()
        self.classes = classes

    def getDataPath(self):
        assert self.state =='train' or self.state == 'val' or self.state =='test'
        if self.state == 'train':
            root = self.train_root
        if self.state == 'val':
            root = self.val_root
        if self.state == 'test':
            root = self.test_root

        pics = []
        masks = []
        data_list = os.listdir(root)
        masks_list = [name for name in data_list if name[-10:-4] == "_label"]
        pics_list = [name for name in data_list if name not in masks_list]
        assert len(masks_list) == len(pics_list)
        masks_list.sort()
        pics_list.sort()
        for i in range(len(masks_list)):
            img = os.path.join(root, pics_list[i])
            mask = os.path.join(root, masks_list[i])
            pics.append(img)
            masks.append(mask)
        return pics,masks

    def __getitem__(self, index):
        x_path = self.pics[index]
        y_path = self.masks[index]
        origin_x = cv2.imread(x_path)
        origin_y = cv2.imread(y_path, cv2.COLOR_BGR2GRAY)

        if self.state == 'train':
            origin_x = origin_x / 255
            # flip
            origin_x, _, origin_y = random_mirror(origin_x, origin_x, origin_y)  # 不知道为啥在这翻转比在后面翻转效果好...
            # # resize
            # origin_x, _, origin_y = random_scale(origin_x, origin_x, origin_y)
            # # crop
            # origin_x, _, origin_y = random_crop(origin_x, origin_x, origin_y)
            # rotate
            # origin_x, _, origin_y = random_rotate(origin_x, origin_x, origin_y)
            # origin_x, _, origin_y = random_crop(origin_x, origin_x, origin_y)
            # 随机旋转,细化旋转角度
            origin_x, _, origin_y, mask = random_rotate_custom(origin_x, origin_x, origin_y, 18, True)
            # flip_x, flip_y = flip(origin_x, origin_y)
        
            # origin_x, mask, origin_y = random_mirror(origin_x, mask, origin_y)

        if self.state == 'train':
            origin_x1 = origin_x * 255
            H,W,_ = origin_x1.shape
            segment_num = H*W//1000  # 1000;500
            # ----- randaugment -----
            randaugmentops = rand_augment_ops()
            randaugment = RandAugment(randaugmentops)
            origin_x_1 = Image.fromarray(cv2.cvtColor(origin_x1.astype(np.uint8),cv2.COLOR_BGR2RGB))
            mixed_x = randaugment(origin_x_1)
            mixed_x = np.array(mixed_x)
            mixed_x = mixed_x[:,:,(2,1,0)]
            mixed_x = mixed_x.astype(np.float32) / 255
            # ----- randaugment -----
            # ----- 超像素获取目标区域 -----
            # expand_origin_y = origin_y
            # segments = slic(origin_x1[:,:,::-1].astype(np.uint8), n_segments=segment_num, compactness=50)  # io.imread的输入通道跟cv2相反
            # for i in range(segment_num):
            #     for j in range(3):
            #         area = np.where(segments==i,j,-1)
            #         overlap = np.where(area==origin_y,1,0)
            #         count = np.bincount(overlap.flatten())
            #         if len(count) > 1:
            #             expand_origin_y = np.where(segments==i,j,expand_origin_y)
            # cv2.imwrite("expand_y.png", expand_origin_y)
            # exit(0)
            # ----- 超像素获取目标区域 -----
        
        # origin_x_small, origin_y_small = fixed_scale(origin_x, origin_y, 0.5)
        # origin_x_large, origin_y_large = fixed_scale(origin_x, origin_y, 0.75)

        trans = transforms.ToTensor()
        img_x = trans(origin_x).float()
        # img_x_small = trans(origin_x_small).float()
        # img_x_large = trans(origin_x_large).float()
        if self.state == 'train':
            img_x1 = trans(mixed_x).float()
            # img_x_flip = trans(flip_x).float()
            # img_y_flip = torch.Tensor(flip_y)
            # expand_y = torch.Tensor(expand_origin_y)
            mask = torch.Tensor(mask)
        img_y = torch.Tensor(origin_y)
        # img_y_small = torch.Tensor(origin_y_small)
        # img_y_large = torch.Tensor(origin_y_large)

        if self.state == 'train':
            return img_x,img_x1,img_y,mask,x_path,y_path
            # return img_x,img_x1,expand_y,x_path,y_path

            # return torch.stack([img_x, img_x_flip],dim=0),img_x1,torch.stack([img_y,img_y_flip],dim=0),x_path,y_path
            # return img_x,img_x1,img_x_small,img_x_large,img_y,img_y_small,img_y_large,x_path,y_path
        else:
            return img_x,img_y,x_path,y_path
            # return img_x,img_x_small,img_x_large,img_y,img_y_small,img_y_large,x_path,y_path

    def __len__(self):
        return len(self.pics)


class MixDataset(data.Dataset):
    def __init__(self, state, classes, aug=True):
        self.state = state
        self.train_root = r"../data/mix/train_2"
        # self.train_root = r"../data/mix/train_4"
        self.val_root = r"../data/mix/test"
        self.test_root = self.val_root
        self.pics,self.masks = self.getDataPath()
        self.classes = classes

    def getDataPath(self):
        assert self.state =='train' or self.state == 'val' or self.state =='test'
        if self.state == 'train':
            root = self.train_root
        if self.state == 'val':
            root = self.val_root
        if self.state == 'test':
            root = self.test_root

        pics = []
        masks = []
        data_list = os.listdir(root)
        masks_list = [name for name in data_list if name[-10:-4] == "_label"]
        pics_list = [name for name in data_list if name not in masks_list]
        assert len(masks_list) == len(pics_list)
        masks_list.sort()
        pics_list.sort()
        for i in range(len(masks_list)):
            img = os.path.join(root, pics_list[i])
            mask = os.path.join(root, masks_list[i])
            pics.append(img)
            masks.append(mask)
        return pics,masks

    def __getitem__(self, index):
        x_path = self.pics[index]
        y_path = self.masks[index]
        origin_x = cv2.imread(x_path)
        origin_y = cv2.imread(y_path, cv2.COLOR_BGR2GRAY)

        if self.state == 'train':
            origin_x = origin_x / 255
            # flip
            origin_x, _, origin_y = random_mirror(origin_x, origin_x, origin_y)
            # # resize
            # origin_x, _, origin_y = random_scale(origin_x, origin_x, origin_y)
            # # crop
            # origin_x, _, origin_y = random_crop(origin_x, origin_x, origin_y)
            # rotate
            # origin_x, _, origin_y = random_rotate(origin_x, origin_x, origin_y)
            # origin_x, _, origin_y = random_crop(origin_x, origin_x, origin_y)
            # 随机旋转,细化旋转角度
            origin_x, _, origin_y, mask = random_rotate_custom(origin_x, origin_x, origin_y, 18, True)
            # flip_x, flip_y = flip(origin_x, origin_y)
        
        if self.state == 'train':
            origin_x1 = origin_x * 255
            H,W,_ = origin_x1.shape
            segment_num = H*W//1000  # 1000;500
            # ----- randaugment -----
            randaugmentops = rand_augment_ops()
            randaugment = RandAugment(randaugmentops)
            origin_x_1 = Image.fromarray(cv2.cvtColor(origin_x1.astype(np.uint8),cv2.COLOR_BGR2RGB))
            mixed_x = randaugment(origin_x_1)
            mixed_x = np.array(mixed_x)
            mixed_x = mixed_x[:,:,(2,1,0)]
            mixed_x = mixed_x.astype(np.float32) / 255
            # ----- randaugment -----
            # ----- 超像素获取目标区域 -----
            # expand_origin_y = origin_y
            # segments = slic(origin_x1[:,:,::-1].astype(np.uint8), n_segments=segment_num, compactness=50)  # io.imread的输入通道跟cv2相反
            # for i in range(segment_num):
            #     for j in range(3):
            #         area = np.where(segments==i,j,-1)
            #         overlap = np.where(area==origin_y,1,0)
            #         count = np.bincount(overlap.flatten())
            #         if len(count) > 1:
            #             expand_origin_y = np.where(segments==i,j,expand_origin_y)
            # cv2.imwrite("expand_y.png", expand_origin_y)
            # exit(0)
            # ----- 超像素获取目标区域 -----
        
        # origin_x_small, origin_y_small = fixed_scale(origin_x, origin_y, 0.5)
        # origin_x_large, origin_y_large = fixed_scale(origin_x, origin_y, 0.75)

        trans = transforms.ToTensor()
        img_x = trans(origin_x).float()
        # img_x_small = trans(origin_x_small).float()
        # img_x_large = trans(origin_x_large).float()
        if self.state == 'train':
            img_x1 = trans(mixed_x).float()
            # img_x_flip = trans(flip_x).float()
            # img_y_flip = torch.Tensor(flip_y)
            # expand_y = torch.Tensor(expand_origin_y)
            mask = torch.Tensor(mask)
        img_y = torch.Tensor(origin_y)
        # img_y_small = torch.Tensor(origin_y_small)
        # img_y_large = torch.Tensor(origin_y_large)

        if self.state == 'train':
            return img_x,img_x1,img_y,mask,x_path,y_path
            # return img_x,img_x1,expand_y,x_path,y_path

            # return torch.stack([img_x, img_x_flip],dim=0),img_x1,torch.stack([img_y,img_y_flip],dim=0),x_path,y_path
            # return img_x,img_x1,img_x_small,img_x_large,img_y,img_y_small,img_y_large,x_path,y_path
        else:
            return img_x,img_y,x_path,y_path
            # return img_x,img_x_small,img_x_large,img_y,img_y_small,img_y_large,x_path,y_path

    def __len__(self):
        return len(self.pics)


class pascalDataset(data.Dataset):

    def __init__(self, state, classes, aug=False):
        self.state = state
        self.train_root = r"../data/pascalvoc2012augmented/img_aug"
        self.val_root = r"../data/pascalvoc2012augmented/img_aug"
        self.test_root = r"../data/pascalvoc2012augmented/img_test"
        self.train_list = r"../data/pascalvoc2012augmented/train.txt"
        self.val_list = r"../data/pascalvoc2012augmented/val.txt"
        self.test_list = r"../data/pascalvoc2012augmented/test.txt"
        self.class_map = r"../data/pascalvoc2012augmented/cls_aug"
        self.class_map_full = r"../data/pascalvoc2012augmented/cls_aug_all"
        self.pics,self.masks = self.getDataPath()
        self.aug = aug
        self.classes = classes

    def getDataPath(self):
        assert self.state =='train' or self.state == 'val' or self.state =='test'
        if self.state == 'train':
            root = self.train_root
            index_list = self.train_list
        if self.state == 'val':
            root = self.val_root
            index_list = self.val_list
        if self.state == 'test':
            root = self.test_root
            index_list = self.test_list

        pics = []
        masks = []
        image_index = open(index_list, "r")
        image_index_list = image_index.readlines()
        image_index.close()
        label_index_list = [name[:-1] + ".png" for name in image_index_list]
        image_index_list = [name[:-1] + ".jpg" for name in image_index_list]
        self.image_name_list = image_index_list
        for i in range(len(image_index_list)):
            img = os.path.join(root, image_index_list[i])
            if self.state == 'test':
                mask = os.path.join(root, image_index_list[i])  # pascal data set return origin image as masks in test
            elif self.state == 'val':
                mask = os.path.join(self.class_map_full, label_index_list[i])
            else:
                mask = os.path.join(self.class_map, label_index_list[i])
            pics.append(img)
            masks.append(mask)
        return pics,masks

    def __getitem__(self, index):
        x_path = self.pics[index]
        y_path = self.masks[index]
        origin_x = cv2.imread(x_path)
        if self.state == 'train':
            root = self.train_root
            index_list = self.train_list
            # image_index = open(index_list, "r")
            # image_index_list = image_index.readlines()
            # image_index_list = [name[:-1] + ".jpg" for name in image_index_list]
            # image_index.close()
            augAddImage = os.path.join(root, np.random.choice(self.image_name_list,1)[0])
            img_add = cv2.imread(augAddImage)
            img_add = img_add.astype(np.float32) / 255
        origin_x = origin_x.astype(np.float32) / 255
        if self.state == 'test':
            origin_y = cv2.imread(y_path)
        else:
            origin_y = cv2.imread(y_path, cv2.COLOR_BGR2GRAY)
        
        if self.state == 'train':  # 虽然有些代码里面是把验证集也做FixScaleCrop这种操作的,但是我感觉还是会影响图像信息,所以对于验证集和测试集一样,除了归一化操作没有做任何尺寸或者数值上的变换
            # flip
            origin_x, img_add, origin_y = random_mirror(origin_x, img_add, origin_y)
            # resize
            origin_x, img_add, origin_y = random_scale(origin_x, img_add, origin_y)
            # crop
            origin_x, img_add, origin_y = random_crop(origin_x, img_add, origin_y)

        # '''    
        if self.aug and self.state == 'train':
            # ----------AugMix----------  mix的操作感觉不适合分割工作,mix的重叠与像素级别分割存在矛盾
            # origin_x1 = origin_x * 255
            # augmixops = augmix_ops()
            # augmix = AugMixAugment(augmixops)
            # origin_x1 = Image.fromarray(cv2.cvtColor(origin_x1.astype(np.uint8),cv2.COLOR_BGR2RGB))
            # mixed_x = augmix(origin_x1)
            # mixed_x = np.array(mixed_x)
            # mixed_x = mixed_x[:,:,(2,1,0)]
            # mixed_x = mixed_x.astype(np.float32) / 255
            # ----------AugMix----------
            # ----------RandAugment----------  randaugment不包含mix操作,比起augmix更适合做语义分割上的增强
            origin_x1 = origin_x * 255
            randaugmentops = rand_augment_ops()
            randaugment = RandAugment(randaugmentops)
            origin_x1 = Image.fromarray(cv2.cvtColor(origin_x1.astype(np.uint8),cv2.COLOR_BGR2RGB))
            mixed_x = randaugment(origin_x1)
            mixed_x = np.array(mixed_x)
            mixed_x = mixed_x[:,:,(2,1,0)]
            mixed_x = mixed_x.astype(np.float32) / 255
            # ----------RandAugment----------
            # ----------Fourier Transform----------
            # origin_x1 = origin_x * 255
            # img_add = img_add * 255
            # fourierAugment = fourierAug(origin_x1,img_add)
            # mixed_x = fourierAugment.astype(np.float32) / 255
            # ----------Fourier Transform----------
        # '''

        origin_x = torch.Tensor(origin_x).permute(2,0,1)
        if self.aug and self.state == 'train':
            mixed_x = torch.Tensor(mixed_x).permute(2,0,1)
        x_transforms = transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ->[-1,1]
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        # mask只需要转换为tensor,转成0~1标签会出问题,不能用crossentropyloss
        # y_transforms = transforms.ToTensor()
        # img_y = y_transforms(origin_y)

        img_x = x_transforms(origin_x)
        if self.aug and self.state == 'train':
            img_x1 = x_transforms(mixed_x)
        img_y = torch.Tensor(origin_y)    
        
        if self.aug and self.state == 'train':
            return img_x, img_x1, img_y, x_path, y_path
        elif ~self.aug and self.state == 'train':
            return img_x, img_x, img_y, x_path, y_path
        else:
            return img_x, img_y, x_path, y_path

    def __len__(self):
        return len(self.pics)


class pascalDatasetTiny(data.Dataset):

    def __init__(self, state, classes, aug=False):
        self.state = state
        self.train_root = r"../data/pascalvoc2012augmented/img_aug"
        self.val_root = r"../data/pascalvoc2012augmented/img_aug"
        self.test_root = r"../data/pascalvoc2012augmented/img_test"
        self.train_list = r"../data/pascalvoc2012augmented/train_tiny.txt"
        self.val_list = r"../data/pascalvoc2012augmented/val_tiny.txt"

        # self.train_list = r"../data/pascalvoc2012augmented/train_1.txt"  # 训练图数量为1
        # self.val_list = r"../data/pascalvoc2012augmented/val_1.txt"  # 验证图数量为1

        self.test_list = r"../data/pascalvoc2012augmented/test_tiny.txt"
        self.class_map = r"../data/pascalvoc2012augmented/cls_aug"
        self.class_map_full = r"../data/pascalvoc2012augmented/cls_aug_all"
        self.pics,self.masks = self.getDataPath()
        self.aug = aug
        self.classes = classes

    def getDataPath(self):
        assert self.state =='train' or self.state == 'val' or self.state =='test'
        if self.state == 'train':
            root = self.train_root
            index_list = self.train_list
        if self.state == 'val':
            root = self.val_root
            index_list = self.val_list
        if self.state == 'test':
            root = self.test_root
            index_list = self.test_list

        pics = []
        masks = []
        image_index = open(index_list, "r")
        image_index_list = image_index.readlines()
        label_index_list = [name[:-1] + ".png" for name in image_index_list]
        image_index_list = [name[:-1] + ".jpg" for name in image_index_list]
        for i in range(len(image_index_list)):
            img = os.path.join(root, image_index_list[i])
            if self.state == 'test':
                mask = os.path.join(root, image_index_list[i])  # pascal data set return origin image as masks in test
            elif self.state == 'val':
                mask = os.path.join(self.class_map_full, label_index_list[i])
            else:
                mask = os.path.join(self.class_map, label_index_list[i])
            pics.append(img)
            masks.append(mask)
        return pics,masks

    def __getitem__(self, index):
        x_path = self.pics[index]
        y_path = self.masks[index]
        origin_x = cv2.imread(x_path)
        origin_x = origin_x.astype(np.float32) / 255
        if self.state == 'test':
            origin_y = cv2.imread(y_path)
        else:
            origin_y = cv2.imread(y_path, cv2.COLOR_BGR2GRAY)
        
        if self.state == 'train':  # 虽然有些代码里面是把验证集也做FixScaleCrop这种操作的,但是我感觉还是会影响图像信息,所以对于验证集和测试集一样,除了归一化操作没有做任何尺寸或者数值上的变换
            # flip
            origin_x, origin_y = random_mirror(origin_x, origin_y)
            # resize
            origin_x, origin_y = random_scale(origin_x, origin_y)
            # crop
            origin_x, origin_y = random_crop(origin_x, origin_y)

        '''    
        if self.aug and self.state == 'train':
            # ----------AugMix----------  mix的操作感觉不适合分割工作,mix的重叠与像素级别分割存在矛盾
            # origin_x1 = origin_x * 255
            # augmixops = augmix_ops()
            # augmix = AugMixAugment(augmixops)
            # origin_x1 = Image.fromarray(cv2.cvtColor(origin_x1.astype(np.uint8),cv2.COLOR_BGR2RGB))
            # mixed_x = augmix(origin_x1)
            # mixed_x = np.array(mixed_x)
            # mixed_x = mixed_x[:,:,(2,1,0)]
            # mixed_x = mixed_x.astype(np.float32) / 255
            # ----------AugMix----------
            # ----------RandAugment----------  randaugment不包含mix操作,比起augmix更适合做语义分割上的增强
            origin_x1 = origin_x * 255
            randaugmentops = rand_augment_ops()
            randaugment = RandAugment(randaugmentops)
            origin_x1 = Image.fromarray(cv2.cvtColor(origin_x1.astype(np.uint8),cv2.COLOR_BGR2RGB))
            mixed_x = randaugment(origin_x1)
            mixed_x = np.array(mixed_x)
            mixed_x = mixed_x[:,:,(2,1,0)]
            mixed_x = mixed_x.astype(np.float32) / 255
            # ----------RandAugment----------
        '''

        origin_x = torch.Tensor(origin_x).permute(2,0,1)
        if self.aug and self.state == 'train':
            mixed_x = torch.Tensor(mixed_x).permute(2,0,1)
        x_transforms = transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ->[-1,1]
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        # mask只需要转换为tensor,转成0~1标签会出问题,不能用crossentropyloss
        # y_transforms = transforms.ToTensor()
        # img_y = y_transforms(origin_y)

        img_x = x_transforms(origin_x)
        if self.aug and self.state == 'train':
            img_x1 = x_transforms(mixed_x)
        img_y = torch.Tensor(origin_y)    
        
        if self.aug and self.state == 'train':
            return img_x, img_x1, img_y, x_path, y_path
        elif ~self.aug and self.state == 'train':
            return img_x, img_x, img_y, x_path, y_path
        else:
            return img_x, img_y, x_path, y_path

    def __len__(self):
        return len(self.pics)

