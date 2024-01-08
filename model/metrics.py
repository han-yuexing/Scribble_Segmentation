import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix

def get_miou(mask, predict, classes):
    image_mask = torch.squeeze(mask, dim=0).long().numpy()

    # ----------下面代码计算的miou去掉了标柱中的忽略类别造成的影响----------
    ioumetric = IOUMetric(classes)
    ioumetric.add_batch(np.expand_dims(predict,0), np.expand_dims(image_mask,0))
    acc, acc_cls, iu, mean_iu, fwavacc = ioumetric.evaluate()

    # ----------下面代码计算的miou没有去掉了标柱中的忽略类别造成的影响----------
    # 这里最后算miou的时候没有忽略类别所以用这个也没关系
    result_image = predict
    label_image = image_mask

    height, width = result_image.shape

    false_count = 0
    correct_count = 0

    confusion_matrix = np.zeros((classes, classes))
    result_matrix = np.zeros((result_image.shape[0], result_image.shape[1]))
    label_matrix = np.zeros((label_image.shape[0], label_image.shape[1]))

    sign = label_image == result_image
    sign = sign.flatten().tolist()
    correct_count = sign.count(1)
    false_count = len(sign) - correct_count
    label_matrix = label_image
    result_matrix = result_image

    accuracy = correct_count/(correct_count+false_count)

    label_matrix = label_matrix.flatten().tolist()
    result_matrix = result_matrix.flatten().tolist()
    mcm = multilabel_confusion_matrix(label_matrix, result_matrix, labels=[i for i in range(classes)])

    tp = mcm[:, 1, 1]
    tn = mcm[:, 0, 0]
    fn = mcm[:, 1, 0]
    fp = mcm[:, 0, 1]
    mIOU = tp / (fp + fn + tp)
    print("Each phase IOU list:", mIOU)
    # mean_mIOU = np.nanmean(mIOU)
    # print("function1:", mean_mIOU)
    # use_sign = np.isnan(mIOU)
    # count = 0
    # sum = 0
    # for i in range(len(use_sign.tolist())):
    #     if ~use_sign[i]:
    #         sum += mIOU[i]
    #         count += 1
    # mean_mIOU = sum/count
    # print("function2:", mean_mIOU)

    return mean_iu, iu


class My_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, aug_output, labels, epoch):
        mask = torch.where(labels == 255, torch.ones_like(labels), torch.zeros_like(labels))
        output_1 = output.max(1)[1]
        aug_output_1 = aug_output.max(1)[1]
        output_1 = output_1 * mask  # 虽然乘0会导致这些区域全部判成0类,但是这里需要得到的实际上是类别差异,所以不影响
        aug_output_1 = aug_output_1 * mask
        loss = nn.CrossEntropyLoss(ignore_index=255)(output, labels)
        aug_loss = torch.where(output_1 == aug_output_1, torch.zeros_like(labels), torch.ones_like(labels)).sum() / (output.shape[2] * output.shape[3])
        print("supervised loss:", loss)
        print("consistence loss", aug_loss)
        total_loss = loss + aug_loss / 100  # 之前试的(epoch + 1) / 10效果不好,loss压不下去,可能是因为线性太大?先用固定比例试下
        return total_loss
        

class Custom_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predict, pseudo_label, feature):
        predict_prob = F.softmax(predict).max(1)[0].squeeze(0).cpu().detach().numpy()
        predict_cls = predict.max(1)[1].squeeze(0).cpu().detach().numpy()
        predict = predict.cpu().detach().numpy().squeeze(0).transpose(1,2,0)
        pseudo_label = pseudo_label.squeeze(0).cpu().detach().numpy()
        feature = feature.squeeze(0).cpu().detach().numpy().transpose(1,2,0)

        h,w = predict_prob.shape
        # print(predict_prob, predict_prob.shape)
        # print(predict_cls, predict_cls.shape)
        # print(predict, predict.shape)
        # print(pseudo_label, pseudo_label.shape)
        # print(feature, feature.shape)
        # exit(0)

        # loss = []
        # for i, cls in enumerate(pseudo_label):
        #     if cls == 255:
        #         continue
        #     x_class = -predict[i][cls]
        #     log_x_j = np.log(sum([np.exp(j) for j in predict[i] ]) )
        #     loss.append((x_class + log_x_j) * last_prob[i])

        loss = []
        for i in range(h):
            for j in range(w):
                loss_temp = []
                for m in [-5,-4,-3,-2,-1,1,2,3,4,5]:
                    if i + m < 0 or i + m >= h:
                        continue
                    for n in [-5,-4,-3,-2,-1,1,2,3,4,5]:
                        if j + n < 0 or j + n >= w:
                            continue
                        if predict_cls[i,j] == predict_cls[i+m,j+n]:
                            continue  # 类别相同的话是不是也需要加一个损失监督
                        distance = np.sqrt(m**2 + n**2)
                        feature_dis = np.linalg.norm(feature[i,j] - feature[i+m,j+n])
                        norm_loss = np.exp(-feature_dis*distance)
                        loss_temp.append(norm_loss)
                loss.append(np.mean(loss_temp))

        loss = torch.Tensor(loss)
        result = torch.mean(loss)
        return result.to(device)


'''
# https://www.cnblogs.com/king-lps/p/9497836.html
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, weight=None, ignore_index=255):
        super().__init__()
        self.alpha = alpha  # alpha=0.5
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)

    def forward(self, preds, labels):
        logpt = -self.ce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss
'''


# https://www.jianshu.com/p/30043bcc90b6
class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)  # 可以把255这些不算进来
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, iu, mean_iu, fwavacc


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, output, target, weight, device):
        target = target.detach()
        ls=nn.LogSoftmax(dim=1)
        log_softmax=ls(output)
        bsize,h,w=target.shape[0],target.shape[1],target.shape[2]
        loss=0

        for b in range(bsize):
            # m = torch.nn.Softmax(dim=0)
            # temp_softmax_output = m(weight[b])
            # mask_fill = torch.max(temp_softmax_output,0)[0]

            # mask_fill = torch.max(weight[b],0)[0]
            
            # m = torch.nn.Softmax(dim=0)
            # temp_softmax_output = m(output[b])
            # mask_fill = torch.max(temp_softmax_output,0)[0]

            # mask_fill = torch.max(output[b],0)[0]  # 这个会有负值,但鬼知道为啥效果还蛮好的
            # zeros_map = torch.zeros_like(mask_fill)
            # mask_fill = torch.where(mask_fill<0,zeros_map,mask_fill)
            
            mask_fill = torch.ones_like(output[b])
            
            # mask_fill = torch.randint(0,3,(output[b].shape[1],output[b].shape[2])).to(device)

            # print(torch.min(mask_fill.flatten()))

            # 获取每个像素点的真实类别标签
            target1 = torch.where(target==255,0,target)
            ind = target1[b, :, :].type(torch.int64).unsqueeze(0)
            # print('ind:',ind.shape)#torch.Size([1, 256, 256])
            
            # 获取预测得到的每个像素点的类别取值分布（3代表类别）
            pred_channels = log_softmax[b,:,:,:]
            # print('pred_3channels:',pred_3channels.shape)#torch.Size([3, 256, 256])
            
            # 使用gather，在第0个维度（类别所在维度）上用ind进行索引得到每个像素点的value
            pred = -pred_channels.gather(0,ind)
            # print('pred:',pred.shape)#torch.Size([1, 256, 256])
            
            # 添加了这句代码，通过两者的点乘实现了对每个像素点的加权
            pred = pred * mask_fill
            
            zero_map = torch.zeros_like(pred).float()
            pred = torch.where(target==255,zero_map,pred)
            #求这些像素点value的平均值，并累加到总的loss中
            current_loss = torch.mean(pred)
            loss += current_loss
        weighted_loss = loss / bsize
        return weighted_loss
