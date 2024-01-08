from torch import nn
import torch
import torch.nn.init
from torch.nn import functional as F
import numpy as np


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
        Parameters:
        ----------
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
       
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
        Parameters:
        ----------
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

class PAM_CAM_Layer(nn.Module):
    """
    Helper Function for PAM and CAM attention
    
    Parameters:
    ----------
    input:
        in_ch : input channels
        use_pam : Boolean value whether to use PAM_Module or CAM_Module
    output:
        returns the attention map
    """
    def __init__(self, in_ch, use_pam = True):
        super(PAM_CAM_Layer, self).__init__()
        
        self.attn = nn.Sequential(
            nn.Conv2d(in_ch * 2, in_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.PReLU(),
            PAM_Module(in_ch) if use_pam else CAM_Module(in_ch),
			nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.PReLU()
        )
    
    def forward(self, x):
        return self.attn(x)

class FAM_Module(nn.Module):
    """ Feature map attention module"""
    def __init__(self, in_ch, num):
        super(FAM_Module, self).__init__()

        self.attn1 = nn.Sequential(
            nn.Conv2d(in_ch * num, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(0.5)
        self.attn2 = nn.Sequential(
			nn.Conv2d(512, 3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )

        # self.l1 = nn.Conv2d(in_ch * num, in_ch, kernel_size=3, stride=1, padding=1)
        # self.l2 = nn.BatchNorm2d(in_ch)
        # self.l3 = nn.ReLU()
        # self.l4 = nn.Dropout(ratio)
        # self.l5 = nn.Conv2d(in_ch, 3, kernel_size=1, stride=1)
        # self.l6 = nn.BatchNorm2d(3)
        # self.l7 = nn.ReLU()

    def forward(self,x,is_training=False):

        x = self.attn1(x)
        if is_training:
            x = self.dropout(x)
        output = self.attn2(x)
        
        return output
        # x = self.l1(x)
        # x = self.l2(x)
        # x = self.l3(x)
        # x = self.l4(x)
        # x = self.l5(x)
        # x = self.l6(x)
        # x = self.l7(x)
        # return x
        
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Self_Attn(nn.Module):  # from self-attention-gan; https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py
    """ Self attention Layer"""
    def __init__(self,in_dim):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out

class Label_Self_Attention(nn.Module):
    def __init__(self,in_dim,attention_dim,num_classes,factor=8,linformer=True):
        super(Label_Self_Attention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=attention_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=attention_dim, kernel_size=1)
        self.out_att_conv = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1)
        self.attn_dim = attention_dim
        self.linformer = linformer
        self.factor = factor
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self,input,logits):
        input_detach = input.detach()
        score = logits.detach()
        value_dim = score.shape[1]
        n,c,H,W = input_detach.size()
        query = self.query_conv(input_detach)
        key = self.key_conv(input_detach)
        query = torch.reshape(query, (-1,self.attn_dim,H*W))
        if self.linformer:
            key = F.interpolate(key, size=((H//self.factor+1),(W//self.factor+1)), mode='bilinear', align_corners=True)
            key = torch.reshape(key,(-1,self.attn_dim,(H//self.factor+1)*(W//self.factor+1)))
        key = key.transpose(0,2,1)
        matmul_qk = torch.matmul(key,query)
        scaled_att_logits = matmul_qk / math.sqrt(self.attn_dim)
        att_weights = self.softmax(scaled_att_logits)
        if self.linformer:
            value = F.interpolate(score, size=((H//self.factor+1),(W//self.factor+1)), mode='bilinear', align_corners=True)
            value = torch.reshape(value,(-1,self.attn_dim,(H//self.factor+1)*(W//self.factor+1)))
        att_score = torch.matmul(att_weights, value)
        att_score = torch.reshape(att_score, score.shape)
        att_score += score
        out_att_logits = self.out_att_conv(att_score)
        out_att_labels = self.softmax(out_att_logits)
        # return out_att_logits
        return out_att_labels 

class Map_Self_Attention(nn.Module):
    def __init__(self,in_dim,attention_dim,factor=8,linformer=False):
        super(Map_Self_Attention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=attention_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=attention_dim, kernel_size=1)
        self.attn_dim = attention_dim
        self.linformer = linformer
        self.factor = factor
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self,input):  # input往里面放的是特征,logits往里面放的是原始标注,输出是经过扩展的标注
        input_detach = input.detach()  # 保证梯度断开,对自注意力机制的学习不会破坏原本的特征
        # score = logits.detach()  # 保证梯度断开,对自注意力机制的学习不会破坏原本的特征
        # value_dim = score.shape[1]
        n,c,H,W = input_detach.size()
        query = self.query_conv(input_detach)
        key = self.key_conv(input_detach)
        query = torch.reshape(query, (-1,self.attn_dim,H*W))
        if self.linformer:
            key = F.interpolate(key, size=((H//self.factor+1),(W//self.factor+1)), mode='bilinear', align_corners=True)
            key = torch.reshape(key,(-1,self.attn_dim,(H//self.factor+1)*(W//self.factor+1)))
        else:
            key = torch.reshape(key, (-1,self.attn_dim,H*W))
        key = key.permute(0,2,1)
        return key,query
        '''
        # 这里的乘法必需得分块做,不然显存肯定不够
        matmul_qk = torch.matmul(key,query)  # -1,H*W,H*W
        scaled_att_logits = matmul_qk / math.sqrt(self.attn_dim)
        att_weights = self.softmax(scaled_att_logits)  # 像素权重关联矩阵
        if self.linformer:
            value = F.interpolate(score, size=((H//self.factor+1),(W//self.factor+1)), mode='bilinear', align_corners=True)
            value = torch.reshape(value,(-1,value_dim,(H//self.factor+1)*(W//self.factor+1)))
        else:
            value = torch.reshape(score,(-1,value_dim,H*W))
        att_score = torch.matmul(value, att_weights)  # -1,value_dim,H*W
        att_score = torch.reshape(att_score, score.shape)  # -1,value_dim,H,W
        att_score += score  # 本质上大概是上面一步给关联之后的点附了值,但是本身没有,这里把值加进去了
        return att_score
        ''' 

# class Custom_Self_Attention(nn.Module):
#     def __init__(self,in_dim,attention_dim):
#         super(Custom_Self_Attention,self).__init__()
#         self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=attention_dim, kernel_size=1)
#         self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=attention_dim, kernel_size=1)
#         self.attn_dim = attention_dim
#         #需要进一步实验确定是使用softmax还是sigmoid,猜想原始softmax可能效果更好,在显存没有问题的情况下考虑如何修改为对softmax进行监督
#         # self.softmax = nn.Softmax(dim=-1)  # 原始self_attention中使用的是softmax,作用是为了实现权重赋值
#         self.sigmoid = nn.Sigmoid()
    
#     def forward(self,input,class_input):  # input里面包含的全图的特征,class_input中包含的是某个类别真实标注部分的特征
#         input_detach = input.detach()  # 保证梯度断开,对自注意力机制的学习不会破坏原本的特征
#         class_input_detach = class_input.detach()  # 保证梯度断开,对自注意力机制的学习不会破坏原本的特征
#         n,c,S1,_ = class_input_detach.shape
#         n,c,S2,_ = input_detach.shape
#         query = self.query_conv(input_detach)
#         key = self.key_conv(class_input_detach)
#         query = torch.reshape(query, (-1,self.attn_dim,S2))
#         key = torch.reshape(key, (-1,self.attn_dim,S1))
#         key = key.permute(0,2,1)
#         attention_map = torch.matmul(key, query)
#         attention_map = self.sigmoid(attention_map)
#         return attention_map

# class Custom_Self_Attention(nn.Module):
#     def __init__(self,in_dim,attention_dim,classes):
#         super(Custom_Self_Attention,self).__init__()
#         self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=attention_dim, kernel_size=1)
#         self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=attention_dim, kernel_size=1)
#         self.output_conv = nn.Conv2d(in_channels=1, out_channels=classes, kernel_size=1)
#         self.attn_dim = attention_dim
#         self.classes = classes
#         #需要进一步实验确定是使用softmax还是sigmoid,猜想原始softmax可能效果更好,在显存没有问题的情况下考虑如何修改为对softmax进行监督
#         # self.softmax = nn.Softmax(dim=-1)  # 原始self_attention中使用的是softmax,作用是为了实现权重赋值
#         self.sigmoid = nn.Sigmoid()
    
#     def forward(self,input,class_input):  # input里面包含的全图的特征,class_input中包含的是某个类别真实标注部分的特征
#         input_detach = input.detach()  # 保证梯度断开,对自注意力机制的学习不会破坏原本的特征
#         class_input_detach = class_input.detach()  # 保证梯度断开,对自注意力机制的学习不会破坏原本的特征
#         n,c,S1,_ = class_input_detach.shape
#         n,c,height,width = input_detach.shape
#         input_detach = torch.reshape(input_detach,(n,c,height*width,1))
#         query = self.query_conv(input_detach)
#         key = self.key_conv(class_input_detach)
#         query = torch.reshape(query, (-1,self.attn_dim,height*width))
#         key = torch.reshape(key, (-1,self.attn_dim,S1))
#         key = key.permute(0,2,1)
#         attention_map = torch.matmul(key, query)
#         attention_map = attention_map.unsqueeze(0)
#         attention_map = attention_map.permute(2,1,0,3)
#         attention_map = self.output_conv(attention_map)
#         attention_map = torch.mean(attention_map,0).squeeze(1).unsqueeze(0)
#         attention_map = torch.reshape(attention_map,(n,self.classes,height,width))
#         return attention_map

# # add rnn for invariant length sequence
# class Custom_Self_Attention(nn.Module):
#     def __init__(self,in_dim,attention_dim,classes):
#         super(Custom_Self_Attention,self).__init__()
#         self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=attention_dim, kernel_size=1)
#         self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=attention_dim, kernel_size=1)
#         self.output_conv = nn.Conv2d(in_channels=1, out_channels=classes, kernel_size=1)
#         self.attn_dim = attention_dim
#         self.classes = classes
#         #需要进一步实验确定是使用softmax还是sigmoid,猜想原始softmax可能效果更好,在显存没有问题的情况下考虑如何修改为对softmax进行监督
#         # self.softmax = nn.Softmax(dim=-1)  # 原始self_attention中使用的是softmax,作用是为了实现权重赋值
#         self.sigmoid = nn.Sigmoid()
#         self.rnn = nn.RNN(307200,307200,2,batch_first=True)  # 这边的实现并没有做到把变长统一成定长
    
#     def forward(self,input,class_input):  # input里面包含的全图的特征,class_input中包含的是某个类别真实标注部分的特征
#         input_detach = input.detach()  # 保证梯度断开,对自注意力机制的学习不会破坏原本的特征
#         class_input_detach = class_input.detach()  # 保证梯度断开,对自注意力机制的学习不会破坏原本的特征
#         n,c,S1,_ = class_input_detach.shape
#         n,c,height,width = input_detach.shape
#         input_detach = torch.reshape(input_detach,(n,c,height*width,1))
#         query = self.query_conv(input_detach)
#         key = self.key_conv(class_input_detach)
#         query = torch.reshape(query, (-1,self.attn_dim,height*width))
#         key = torch.reshape(key, (-1,self.attn_dim,S1))
#         key = key.permute(0,2,1)
#         attention_map = torch.matmul(key, query)
#         attention_map = attention_map.unsqueeze(0)
#         attention_map = attention_map.permute(2,1,0,3)
#         attention_map = self.output_conv(attention_map)
#         attention_map = torch.mean(attention_map,0).squeeze(1).unsqueeze(0)
#         attention_map = torch.reshape(attention_map,(n,self.classes,height,width))
#         return attention_map

class Custom_Self_Attention(nn.Module):
    def __init__(self,in_dim,attention_dim,classes):
        super(Custom_Self_Attention,self).__init__()
        # self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=attention_dim, kernel_size=1)
        # self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=attention_dim, kernel_size=1)
        # self.embedding_conv_1 = nn.Conv2d(in_channels=in_dim, out_channels=attention_dim, kernel_size=1)
        # self.embedding_conv_2 = nn.Conv2d(in_channels=in_dim, out_channels=attention_dim, kernel_size=1)
        self.embedding_conv_1 = nn.Sequential(
            # nn.Conv2d(in_channels=in_dim, out_channels=attention_dim, kernel_size=1),
            nn.Conv2d(in_channels=in_dim, out_channels=attention_dim, kernel_size=3, padding=1),
            # nn.Conv2d(in_channels=in_dim, out_channels=attention_dim, kernel_size=3, padding=1),
            # nn.Conv2d(in_channels=attention_dim, out_channels=attention_dim, kernel_size=3, padding=1)
        )
        self.embedding_conv_2 = nn.Sequential(
            # nn.Conv2d(in_channels=in_dim, out_channels=attention_dim, kernel_size=1),
            nn.Conv2d(in_channels=in_dim, out_channels=attention_dim, kernel_size=3, padding=1),
            # nn.Conv2d(in_channels=in_dim, out_channels=attention_dim, kernel_size=3, padding=1),
            # nn.Conv2d(in_channels=attention_dim, out_channels=attention_dim, kernel_size=3, padding=1)
        )
        # self.embedding_conv_3 = nn.Sequential(
        #     # nn.Conv2d(in_channels=in_dim, out_channels=attention_dim, kernel_size=1),
        #     nn.Conv2d(in_channels=in_dim, out_channels=attention_dim, kernel_size=3, padding=1),
        #     # nn.Conv2d(in_channels=in_dim, out_channels=attention_dim, kernel_size=3, padding=1),
        #     # nn.Conv2d(in_channels=attention_dim, out_channels=attention_dim, kernel_size=3, padding=1)
        # )
        # self.embedding_conv_4 = nn.Sequential(
        #     # nn.Conv2d(in_channels=in_dim, out_channels=attention_dim, kernel_size=1),
        #     nn.Conv2d(in_channels=in_dim, out_channels=attention_dim, kernel_size=3, padding=1),
        #     # nn.Conv2d(in_channels=in_dim, out_channels=attention_dim, kernel_size=3, padding=1),
        #     # nn.Conv2d(in_channels=attention_dim, out_channels=attention_dim, kernel_size=3, padding=1)
        # )
        # self.embedding_conv_3 = nn.Sequential(
        #     nn.Conv2d(in_channels=in_dim, out_channels=attention_dim, kernel_size=3, padding=1)
        # )
        # self.embedding_conv_1 = nn.Sequential(
        #     nn.Conv2d(in_channels=in_dim, out_channels=attention_dim, kernel_size=7, padding=3),
        #     # nn.Conv2d(in_channels=attention_dim, out_channels=attention_dim, kernel_size=3, padding=1)
        # )
        # self.embedding_conv_2 = nn.Sequential(
        #     nn.Conv2d(in_channels=in_dim, out_channels=attention_dim, kernel_size=7, padding=3),
        #     # nn.Conv2d(in_channels=attention_dim, out_channels=attention_dim, kernel_size=3, padding=1)
        # )
        # self.embedding_conv_3 = nn.Sequential(
        #     nn.Conv2d(in_channels=in_dim, out_channels=attention_dim, kernel_size=7, padding=3),
        #     # nn.Conv2d(in_channels=attention_dim, out_channels=attention_dim, kernel_size=3, padding=1)
        # )
        # self.embedding_conv_4 = nn.Sequential(
        #     nn.Conv2d(in_channels=in_dim, out_channels=attention_dim, kernel_size=7, padding=3),
        #     # nn.Conv2d(in_channels=attention_dim, out_channels=attention_dim, kernel_size=3, padding=1)
        # )
        # self.embedding_conv_1 = nn.Conv2d(in_channels=in_dim, out_channels=attention_dim, kernel_size=5, padding=2)
        # self.embedding_conv_2 = nn.Conv2d(in_channels=in_dim, out_channels=attention_dim, kernel_size=5, padding=2)
        # self.embedding_conv_1 = nn.Conv2d(in_channels=in_dim, out_channels=attention_dim, kernel_size=7, padding=3)
        # self.embedding_conv_2 = nn.Conv2d(in_channels=in_dim, out_channels=attention_dim, kernel_size=7, padding=3)
        # self.embedding_conv_1 = nn.Conv2d(in_channels=in_dim, out_channels=attention_dim, kernel_size=9, padding=4)
        # self.embedding_conv_2 = nn.Conv2d(in_channels=in_dim, out_channels=attention_dim, kernel_size=9, padding=4)
        # self.embedding_conv_3 = nn.Conv2d(in_channels=in_dim, out_channels=attention_dim, kernel_size=1)
        self.combine_channel1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1)  # 双头
        self.combine_channel2 = nn.Conv2d(in_channels=2*classes, out_channels=classes, kernel_size=1)  # 双头
        # self.combine_channel1 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1)  # 三头
        # self.combine_channel2 = nn.Conv2d(in_channels=3*classes, out_channels=classes, kernel_size=1)  # 三头
        # self.combine_channel2_0 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1)  # 双头
        # self.combine_channel2_1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1)  # 双头
        # self.combine_channel2_2 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1)  # 双头
        # self.combine_channel = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)  # 单头仿双头实现
        # self.output_conv = nn.Conv2d(in_channels=1, out_channels=classes, kernel_size=1)
        self.attn_dim = attention_dim
        self.classes = classes
        #TODO:需要进一步实验确定是使用softmax还是sigmoid,猜想原始softmax可能效果更好,在显存没有问题的情况下考虑如何修改为对softmax进行监督
        # self.softmax = nn.Softmax(dim=-1)  # 原始self_attention中使用的是softmax,作用是为了实现权重赋值
        self.sigmoid = nn.Sigmoid()
        # self.bn = nn.BatchNorm2d(img_size)
    
    def forward(self,input,class_input,class_label,saved_class_input,saved_class_label):  # input里面包含的全图的特征,class_input中包含的是某个类别真实标注部分的特征
        '''
        # ----- 混在一起了 -----
        input_detach = input.detach()  # 保证梯度断开,对自注意力机制的学习不会破坏原本的特征
        class_input_detach = class_input.detach()  # 保证梯度断开,对自注意力机制的学习不会破坏原本的特征
        # input_detach = input
        # class_input_detach = class_input
        n,c,S1,_ = class_input_detach.shape
        n,c,height,width = input_detach.shape
        n,c,S2,_ = saved_class_input.shape
        input_detach = torch.reshape(input_detach,(n,c,height*width,1))
        # query = self.query_conv(input_detach)
        # key = self.key_conv(class_input_detach)
        query1 = self.embedding_conv_1(input_detach)
        query2 = self.embedding_conv_2(input_detach)
        key1 = self.embedding_conv_1(class_input_detach)
        key2 = self.embedding_conv_2(class_input_detach)
        saved_key1 = self.embedding_conv_1(saved_class_input)
        saved_key2 = self.embedding_conv_2(saved_class_input)
        query1 = torch.reshape(query1, (-1,self.attn_dim,height*width))
        query2 = torch.reshape(query2, (-1,self.attn_dim,height*width))
        key1 = torch.reshape(key1, (-1,self.attn_dim,S1))
        key2 = torch.reshape(key2, (-1,self.attn_dim,S1))
        saved_key1 = torch.reshape(saved_key1, (-1,self.attn_dim,S2))
        saved_key2 = torch.reshape(saved_key2, (-1,self.attn_dim,S2))
        # ----- dot product norm -----
        saved_key1_ = saved_key1 ** 2
        key1_ = key1 ** 2
        saved_key1_ = torch.sum(saved_key1_, 1)
        key1_ = torch.sum(key1_, 1)
        saved_key1_ = saved_key1_ ** 0.5
        key1_ = key1_ ** 0.5
        memory_norm = torch.matmul(saved_key1_.permute(1,0),key1_)
        # ----- dot product norm -----
        memory_map1 = torch.matmul(saved_key1.permute(0,2,1),key1)
        memory_map1 = memory_map1 / memory_norm
        # print(np.unique(memory_map1.flatten().detach().cpu().numpy()))
        
        
        # ----- dot product norm -----
        saved_key1_ = saved_key2 ** 2
        key1_ = key2 ** 2
        saved_key1_ = torch.sum(saved_key1_, 1)
        key1_ = torch.sum(key1_, 1)
        saved_key1_ = saved_key1_ ** 0.5
        key1_ = key1_ ** 0.5
        memory_norm = torch.matmul(saved_key1_.permute(1,0),key1_)
        # ----- dot product norm -----
        memory_map2 = torch.matmul(saved_key2.permute(0,2,1),key2)
        memory_map2 = memory_map2 / memory_norm
        # print(np.unique(memory_map2.flatten().detach().cpu().numpy()))
        
        
        memory_map1 = memory_map1.squeeze(0)  # -1~1
        memory_map2 = memory_map2.squeeze(0)  # -1~1

        key1 = key1.permute(0,2,1)
        key2 = key2.permute(0,2,1)
        # ----- dot product norm -----
        query1_ = query1 ** 2
        query1_ = torch.sum(query1_, 1)
        query1_ = query1_ ** 0.5
        key1_ = key1 ** 2
        key1_ = torch.sum(key1_, 2)
        key1_ = key1_ ** 0.5
        attention_norm = torch.matmul(key1_.permute(1,0),query1_)
        # ----- dot product norm -----
        attention_map1 = torch.matmul(key1, query1)
        attention_map1 = attention_map1 / attention_norm
        # print(np.unique(attention_map1.flatten().detach().cpu().numpy()))
        
        
        # ----- dot product norm -----
        query1_ = query2 ** 2
        query1_ = torch.sum(query1_, 1)
        query1_ = query1_ ** 0.5
        key1_ = key2 ** 2
        key1_ = torch.sum(key1_, 2)
        key1_ = key1_ ** 0.5
        attention_norm = torch.matmul(key1_.permute(1,0),query1_)
        # ----- dot product norm -----
        attention_map2 = torch.matmul(key2, query2)
        attention_map2 = attention_map2 / attention_norm
        # print(np.unique(attention_map2.flatten().detach().cpu().numpy()))
        
        
        # print(np.unique(attention_map1.detach().cpu().numpy()))

        # attention_map1 = self.sigmoid(attention_map1)
        # attention_map2 = self.sigmoid(attention_map2)
        attention_map1 = attention_map1.squeeze(0)
        attention_map2 = attention_map2.squeeze(0)

        # attention_map_cat = torch.cat((attention_map1, attention_map2),0).unsqueeze(0)
        # attention_map = self.combine_channel(attention_map_cat)
        # attention_map = attention_map.squeeze(0).squeeze(0)
        # attention_map = self.sigmoid(attention_map)
        # attention_map = (attention_map1 + 1) /2

        class_label = class_label.unsqueeze(0)
        result_map1 = None
        result_map2 = None
        # result_map = None
        for i in range(self.classes):
            temp_index = torch.where(class_label == i, 1, 0).float()
            count_num = len(torch.nonzero(temp_index))
            if result_map1 == None or result_map2 == None:
                result_map1 = torch.matmul(temp_index, attention_map1).unsqueeze(0) / count_num
                result_map2 = torch.matmul(temp_index, attention_map2).unsqueeze(0) / count_num
            else:
                result_map1 = torch.cat((result_map1, torch.matmul(temp_index, attention_map1).unsqueeze(0) / count_num), dim=0)
                result_map2 = torch.cat((result_map2, torch.matmul(temp_index, attention_map2).unsqueeze(0) / count_num), dim=0)
            # if result_map == None:
            #     result_map = torch.matmul(temp_index, attention_map).unsqueeze(0) / count_num
            # else:
            #     result_map = torch.cat((result_map, torch.matmul(temp_index, attention_map).unsqueeze(0) / count_num), dim=0)
            # if result_map1 == None:
            #     result_map1 = torch.matmul(temp_index, attention_map1).unsqueeze(0) / count_num
            # else:
            #     result_map1 = torch.cat((result_map1, torch.matmul(temp_index, attention_map1).unsqueeze(0) / count_num), dim=0)
        result_map1 = result_map1.unsqueeze(0)
        result_map2 = result_map2.unsqueeze(0)

        # result_map = result_map.unsqueeze(0)
        result_map1 = torch.reshape(result_map1, (1,self.classes,height,width))
        result_map2 = torch.reshape(result_map2, (1,self.classes,height,width))
        # result_map = torch.reshape(result_map, (1,self.classes,height,width))

        memory_map_stack = torch.cat((memory_map1.unsqueeze(0),memory_map2.unsqueeze(0)),0).unsqueeze(0)  # -1~1
        memory_map = self.combine_channel1(memory_map_stack)
        memory_map = memory_map.squeeze(0).squeeze(0)
        memory_map = self.sigmoid(memory_map)  # 这边相当于同时要做两个头的合并以及范围转换
        
        # memory_map = (memory_map1 + 1) / 2
        
        # memory_map = self.combine_channel(memory_map1.unsqueeze(0).unsqueeze(0))
        # memory_map = memory_map.squeeze(0).squeeze(0)
        # memory_map = self.sigmoid(memory_map)

        result_map_stack = torch.cat((result_map1,result_map2),1)
        result_map = self.combine_channel2(result_map_stack)
        result_map = self.sigmoid(result_map)

        # result_map = self.combine_channel(result_map1.permute(1,0,2,3)).permute(1,0,2,3)
        return result_map, memory_map
        # return result_map1, memory_map
        # return result_map, supervision_map, memory_map
        # ----- 混在一起了 -----
        '''

        
        # ----- 多头 -----
        input_detach = input.detach()  # 保证梯度断开,对自注意力机制的学习不会破坏原本的特征
        class_input_detach = class_input.detach()  # 保证梯度断开,对自注意力机制的学习不会破坏原本的特征
        n,c,S1,_ = class_input_detach.shape
        n,c,height,width = input_detach.shape
        n,c,S2,_ = saved_class_input.shape
        input_detach = torch.reshape(input_detach,(n,c,height*width,1))
        query1 = self.embedding_conv_1(input_detach)
        query2 = self.embedding_conv_2(input_detach)
        # query3 = self.embedding_conv_3(input_detach)
        # query4 = self.embedding_conv_4(input_detach)
        key1 = self.embedding_conv_1(class_input_detach)
        key2 = self.embedding_conv_2(class_input_detach)
        # key3 = self.embedding_conv_3(class_input_detach)
        # key4 = self.embedding_conv_4(class_input_detach)
        saved_key1 = self.embedding_conv_1(saved_class_input)
        saved_key2 = self.embedding_conv_2(saved_class_input)
        # saved_key3 = self.embedding_conv_3(saved_class_input)
        # saved_key4 = self.embedding_conv_4(saved_class_input)
        query1 = torch.reshape(query1, (-1,self.attn_dim,height*width))
        query2 = torch.reshape(query2, (-1,self.attn_dim,height*width))
        # query3 = torch.reshape(query3, (-1,self.attn_dim,height*width))
        # query4 = torch.reshape(query4, (-1,self.attn_dim,height*width))
        key1 = torch.reshape(key1, (-1,self.attn_dim,S1))
        key2 = torch.reshape(key2, (-1,self.attn_dim,S1))
        # key3 = torch.reshape(key3, (-1,self.attn_dim,S1))
        # key4 = torch.reshape(key4, (-1,self.attn_dim,S1))
        saved_key1 = torch.reshape(saved_key1, (-1,self.attn_dim,S2))
        saved_key2 = torch.reshape(saved_key2, (-1,self.attn_dim,S2))
        # saved_key3 = torch.reshape(saved_key3, (-1,self.attn_dim,S2))
        # saved_key4 = torch.reshape(saved_key4, (-1,self.attn_dim,S2))
        # ----- dot product norm -----
        query1_ = saved_key1 ** 2
        key1_ = key1 ** 2
        query1_ = torch.sum(query1_, 1)
        key1_ = torch.sum(key1_, 1)
        query1_ = query1_ ** 0.5
        key1_ = key1_ ** 0.5
        norm = torch.matmul(query1_.permute(1,0),key1_)
        # ----- dot product norm -----
        memory_map1 = torch.matmul(saved_key1.permute(0,2,1),key1)
        memory_map1 = memory_map1 / norm
        
        
        # ----- dot product norm -----
        query1_ = saved_key2 ** 2
        key1_ = key2 ** 2
        query1_ = torch.sum(query1_, 1)
        key1_ = torch.sum(key1_, 1)
        query1_ = query1_ ** 0.5
        key1_ = key1_ ** 0.5
        norm = torch.matmul(query1_.permute(1,0),key1_)
        # ----- dot product norm -----
        memory_map2 = torch.matmul(saved_key2.permute(0,2,1),key2)
        memory_map2 = memory_map2 / norm


        # # ----- dot product norm -----
        # query1_ = saved_key3 ** 2
        # key1_ = key3 ** 2
        # query1_ = torch.sum(query1_, 1)
        # key1_ = torch.sum(key1_, 1)
        # query1_ = query1_ ** 0.5
        # key1_ = key1_ ** 0.5
        # norm = torch.matmul(query1_.permute(1,0),key1_)
        # # ----- dot product norm -----
        # memory_map3 = torch.matmul(saved_key3.permute(0,2,1),key3)
        # memory_map3 = memory_map3 / norm


        # # ----- dot product norm -----
        # query1_ = saved_key4 ** 2
        # key1_ = key4 ** 2
        # query1_ = torch.sum(query1_, 1)
        # key1_ = torch.sum(key1_, 1)
        # query1_ = query1_ ** 0.5
        # key1_ = key1_ ** 0.5
        # norm = torch.matmul(query1_.permute(1,0),key1_)
        # # ----- dot product norm -----
        # memory_map4 = torch.matmul(saved_key4.permute(0,2,1),key4)
        # memory_map4 = memory_map4 / norm
        
        memory_map1 = memory_map1.squeeze(0)  # -1~1
        memory_map2 = memory_map2.squeeze(0)  # -1~1
        # memory_map3 = memory_map3.squeeze(0)  # -1~1
        # memory_map4 = memory_map4.squeeze(0)  # -1~1

        key1 = key1.permute(0,2,1)
        key2 = key2.permute(0,2,1)
        # key3 = key3.permute(0,2,1)
        # key4 = key4.permute(0,2,1)
        # ----- dot product norm -----
        query1_ = query1 ** 2
        query1_ = torch.sum(query1_, 1)
        query1_ = query1_ ** 0.5
        key1_ = key1 ** 2
        key1_ = torch.sum(key1_, 2)
        key1_ = key1_ ** 0.5
        norm = torch.matmul(key1_.permute(1,0),query1_)
        # ----- dot product norm -----
        attention_map1 = torch.matmul(key1, query1)
        attention_map1 = attention_map1 / norm        
        
        # ----- dot product norm -----
        query1_ = query2 ** 2
        query1_ = torch.sum(query1_, 1)
        query1_ = query1_ ** 0.5
        key1_ = key2 ** 2
        key1_ = torch.sum(key1_, 2)
        key1_ = key1_ ** 0.5
        norm = torch.matmul(key1_.permute(1,0),query1_)
        # ----- dot product norm -----
        attention_map2 = torch.matmul(key2, query2)
        attention_map2 = attention_map2 / norm

        # # ----- dot product norm -----
        # query1_ = query3 ** 2
        # query1_ = torch.sum(query1_, 1)
        # query1_ = query1_ ** 0.5
        # key1_ = key3 ** 2
        # key1_ = torch.sum(key1_, 2)
        # key1_ = key1_ ** 0.5
        # norm = torch.matmul(key1_.permute(1,0),query1_)
        # # ----- dot product norm -----
        # attention_map3 = torch.matmul(key3, query3)
        # attention_map3 = attention_map3 / norm

        # # ----- dot product norm -----
        # query1_ = query4 ** 2
        # query1_ = torch.sum(query1_, 1)
        # query1_ = query1_ ** 0.5
        # key1_ = key4 ** 2
        # key1_ = torch.sum(key1_, 2)
        # key1_ = key1_ ** 0.5
        # norm = torch.matmul(key1_.permute(1,0),query1_)
        # # ----- dot product norm -----
        # attention_map4 = torch.matmul(key4, query4)
        # attention_map4 = attention_map4 / norm

        '''
        # ----- 先合并后计算 -----  原尺寸图过大太慢
        attention_map_stack = torch.cat((attention_map1.unsqueeze(0),attention_map2.unsqueeze(0)),1)
        attention_map = self.combine_channel1(attention_map_stack)
        attention_map = self.sigmoid(attention_map)
        attention_map = attention_map.squeeze(0).squeeze(0)
        
        class_label = class_label.unsqueeze(0)
        result_map = None
        for i in range(self.classes):
            temp_index = torch.where(class_label == i, 1, 0).float()
            count_num = len(torch.nonzero(temp_index))
            if result_map == None:
                result_map = torch.matmul(temp_index, attention_map).unsqueeze(0) / count_num
            else:
                result_map = torch.cat((result_map, torch.matmul(temp_index, attention_map).unsqueeze(0) / count_num), dim=0)

        result_map = result_map.unsqueeze(0)

        print(result_map)
    
        result_map = torch.reshape(result_map, (1,self.classes,height,width))
        # ----- 先合并后计算 -----
        '''

        
        # ----- 先计算后合并 -----
        attention_map1 = (attention_map1 + 1) / 2  # -1~1归一化到0~1
        attention_map2 = (attention_map2 + 1) / 2  # -1~1归一化到0~1
        # attention_map3 = (attention_map3 + 1) / 2  # -1~1归一化到0~1
        # attention_map4 = (attention_map4 + 1) / 2  # -1~1归一化到0~1

        attention_map1 = attention_map1.squeeze(0)
        attention_map2 = attention_map2.squeeze(0)
        # attention_map3 = attention_map3.squeeze(0)
        # attention_map4 = attention_map4.squeeze(0)

        class_label = class_label.unsqueeze(0)
        result_map1 = None
        result_map2 = None
        # result_map3 = None
        # result_map4 = None
        for i in range(self.classes):
            temp_index = torch.where(class_label == i, 1, 0).float()
            count_num = len(torch.nonzero(temp_index))
            if count_num == 0:  # TODO:目前该处理方案对于不加边缘不合适;避免出现除0的情况导致nan,主要是因为边界旋转而不使用全图导致的
                count_num = height * width  # 其实这里随便放个非0值应该都可以,因为temp_index里面的值都是0,所以乘完的矩阵内的所有值还都是0
            if result_map1 == None or result_map2 == None:
            # if result_map1 == None or result_map2 == None or result_map3 == None:
            # if result_map1 == None or result_map2 == None or result_map3 == None or result_map4 == None:
                result_map1 = torch.matmul(temp_index, attention_map1).unsqueeze(0) / count_num
                result_map2 = torch.matmul(temp_index, attention_map2).unsqueeze(0) / count_num
                # result_map3 = torch.matmul(temp_index, attention_map3).unsqueeze(0) / count_num
                # result_map4 = torch.matmul(temp_index, attention_map4).unsqueeze(0) / count_num
            else:
                result_map1 = torch.cat((result_map1, torch.matmul(temp_index, attention_map1).unsqueeze(0) / count_num), dim=0)
                result_map2 = torch.cat((result_map2, torch.matmul(temp_index, attention_map2).unsqueeze(0) / count_num), dim=0)
                # result_map3 = torch.cat((result_map3, torch.matmul(temp_index, attention_map3).unsqueeze(0) / count_num), dim=0)
                # result_map4 = torch.cat((result_map4, torch.matmul(temp_index, attention_map4).unsqueeze(0) / count_num), dim=0)
        result_map1 = result_map1.unsqueeze(0)
        result_map2 = result_map2.unsqueeze(0)
        # result_map3 = result_map3.unsqueeze(0)
        # result_map4 = result_map4.unsqueeze(0)

        # print(result_map1)
        # print(result_map2)

        result_map1 = torch.reshape(result_map1, (1,self.classes,height,width))
        result_map2 = torch.reshape(result_map2, (1,self.classes,height,width))
        # result_map3 = torch.reshape(result_map3, (1,self.classes,height,width))
        # result_map4 = torch.reshape(result_map4, (1,self.classes,height,width))

        # 下面是直接从2*classes拉到classes的,效果好像不行
        result_map_stack = torch.cat((result_map1,result_map2),1)
        # result_map_stack = torch.cat((result_map_stack,result_map3),1)
        # result_map_stack = torch.cat((result_map_stack,result_map4),1)
        # result_map_stack = result_map_stack.detach()  # 这里如果断开梯度就是为了让attention_loss不影响memory_loss
        result_map = self.combine_channel2(result_map_stack)

        # result_map = (result_map1 + result_map2) / 2

        # result_map_stack_0 = torch.stack((result_map1[:,0,:,:], result_map2[:,0,:,:]),1)
        # result_map_0 = self.combine_channel2_0(result_map_stack_0)
        # result_map_stack_1 = torch.stack((result_map1[:,1,:,:], result_map2[:,1,:,:]),1)
        # result_map_1 = self.combine_channel2_1(result_map_stack_1)
        # result_map_stack_2 = torch.stack((result_map1[:,2,:,:], result_map2[:,2,:,:]),1)
        # result_map_2 = self.combine_channel2_2(result_map_stack_2)
        # result_map = torch.cat((result_map_0, result_map_1),1)
        # result_map = torch.cat((result_map, result_map_2),1)
        # ----- 先计算后合并 -----
        

        memory_map_stack = torch.cat((memory_map1.unsqueeze(0),memory_map2.unsqueeze(0)),0)  # -1~1
        # memory_map_stack = torch.cat((memory_map_stack,memory_map3.unsqueeze(0)),0)  # -1~1
        # memory_map_stack = torch.cat((memory_map_stack,memory_map4.unsqueeze(0)),0)  # -1~1
        memory_map_stack = memory_map_stack.unsqueeze(0)
        memory_map = self.combine_channel1(memory_map_stack)
        memory_map = memory_map.squeeze(0).squeeze(0)
        memory_map = self.sigmoid(memory_map)  # 这边相当于同时要做两个头的合并以及范围转换

        # memory_map = (memory_map1 + memory_map2) / 2

        return result_map, memory_map
        # ----- 多头 -----
        

        '''
        # ----- 单头 -----
        input_detach = input.detach()  # 保证梯度断开,对自注意力机制的学习不会破坏原本的特征
        class_input_detach = class_input.detach()  # 保证梯度断开,对自注意力机制的学习不会破坏原本的特征
        n,c,S1,_ = class_input_detach.shape
        n,c,height,width = input_detach.shape
        n,c,S2,_ = saved_class_input.shape
        input_detach = torch.reshape(input_detach,(n,c,height*width,1))
        query1 = self.embedding_conv_1(input_detach)
        key1 = self.embedding_conv_1(class_input_detach)
        saved_key1 = self.embedding_conv_1(saved_class_input)
        query1 = torch.reshape(query1, (-1,self.attn_dim,height*width))
        key1 = torch.reshape(key1, (-1,self.attn_dim,S1))
        saved_key1 = torch.reshape(saved_key1, (-1,self.attn_dim,S2))
        # ----- dot product norm -----
        saved_key1_ = saved_key1 ** 2
        key1_ = key1 ** 2
        saved_key1_ = torch.sum(saved_key1_, 1)
        key1_ = torch.sum(key1_, 1)
        saved_key1_ = saved_key1_ ** 0.5
        key1_ = key1_ ** 0.5
        memory_norm = torch.matmul(saved_key1_.permute(1,0),key1_)
        # ----- dot product norm -----
        memory_map1 = torch.matmul(saved_key1.permute(0,2,1),key1)
        memory_map1 = memory_map1 / memory_norm
        
        memory_map1 = memory_map1.squeeze(0)  # -1~1

        key1 = key1.permute(0,2,1)
        # ----- dot product norm -----
        query1_1 = query1 ** 2
        query1_1 = torch.sum(query1_1, 1)
        query1_1 = query1_1 ** 0.5
        key1_1 = key1 ** 2
        key1_1 = torch.sum(key1_1, 2)
        key1_1 = key1_1 ** 0.5
        attention_norm = torch.matmul(key1_1.permute(1,0),query1_1)
        # ----- dot product norm -----
        attention_map1 = torch.matmul(key1, query1)
        attention_map1 = attention_map1 / attention_norm
        
        attention_map1 = attention_map1.squeeze(0)

        attention_map1 = (attention_map1 + 1) /2

        class_label = class_label.unsqueeze(0)
        result_map1 = None
        # print(attention_map1)
        for i in range(self.classes):
            temp_index = torch.where(class_label == i, 1, 0).float()
            count_num = len(torch.nonzero(temp_index))
            # print(count_num)
            # print(torch.matmul(temp_index, attention_map1))
            if result_map1 == None:
                result_map1 = torch.matmul(temp_index, attention_map1).unsqueeze(0) / count_num
            else:
                result_map1 = torch.cat((result_map1, torch.matmul(temp_index, attention_map1).unsqueeze(0) / count_num), dim=0)
        result_map1 = result_map1.unsqueeze(0)
        
        # print(result_map1)
        # exit(0)

        result_map1 = torch.reshape(result_map1, (1,self.classes,height,width))

        memory_map = (memory_map1 + 1) / 2

        return result_map1, memory_map
        # ----- 单头 -----
        '''

        '''
        attention_map = attention_map.detach().unsqueeze(0)
        attention_map = attention_map.permute(2,1,0,3)
        attention_map = self.output_conv(attention_map)
        attention_map = torch.mean(attention_map,0).squeeze(1).unsqueeze(0)
        attention_map = torch.reshape(attention_map,(n,self.classes,height,width))
        return attention_map, supervision_map
        '''