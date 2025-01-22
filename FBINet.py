import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.pvtv2 import *
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import get_unknown_tensor_from_pred
from einops import rearrange
import numbers
import torchvision.ops
from mmcv.cnn import ConvModule, build_upsample_layer
from einops.layers.torch import Rearrange


class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias=False):
        super(DeformableConv2d, self).__init__()

        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.dilation = dilation

        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     dilation=self.dilation,
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels,
                                        1 * kernel_size[0] * kernel_size[1],
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=self.padding,
                                        dilation=self.dilation,
                                        bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      dilation=self.dilation,
                                      bias=bias)

    def forward(self, x):
        # h, w = x.shape[2:]
        # max_offset = max(h, w)/4.

        offset = self.offset_conv(x)  # .clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        # op = (n - (k * d - 1) + 2p / s)
        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.regular_conv.weight,
                                          bias=self.regular_conv.bias,
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          dilation=self.dilation)
        return x

class DeformSpatialAttention(nn.Module):
    def __init__(self):
        super(DeformSpatialAttention, self).__init__()
        self.sa = DeformableConv2d(2,1,7,padding=3, bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.cat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn


class DeformChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(DeformChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)        
        self.ca = nn.Sequential(
            DeformableConv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            DeformableConv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn


class DeformPixelAttention(nn.Module):
    def __init__(self, dim):
        super(DeformPixelAttention, self).__init__()
        self.pa2 = DeformableConv2d(2 * dim, dim, 7, padding=3, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2
      
class DeformCGAFusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super(DeformCGAFusion, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Sequential(CBR(64, 64, 3, 1, 1), nn.Conv2d(64, 1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        initial = x + y
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result

def adj_index(h, k, node_num):
    dist = torch.cdist(h, h, p=2)
    each_adj_index = torch.topk(dist, k, dim=2).indices
    adj = torch.zeros(
        h.size(0), node_num, node_num, 
        dtype=torch.int, device=h.device, requires_grad = False
    ).scatter_(dim=2, index=each_adj_index, value=1)
    return adj


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))

        self.activation = nn.ELU()
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)  
        e = self._prepare_attentional_mechanism_input(Wh)
        attention = torch.where(adj > 0, e, -9e15 * torch.ones_like(e))
        attention = F.softmax(attention, dim=2)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return self.activation(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.transpose(1, 2)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class MDC(nn.Module):
    def __init__(self, in_channels):
        super(MDC, self).__init__()
      
        self.horizontal_conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), padding=(0, 1))
        self.vertical_conv = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1), padding=(1, 0))
        self.dilated_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2)
        self.adjust_channels = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)
        
    def forward(self, x):
        branch1 = self.horizontal_conv(x)
        branch2 = self.vertical_conv(x)
        branch3 = self.dilated_conv(x)
        out = torch.cat((branch1, branch2, branch3), dim=1)
        out = self.adjust_channels(out)
        out = F.relu(out)
        return out

class TPMFB(nn.Module):
    def __init__(self, in_feature, out_feature, top_k=7, token=3, alpha=0.2, num_heads=1):
        super(TPMFB, self).__init__()
      
        self.gat_branch = GAT(in_feature, out_feature, top_k, token, alpha, num_heads)
        self.dire_branch = MDB(in_feature)
        self.output_layer = nn.Conv2d(in_feature, out_feature, kernel_size=1)

    def forward(self, x):
      
        gat_out = self.gat_branch(x) 
        conv_out = self.dire_branch(x)  
        fused_out = gat_out + conv_out
        out = self.output_layer(fused_out)
        return out

class GAT(nn.Module):
    def __init__(self, in_feature, out_feature,top_k=4, token=3, alpha=0.2, num_heads=1):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.top_k = top_k
        hidden_feature = in_feature 
        self.conv = ConvModule(in_feature, hidden_feature, token, stride=token)
        self.attentions = [
            GraphAttentionLayer(
                hidden_feature, hidden_feature, alpha=alpha, concat=True
            )for _ in range(num_heads)
        ]
        
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(
            hidden_feature * num_heads, out_feature, alpha=alpha, concat=False)

        self.deconv = build_upsample_layer(
            cfg=dict(type='deconv', in_channels=out_feature, out_channels=out_feature, kernel_size=token, stride=token)
        )
        self.activation = nn.ELU()
        self._init_weights()

    def _init_weights(self):
        for m in [self.deconv]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        h = self.conv(x)
        batch_size, in_feature, column, row = h.shape  # h (N,C,H,W)
        node_num = column * row
        h = h.view(
            batch_size, in_feature, node_num).permute(0, 2, 1)  # h (N,H*W,in_feature)
        adj = adj_index(h, self.top_k, node_num)

        h = torch.cat([att(h, adj) for att in self.attentions], dim=2)
        h = self.activation(self.out_att(h, adj))

        h = h.view(batch_size, column, row, -1).permute(0, 3, 1, 2)
        h = F.interpolate(self.deconv(h), x.shape[-2:], mode='bilinear', align_corners=True)
        return F.relu(h+x)

    
class CBR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(CBR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups,bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DCAB(nn.Module):
    def __init__(self):
        super(DCAB, self).__init__()
        self.reduce1 = CBR(64, 64, 3, 1, 1)
        self.reduce4 = CBR(64, 64, 3, 1, 1)
        self.cga = CGAFusion(64)
        
    def forward(self, x4, x1):
        size = x1.size()[2:]
        x1 = self.reduce1(x1)
        x4 = self.reduce4(x4)
        x4 = F.interpolate(x4, size, mode='bilinear', align_corners=False)
        out = self.dcga(x4,x1)
        return out


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,kernel_size=kernel_size, stride=stride,padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x



class Net(nn.Module):
    def __init__(self, n_feat=64):
        super(Net, self).__init__()
        self.backbone = pvt_v2_b4()  # [64, 128, 320, 512]
        path = './pretrained_pvt/pvt_v2_b4.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        
        self.Translayer1_1 = BasicConv2d(64, n_feat, 1)
        self.Translayer2_1 = BasicConv2d(128, n_feat, 1)
        self.Translayer3_1 = BasicConv2d(320, n_feat, 1)
        self.Translayer4_1 = BasicConv2d(512, n_feat, 1)
      
        self.dcab = DCAB()
        self.tpmfb = TPMFB(64,64,top_k=3)
        
        
    def forward(self, x):
        pvt = self.backbone(x)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]

        x1_fe = self.Translayer1_1(x1)
        x2_fe = self.Translayer2_1(x2)
        x3_fe = self.Translayer3_1(x3)
        x4_fe = self.Translayer4_1(x4)
      
        x5_fe = self.tpmfb(x4_fe)
        edge_1 = self.dcab(x5_fe, x1_fe)
        edge_2 = self.dcab(x5_fe, x2_fe)
        edge_3 = self.dcab(x5_fe, x3_fe)
        edge_4 = self.dcab(x5_fe, x4_fe)
      
        oe_1 = F.interpolate(edge_1, scale_factor=4, mode='bilinear', align_corners=False)
        oe_2 = F.interpolate(edge_2, scale_factor=8, mode='bilinear', align_corners=False)
        oe_3 = F.interpolate(edge_3, scale_factor=16, mode='bilinear', align_corners=False)
        oe_4 = F.interpolate(edge_4, scale_factor=32, mode='bilinear', align_corners=False)

        return [oe_4,oe_3,oe_2,oe_1]



if __name__ == '__main__':
    model = Net().cuda()
    input_tensor = torch.randn(1, 3, 384, 384).cuda()
    prediction = model(input_tensor)
    print(prediction[-1].shape)
