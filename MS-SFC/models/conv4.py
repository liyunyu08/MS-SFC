import math
import torch
import torch.nn as nn
from models.SFSM import SFSM
from kymatio.torch import Scattering2D
import torch.nn.functional as F
from thop import profile


def kd_loss_fn(student_logits, teacher_logits, T=3.0):
    s = F.log_softmax(student_logits / T, dim=1)
    t = F.softmax(teacher_logits / T, dim=1).detach()
    loss = F.kl_div(s, t, reduction='batchmean') * (T * T)
    return loss

def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0] * L.kernel_size[1] * L.out_channels
        L.weight.data.normal_(0, math.sqrt(2.0 / float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)

class Conv(nn.Module):
    """简单封装 Conv2d + BN + ReLU"""
    def __init__(self, in_ch, out_ch, k, g=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=k//2, groups=g, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class SPM(nn.Module):
    def __init__(self, in_channels, out_channels, n_groups=2, base_k=3):
        super().__init__()
        self.n_groups = n_groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 每组通道数
        ch_per_group = in_channels // n_groups
        assert in_channels % n_groups == 0, "in_channels must be divisible by n_groups"

        # 构造多个卷积 (3,5,7,...)
        self.convs = nn.ModuleList()
        k = base_k
        for _ in range(n_groups):
            self.convs.append(Conv(ch_per_group, ch_per_group, k, g=ch_per_group))
            k += 2

        # 最后的 1x1 卷积融合
        self.fuse = Conv(in_channels, out_channels, 1)

    def forward(self, x):
        xs = torch.chunk(x, self.n_groups, dim=1)
        ys = []
        for i in range(self.n_groups):
            if i == 0:
                y = self.convs[i](xs[i])
            else:
                y = self.convs[i](xs[i] + ys[i-1])  # 层级残差连接
            ys.append(y)

        out = torch.cat(ys, dim=1)
        out = self.fuse(out)

        # 残差连接
        if out.shape[1] == x.shape[1]:
            out = out + x
        return out

class FPM(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch,reduction=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),  # 降维
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, groups=mid_ch, bias=False),  # depthwise
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),  # 映射到 CNN 通道
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(out_ch, out_ch // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(out_ch // reduction, out_ch, bias=False),
            nn.Sigmoid()
        )
        for m in self.net.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d):
                try:
                    init_layer(m)
                except Exception:
                    pass

    def forward(self, x):
        x = self.net(x)
        b, c, _, _ = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        out = x * y.expand_as(x)

        return out


class Scattering(nn.Module):
    def __init__(self, H,W,J,channels,L=8):
        super(Scattering, self).__init__()
        self.scat = Scattering2D(J=J, L=L,shape=(H, W))
        self.num_coeffs = int(1 + L * J + L * L * J * (J - 1) / 2)

    def forward(self, x):
        x = torch.mean(x, dim=1, keepdims=True)
        scat_feat = self.scat(x)

        b,c,n,h,w = scat_feat.shape

        scat_high = scat_feat[:, :, 1:, :, :]
        scat_high = scat_high.contiguous().view(b, -1, h, w)
        return scat_high

    # def drop_featuremaps(self,feature_maps):
    #     A = torch.sum(feature_maps, dim=1, keepdim=True)
    #     a = torch.max(A, dim=3, keepdim=True)[0]
    #     a = torch.max(a, dim=2, keepdim=True)[0]
    #     # a = torch.mean(A, dim=[2, 3], keepdim=True)
    #     # M = (A > a).float()
    #     threshold = 0.1
    #     M = (A >= threshold * a).float()
    #     fm_temp = feature_maps * M
    #     return fm_temp


class ConvBlock(nn.Module):

    def __init__(self, indim, outdim, pool=True, padding=1):
        super(ConvBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim

        self.C = nn.Conv2d(indim, outdim, 3, padding=padding)
        self.BN = nn.BatchNorm2d(outdim)
        self.relu = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C, self.BN, self.relu]
        if pool:
            self.pool = nn.MaxPool2d(2)
            self.parametrized_layers.append(self.pool)

        for layer in self.parametrized_layers:
            init_layer(layer)

        self.trunk = nn.Sequential(*self.parametrized_layers)

    def forward(self, x):
        out = self.trunk(x)
        return out


class ConvNet(nn.Module):
    def __init__(self,depth=4):
        super(ConvNet, self).__init__()
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool=(i <3))  # only pooling for fist 4 layers
            trunk.append(B)

        self.trunk = nn.Sequential(*trunk)
        self.scat = Scattering(H=84,W=84,J=2,channels=64)
        self.fpm = FPM(in_ch=80, mid_ch=32, out_ch=64)
        self.fuse = SFSM(channels=64, attn_heads=2, use_coords=True)
        self.spm = SPM(64,64)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.cls_spa = nn.Linear(outdim, 5)
        self.cls_fre = nn.Linear(outdim, 5)


    def forward(self, spac):
        wst_feat = self.scat(spac)
        wst_feat =self.fpm(wst_feat)

        out_0 = self.trunk[0](spac)
        out_1 = self.trunk[1](out_0)
        feat_1 = self.spm(out_1)
        feat_1 = self.fuse(feat_1, wst_feat)
        out_2 = self.trunk[2](feat_1)
        out_3 = self.trunk[3](out_2)

        spa_pool = self.global_avgpool(out_3).view(out_3.size(0), -1)
        fre_pool = self.global_avgpool(wst_feat).view(wst_feat.size(0), -1)

        logit_spa = self.cls_spa(spa_pool)
        logit_fre = self.cls_fre(fre_pool)

        kd_loss = kd_loss_fn(logit_spa, logit_fre, T=2.0)

        return {
            'out3': out_3,
            'kd_loss': kd_loss
        }



if __name__ == '__main__':

    support_5shot = torch.randn(100, 3, 84, 84)
    support_1shot = torch.randn(80, 3, 84, 84)
    tem = ConvNet()
    flops1, params1 = profile(tem, inputs=(support_1shot, ),verbose=False)
    flops5, params5 = profile(tem, inputs=(support_5shot, ),verbose=False)
    print('1-shot'+'FLOPs1 = ' + str(flops1 / 1000 ** 3) + 'B', 'Params1 = ' + str(params1 / 1000 ** 2) + 'M')
    print('5-shot'+'FLOPs1 = ' + str(flops5 / 1000 ** 3) + 'B', 'Params1 = ' + str(params5 / 1000 ** 2) + 'M')
