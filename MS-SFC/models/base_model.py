import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone_res12 import ResNet
from models.conv4 import ConvNet




class Backbone(nn.Module):

    def __init__(self, args,  mode=None):
        super().__init__()
        self.mode = mode
        self.args = args
        self.resnet = resnet
        self.scale = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.method  = args.method
        self.k = args.way * args.shot

        self.encoder_dim = 64
        self.encoder = ConvNet()
        self.fc = nn.Linear(64, self.args.num_class)
        print("This is ConvNet")

        self.conv = nn.Sequential(
            nn.Conv2d(self.encoder_dim, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )


    def forward(self, input):
        if self.mode == 'fc':
            return self.fc_forward(input)

        elif self.mode == 'encoder':

            x = self.encoder(input)

            return x

        elif self.mode == 'base':
            spt,qry = input

            return self.metric(spt,qry)

        else:
            raise ValueError('Unknown mode')


    def fc_forward(self, x):

        x = x.mean(dim=[-1,-2])
        # x = x.mean(dim=[-1])
        return self.fc(x)


    def metric(self, token_support, token_query):
        qry_pooled = token_query.mean(dim=[-1,-2])

        spt = self.normalize_feature(token_support)
        qry = self.normalize_feature(token_query)

        corr4d = self.SAM(spt,qry)  #q s n1 n2
        num_qry, way, H_s, W_s, H_q, W_q = corr4d.size()

        corr4d_s = corr4d.view(num_qry, way, H_s * W_s, H_q, W_q)
        corr4d_q = corr4d.view(num_qry, way, H_s, W_s, H_q * W_q)


        corr4d_s = self.gaussian_normalize(corr4d_s, dim=2)
        corr4d_q = self.gaussian_normalize(corr4d_q, dim=4)

        corr4d_s = F.softmax(corr4d_s / self.args.temperature_attn, dim=2)
        corr4d_s = corr4d_s.view(num_qry, way, H_s, W_s, H_q, W_q)
        corr4d_q = F.softmax(corr4d_q / self.args.temperature_attn, dim=4)
        corr4d_q = corr4d_q.view(num_qry, way, H_s, W_s, H_q, W_q)

        attn_s = corr4d_s.sum(dim=[4, 5])
        attn_q = corr4d_q.sum(dim=[2, 3])

        spt_attended = attn_s.unsqueeze(2) * spt.unsqueeze(0)
        qry_attended = attn_q.unsqueeze(2) * qry.unsqueeze(1)

        spt_attended_pooled = spt_attended.mean(dim=[-1, -2])
        qry_attended_pooled = qry_attended.mean(dim=[-1, -2])

        similarity_matrix = F.cosine_similarity(spt_attended_pooled, qry_attended_pooled, dim=-1,eps=1e-6)
        logits = similarity_matrix * self.scale

        if self.training:
            return logits, self.fc(qry_pooled)
        else:
            return logits

    def SAM(self, spt, qry):

        way = spt.shape[0]
        num_qry = qry.shape[0]

        # reduce channel size via 1x1 conv
        spt = self.conv(spt)
        qry = self.conv(qry)

        # normalize channels for later cosine similarity
        spt = F.normalize(spt, p=2, dim=1, eps=1e-8)
        qry = F.normalize(qry, p=2, dim=1, eps=1e-8)

        spt = spt.unsqueeze(0).repeat(num_qry, 1, 1, 1, 1)
        qry = qry.unsqueeze(1).repeat(1, way, 1, 1, 1)

        similarity_map_einsum = torch.einsum('qncij,qnckl->qnijkl', spt, qry)

        return similarity_map_einsum


    def gaussian_normalize(self, x, dim, eps=1e-05):
        x_mean = torch.mean(x, dim=dim, keepdim=True)
        x_var = torch.var(x, dim=dim, keepdim=True)
        x = torch.div(x - x_mean, torch.sqrt(x_var + eps))
        return x


    def normalize_feature(self, x):
        return x - x.mean(1).unsqueeze(1)

