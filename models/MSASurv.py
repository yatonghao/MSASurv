import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.cross_attention import FeedForward, MMAttentionLayer

class MultiScaleDWConv(nn.Module):
    def __init__(self, dim=768, scale=(1, 3, 5)):
        super().__init__()
        self.scale = scale
        self.channels = []
        self.proj = nn.ModuleList()
        for i in range(len(scale)):
            if i == 0:
                channels = dim - dim // len(scale) * (len(scale) - 1)
            else:
                channels = dim // len(scale)
            conv = nn.Conv2d(channels, channels, kernel_size=scale[i], padding=scale[i]//2, groups=channels)
            self.channels.append(channels)
            self.proj.append(conv)

    def forward(self, x):
        x = torch.split(x, split_size_or_sections=self.channels, dim=1)
        out = []
        for i, feat in enumerate(x):
            out.append(self.proj[i](feat))
        x = torch.cat(out, dim=1)
        return x

class MSConvolutionalGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.dwconv = MultiScaleDWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x, v = self.fc1(x).chunk(2, dim=-1)
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


from nystrom_attention import NystromAttention
class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.MSConGLU = MSConvolutionalGLU(in_features=dim)

    def forward(self, x, H, W):
        x = self.norm(x)
        x = x + self.MSConGLU(x, H, W)
        return x

class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.
        )

    def forward(self, x):
        x = self.attn(self.norm(x))
        return x

class Attn_Net(nn.Module):

    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x  # N x n_classes

class MSASurv(nn.Module):
    def __init__(self, num_classes=4,):
        super(MSASurv, self).__init__()
        self.num_classes = num_classes
        self.in_chans = 768
        self.embed_dim = 512

        self.pos_layer = PPEG(512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)

        self.pos_layer_10x = PPEG(512)
        self.layer2_10x = TransLayer(dim=512)
        self.norm_10x = nn.LayerNorm(512)

        self.norm2 = nn.LayerNorm(512)
        self.identity = nn.Identity()
        self.layer3 = MMAttentionLayer(
            dim=512,
            dim_head=512,
            heads=1,
            residual=True,
            dropout=0.
        )

        self.layer4 = TransLayer(dim=512)
        self.norm4 = nn.LayerNorm(512)
        self.ffn = FeedForward(512)
        self.norm5 = nn.LayerNorm(512)

        self.feed_forward = FeedForward(512)
        self.norm3 = nn.LayerNorm(512)

        self.fc2 = nn.Sequential(nn.Linear(768, 512), nn.ReLU())
        self.fc1 = nn.Sequential(nn.Linear(768, 512), nn.ReLU())

        self.to_logits = nn.Sequential(
            nn.Linear(512, int(self.embed_dim / 5)),
            nn.ReLU(),
            nn.Linear(int(self.embed_dim / 5), self.num_classes)
        )

        self.attnpool = Attn_Net(L=512, D=512, dropout=False, n_classes=1)


    def forward(self, **kwargs):
        wsi_20x = kwargs['x_20x']
        wsi_10x = kwargs['x_10x']

        group_after_attn_20x = []
        for h in wsi_20x:
            h = self.fc1(h)
            H = h.shape[1]
            _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
            add_length = _H * _W - H
            h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 512]

            h = self.layer2(h)  # [B, N, 512]

            h = self.pos_layer(h, _H, _W)  # [B, N, 512]

            group_after_attn_20x.append(h)

        msges_20x = torch.cat(group_after_attn_20x, dim=1)

        group_after_attn_10x = []
        for h in wsi_10x:
            h = self.fc2(h)
            H = h.shape[1]
            _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
            add_length = _H * _W - H
            h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 512]

            h = self.layer2_10x(h)  # [B, N, 512]

            h = self.pos_layer_10x(h, _H, _W)  # [B, N, 512]

            group_after_attn_10x.append(h)

        msges_10x = torch.cat(group_after_attn_10x, dim=1)

        numpath = msges_10x.size(1)
        msges = torch.cat([msges_10x, msges_20x], dim=1)

        msges = self.norm2(msges)
        msges = self.identity(msges)
        msges = self.layer3(x=msges, numpath=numpath)

        msges = msges + self.feed_forward(msges)
        msges = self.norm3(msges)

        msges = self.layer4(msges)
        msges = msges + self.ffn(msges)
        msges = self.norm4(msges)

        A, msges = self.attnpool(msges.squeeze(0))  # B C 1
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        msges = torch.mm(A, msges)

        logits = self.to_logits(msges)

        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        results_dict = {'hazards': hazards, 'S': S, 'Y_hat': Y_hat}

        return results_dict


