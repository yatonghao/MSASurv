from math import ceil

import torch
import torch.nn as nn
from torch import nn, einsum
from einops import rearrange, reduce

import pdb

"""

Contains the custom implementation of cross attention between pathways and histology and self attention between pathways 

"""

NUM_PATHWAYS = 1280

def exists(val):
    return val is not None


class FeedForward(nn.Module):
    def __init__(self, dim, mult=1, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        hidden_features = int(2 * dim / 3)
        self.fc1 = nn.Linear(dim, hidden_features * 2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm(x)
        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.act(x) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MMAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        residual = True,
        residual_conv_kernel = 33,
        eps = 1e-8,
        dropout = 0.,
    ):
        super().__init__()
        self.eps = eps
        inner_dim = heads * dim_head

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding = (padding, 0), groups = heads, bias = False)

    def forward(self, x, numpath, mask=None, return_attn=False):
        b, n, _ = x.shape
        h = self.heads

        # derive query, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # set masked positions to 0 in queries, keys, values
        if mask != None:
            mask = rearrange(mask, 'b n -> b () n')
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        # regular transformer scaling
        q = q * self.scale

        # extract the pathway/histology queries and keys
        q_pathways = q[:, :, :numpath, :]  # bs x head x num_pathways x dim
        k_pathways = k[:, :, :numpath, :]

        q_histology = q[:, :, numpath:, :]  # bs x head x num_patches x dim
        k_histology = k[:, :, numpath:, :]

        # similarities
        einops_eq = '... i d, ... j d -> ... i j'
        cross_attn_histology = einsum(einops_eq, q_histology, k_pathways)
        attn_pathways = einsum(einops_eq, q_pathways, k_pathways)
        cross_attn_pathways = einsum(einops_eq, q_pathways, k_histology)
        attn_histology = einsum(einops_eq, q_histology, k_histology)
        # softmax

        attn_pathways_histology = torch.cat((attn_pathways, cross_attn_pathways), dim=-1).softmax(dim=-1)
        attn_histology_pathways = torch.cat((cross_attn_histology, attn_histology), dim=-1).softmax(dim=-1)

        # compute output
        out_pathways = attn_pathways_histology @ v
        out_histology = attn_histology_pathways @ v

        out = torch.cat((out_pathways, out_histology), dim=2)

        # add depth-wise conv residual of values
        if self.residual:
            out += self.res_conv(v)

        # merge and combine heads
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)

        return out


class MMAttentionLayer(nn.Module):
    """
    Applies layers norm --> attention
    """

    def __init__(
        self,
        norm_layer=nn.LayerNorm,
        dim=512,
        dim_head=64,
        heads=6,
        residual=True,
        dropout=0.,
    ):

        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = MMAttention(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            residual=residual,
            dropout=dropout,
        )

    def forward(self,numpath, x=None, mask=None, return_attention=False):

        if return_attention:
            x, attn_pathways, cross_attn_pathways, cross_attn_histology = self.attn(x=self.norm(x), mask=mask, return_attn=True)
            return x, attn_pathways, cross_attn_pathways, cross_attn_histology
        else:
            x = self.attn(numpath=numpath,x=self.norm(x), mask=mask)

        return x
