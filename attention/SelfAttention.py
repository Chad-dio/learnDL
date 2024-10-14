import numpy as np
import torch
from torch import nn

from attention.Attention import ScaledDotProductAttention
from attention.MultiHeadAttention import MultiHeadAttention


class SelfAttention(nn.Module):
    def __init__(self, n_head, d_k, d_v, d_x, d_o):
        """
        nn.Parameter:tensor的一个子类， 默认会将 requires_grad=True
        主要作用：表示该张量是一个可以训练的参数，会被加到self.parameters()中
        """
        self.wq = nn.Parameter(torch.Tensor(d_x, d_k))
        self.wk = nn.Parameter(torch.Tensor(d_x, d_k))
        self.wv = nn.Parameter(torch.Tensor(d_x, d_v))

        self.mha = MultiHeadAttention(n_head=n_head, d_k_=d_k, d_v_=d_v, d_k=d_k, d_v=d_v, d_o=d_o)
        self.ha = ScaledDotProductAttention(scale=np.power(d_k, 0.5))
        self.init_parameters()

    # 初始化Wq,Wk,Wx
    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / np.power(param.size(-1), 0.5)
            param.data.uniform_(-stdv, stdv)

    def forward(self, x):
        # 得到初始化的QKV
        q = torch.matmul(x, self.wq)
        k = torch.matmul(x, self.wk)
        v = torch.matmul(x, self.wv)
        # 进行自注意力计算，使用多头
        attn, output = self.mha(q, k, v)
        # 使用单头
        # attn, output = self.ha(q, k, v)
        return attn, output


if __name__ == "__main__":
    n_x = 4
    d_x = 80
    batch = 64
    x = torch.randn(batch, n_x, d_x)

    selfattn = SelfAttention(n_head=8, d_k=128, d_v=64, d_x=80, d_o=80)
    attn, output = selfattn(x)

    print(attn.size())
    print(output.size())