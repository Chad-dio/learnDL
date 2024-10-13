import numpy as np
import torch
from torch import nn


class ScaledDotProductAttention(nn.Module):
    # scale:缩放因子
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        # 1. q和k进行点积计算
        t = torch.matmul(q, k.transpose(1, 2))
        # 2.缩放
        t = t * self.scale
        # 3.softmax层
        attention = self.softmax(t)
        # 4.获取输出
        output = torch.bmm(attention, v)
        return attention, output


if __name__ == "__main__":
    n_q, n_k, n_v = 2, 4, 4
    d_q, d_k, d_v = 128, 128, 64
    batch = 64
    q = torch.randn(batch, n_q, d_q)
    k = torch.randn(batch, n_k, d_k)
    v = torch.randn(batch, n_v, d_v)

    attention = ScaledDotProductAttention(scale=np.power(d_k, 0.5))
    attn, output = attention(q, k, v)

    print(attn)
    print(output)