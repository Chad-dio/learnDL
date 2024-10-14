import numpy as np
import torch
from torch import nn

from attention.Attention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    """
    n_head:分成几个头部
    d_k_:输入的Q和K的维度
    d_v_:输入的V的维度
    d_k:变换后Q和K的维度
    d_v:变换后V的维度
    d_o:输出维度
    """
    def __init__(self, n_head, d_k_, d_v_, d_k, d_v, d_o):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        # 将原始的维度进行多头变换，即映射到多个子空间上
        self.fc_q = nn.Linear(d_k_, n_head * d_k)
        self.fc_k = nn.Linear(d_k_, n_head * d_k)
        self.fc_v = nn.Linear(d_v_, n_head * d_v)
        # 每个头单独进行注意力计算
        self.attention = ScaledDotProductAttention(scale=np.power(d_k, 0.5))
        # 输出层，将多头注意力拼接起来
        self.fc_o = nn.Linear(n_head * d_v, d_o)

    def forward(self, q, k, v):

        n_head, d_q, d_k, d_v = self.n_head, self.d_k, self.d_k, self.d_v
        #获取参数
        batch, n_q, d_q_ = q.size()
        batch, n_k, d_k_ = k.size()
        batch, n_v, d_v_ = v.size()
        # 1.扩展成多头
        q = self.fc_q(q)
        k = self.fc_k(k)
        v = self.fc_v(v)
        """
        q:[batch, n_q, n_head * d_q]
        view:重新塑性 q的维度=>[batch, n_q, n_head, d_q]
        permute:置换维度，即调整张量顺序，将原始的维度移到目标位置上去[n_head, batch, n_q, d_q]
        contiguous():由于permute操作可能会导致张量在内存中的存储不连续，
                     使用.contiguous()确保张量在内存中连续存储
        view(-1, n_q, d_q):[n_head, batch, n_q, d_q] => [n_head * batch, n_q, d_q]
        最原始数据：假设batch = 2. n_head = 4
         Q:[数据1， 数据2， 数据3]
           [数据4， 数据5， 数据6]
         变换后的：
         头1批次1: [数据1头1部分, 数据2头1部分, 数据3头1部分]
         头1批次2: [数据4头1部分, 数据5头1部分, 数据6头1部分]
         ...
         头4批次1: [数据1头4部分, 数据2头4部分, 数据3头4部分]
         头4批次2: [数据4头4部分, 数据5头4部分, 数据6头4部分]
        """
        q = q.view(batch, n_q, n_head, d_q).permute(2, 0, 1, 3).contiguous().view(-1, n_q, d_q)
        k = k.view(batch, n_k, n_head, d_k).permute(2, 0, 1, 3).contiguous().view(-1, n_k, d_k)
        v = v.view(batch, n_v, n_head, d_v).permute(2, 0, 1, 3).contiguous().view(-1, n_v, d_v)

        # 2.当成单头注意力求输出
        attn, output = self.attention(q, k, v)
        # 3.拼接多头的输出
        """
        output:[n_head * batch, n_q, d_q]
        view(n_head, batch, n_q, d_v):[n_head, batch, n_q, d_q]
        permute(1, 2, 0, 3):[batch, n_q, n_head, d_q]
        view(batch, n_q, -1):[batch, n_q, n_head * d_v]
        作用：将多头的输出拼接起来
        """
        output = output.view(n_head, batch, n_q, d_v).permute(1, 2, 0, 3).contiguous().view(batch, n_q, -1)
        # 4.仿射变换得到最终输出
        output = self.fc_o(output)

        return attn, output


if __name__ == "__main__":
    n_q, n_k, n_v = 2, 4, 4
    d_q_, d_k_, d_v_ = 128, 128, 64
    batch = 5
    q = torch.randn(batch, n_q, d_q_)
    k = torch.randn(batch, n_k, d_k_)
    v = torch.randn(batch, n_v, d_v_)
    mask = torch.zeros(batch, n_q, n_k).bool()

    mha = MultiHeadAttention(n_head=8, d_k_=128, d_v_=64, d_k=256, d_v=128, d_o=128)
    attn, output = mha(q, k, v, mask=mask)

    print(attn.size())
    print(output.size())