import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        初始化位置编码模块。
        :param d_model: 嵌入的维度
        :param max_len: 最大序列长度
        """
        super(PositionalEncoding, self).__init__()
        # 创建一个足够长的位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        # 0 到 maxLen - 1 的 张量
        # unsqueeze(1): [maxLen,] => [maxLen, 1]，即[0 到 maxLen] => [[0 到 maxLen]]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 除数张量, [,maxLen]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 奇数和偶数下标的分别处理
        pe[:, 0::2] = torch.sin(position * div_term) # 因为position是[maxLen, 1]，所以会有广播机制
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加一个维度，将位置编码设置为不可训练
        pe = pe.unsqueeze(0).detach()

        # 注册缓冲区，这样pe不会在训练过程中被视为模型的可训练参数
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        将位置编码添加到输入嵌入中。
        :param x: 输入嵌入，形状为 (Batch size, Sequence length, d_model)
        """
        # x的形状是 [Batch size, Sequence length, d_model]
        # 从缓冲区中取出相应长度的pe，并添加到x上
        x = x + self.pe[:, :x.size(1)]
        return x
