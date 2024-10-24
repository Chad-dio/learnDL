import numpy as np
import torch
from torch import nn

from attention.Attention import ScaledDotProductAttention


class MultiHeadAttentionByIID(nn.Module):
    def __init__(self, n_head, d_k_, d_v_, d_k, d_v, d_o):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        # 多个子空间上学习特征
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


class LinearFeatureLearner(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearFeatureLearner, self).__init__()
        # 定义一个线性层，将input_size 映射到 output_size
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        # 执行线性变换
        return self.linear(x)

enciid = MultiHeadAttention()
layer_input = LinearFeatureLearner(input_size, output_size)

def packg(input_data):
    output_new = layer_input(input_data)



if __name__ == "__main__":
    for epoch in range(num_epochs): # 512轮
        lstm_model.train()
        optimizer.zero_grad()

        # 线性变换后的参数塞进模型去
        output_sequence = gru_model(pkg_seq)

        # 参数复原
        output_w = output_sequence[:, :param_size * param_size].view(param_size, param_size)
        output_b = output_sequence[:, param_size * param_size:].view(param_size)
        """
        client1部分1: [数据1 1部分, 数据2 1部分, 数据3 1部分]
        client1部分2: [数据4 1部分, 数据5 1部分, 数据6 1部分]
        """
        losses = []
        for i in range(num_clients):
            original_w = client_params[i][0] #[]
            original_b = client_params[i][1]
            new_w = output_w[i]
            new_b = output_b[i]
            acc_diff_loss = accuracy_loss(original_w, original_b, new_w, new_b)
            losses.append(acc_diff_loss)

        # 平均一下，减少偏差漂移
        loss = torch.mean(torch.stack(losses))

        # 反向传播并优化
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

