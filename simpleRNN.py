import math
import time

import numpy as np
from matplotlib import pyplot as plt
from torch import optim

from NameDataset import NameDataSet
from torch.utils.data import DataLoader
import torch
from torch.nn.utils.rnn import pack_padded_sequence

# 1.设置超参数
HIDDEN_SIZE = 100
BATCH_SIZE = 256
N_LAYER = 2  # RNN的层数
N_EPOCHS = 100
N_CHARS = 400  # ASCII码的个数
USE_GPU = False

# 2.数据集相关
train_set = NameDataSet(True)
train_loader = DataLoader(train_set, shuffle=True, batch_size=BATCH_SIZE, num_workers=2)
test_set = NameDataSet(False)
test_loder = DataLoader(test_set, shuffle=False, batch_size=BATCH_SIZE, num_workers=2)
N_COUNTRY = train_set.get_countries_num()


# 工具类函数
# 把名字转换成ASCII码
# 返回ASCII码值列表和名字的长度
def name2list(name):
    arr = [ord(c) for c in name]
    return arr, len(arr)


# 是否把数据放到GPU上
def create_tensor(tensor):
    if USE_GPU:
        device = torch.device('cuda:0')
        tensor = tensor.to(device)
    return tensor


def timesince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)  # math.floor()向下取整
    s -= m * 60
    return '%dmin %ds' % (m, s)  # 多少分钟多少秒


class GRUClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        super(GRUClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1

        # 词嵌入层,将词语映射到hidden维度
        # 特征处理，将稀疏高维向量变成连续的稠密表示作为隐藏层的输入
        self.embedding = torch.nn.Embedding(input_size, hidden_size)
        # GRU层(输入为特征数，这里是embedding_size,其大小等于hidden_size))
        self.gru = torch.nn.GRU(hidden_size, hidden_size, num_layers=n_layers,
                                bidirectional=bidirectional)
        # 线性层，输出结果
        self.fc = torch.nn.Linear(hidden_size * self.n_directions, output_size)

    def _init_hidden(self, bath_size):
        # 初始化权重，(n_layers * num_directions 双向, batch_size, hidden_size)
        hidden = torch.zeros(self.n_layers * self.n_directions, bath_size, self.hidden_size)
        return create_tensor(hidden)

    def forward(self, input, seq_lengths):
        # 转置 B X S -> S X B
        input = input.t()  # 此时的维度为seq_len, batch_size
        batch_size = input.size(1)
        hidden = self._init_hidden(batch_size)

        # 嵌入层处理 input:(seq_len,batch_size) -> embedding:(seq_len,batch_size,embedding_size)
        embedding = self.embedding(input)

        # 进行打包（不考虑0元素，提高运行速度）需要将嵌入数据按长度排好
        gru_input = pack_padded_sequence(embedding, seq_lengths)

        # output:(*, hidden_size * num_directions)，*表示输入的形状(seq_len,batch_size)
        # hidden:(num_layers * num_directions, batch, hidden_size)
        output, hidden = self.gru(gru_input, hidden)
        if self.n_directions == 2:
            hidden_cat = torch.cat([hidden[-1], hidden[-2]],
                                   dim=1)  # hidden[-1]的形状是(1,256,100)，hidden[-2]的形状是(1,256,100)，拼接后的形状是(1,256,200)
        else:
            hidden_cat = hidden[-1]  # (1,256,100)
        fc_output = self.fc(hidden_cat)
        return fc_output


def make_tensors(names, countries):
    # eg. name2list(name):chad -> ([ascall], 4)
    name_len_list = [name2list(name) for name in names]
    # 拿到每个名字拆分的ascall列表，即[ascall]
    name_seq = [sl[0] for sl in name_len_list]
    # 拿到每个名字的长度，即时间步
    seq_lengths = torch.LongTensor([sl[1] for sl in name_len_list])
    # 消除floatTensor带来的精度影响
    countries = countries.long()
    # 创建全0的tensor，用于待会将names得到的ascall列表填充进去
    # 维度是len(name_seq) * seq_lengths.max(),
    # 已经进行了padding,将所有的name都对齐了
    seq_tensor = torch.zeros(len(name_seq), seq_lengths.max().item()).long()
    # 进行填充
    for idx, (seq, seq_len) in enumerate(zip(name_seq, seq_lengths)):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)
    '''
    [[0, 0, 0, 0, 0],   # 第1个名字，"John"
     [0, 0, 0, 0, 0],   # 第2个名字，"Alice"
     [0, 0, 0, 0, 0]]   # 第3个名字，"Bob"
     [[74, 111, 104, 110, 0],   # John -> [74, 111, 104, 110]，长度 4，剩余部分补零
     [65, 108, 105, 99, 101],  # Alice -> [65, 108, 105, 99, 101]，长度 5
     [66, 111, 98, 0, 0]]      # Bob -> [66, 111, 98]，长度 3，剩余部分补零
    '''
    # 在行的方向进行降序，即对每行进行降序
    # perm_idx：排序后的索引，用来记录原始数据的顺序变化。
    seq_lengths, perm_idx = seq_lengths.sort(dim=0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    countries = countries[perm_idx]
    return create_tensor(seq_tensor), create_tensor(seq_lengths), create_tensor(countries)


# 训练循环
def train(epoch, start):
    total_loss = 0
    for i, (names, countries) in enumerate(train_loader, 1):
        # 将数据进行处理
        inputs, seq_lengths, target = make_tensors(names, countries)
        # 前向传播，获取输出
        output = model(inputs, seq_lengths)
        # 计算损失
        loss = criterion(output, target)
        # 后向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 统计损失
        total_loss += loss.item()
        if i % 10 == 0:
            print(f'[{timesince(start)}] Epoch {epoch} ', end='')
            print(f'[{i * len(inputs)}/{len(train_set)}] ', end='')
            print(f'loss={total_loss / (i * len(inputs))}')  # 打印每个样本的平均损失

    return total_loss


# 测试循环
def t():
    correct = 0
    total = len(test_set)
    print('evaluating trained model ...')
    # print("len = " + total.__str__())
    with torch.no_grad():
        for i, (names, countries) in enumerate(test_loder, 1):
            inputs, seq_lengths, target = make_tensors(names, countries)
            output = model(inputs, seq_lengths)
            pred = output.max(dim=1, keepdim=True)[1]
            # 返回每一行中最大值的那个元素的索引，且keepdim=True，表示保持输出的二维特性
            correct += pred.eq(target.view_as(pred)).sum().item()  # 计算正确的个数
        percent = '%.2f' % (100 * correct / total)
        print(f'Test set: Accuracy {correct}/{total} {percent}%')
    return correct / total  # 返回的是准确率，0.几几的格式，用来画图


if __name__ == '__main__':
    model = GRUClassifier(N_CHARS, HIDDEN_SIZE, N_COUNTRY, N_LAYER)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = 'cuda:0' if USE_GPU else 'cpu'
    model.to(device)
    start = time.time()
    print('Training for %d epochs...' % N_EPOCHS)
    acc_list = []
    # 在每个epoch中，训练完一次就测试一次
    for epoch in range(1, N_EPOCHS + 1):
        # Train cycle
        train(epoch, start)
        acc = t()
        acc_list.append(acc)

    # 绘制在测试集上的准确率
    epoch = np.arange(1, len(acc_list) + 1)
    acc_list = np.array(acc_list)
    plt.plot(epoch, acc_list)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()
