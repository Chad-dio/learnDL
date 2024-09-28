import torch
from torch.nn.utils.rnn import pack_padded_sequence


class GRUClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        super(GRUClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1

        # 词嵌入层,将词语映射到hidden维度
        self.embedding = torch.nn.Embedding(input_size, hidden_size)
        # GRU层(输入为特征数，这里是embedding_size,其大小等于hidden_size))
        self.gru = torch.nn.GRU(hidden_size, hidden_size, num_layers=n_layers,
                                bidirectional=bidirectional)  # bidirectional双向神经网络
        # 线性层
        self.fc = torch.nn.Linear(hidden_size * self.n_directions, output_size)

    def _init_hidden(self, bath_size, USE_GPU):
        # 初始化权重，(n_layers * num_directions 双向, batch_size, hidden_size)
        hidden = torch.zeros(self.n_layers * self.n_directions, bath_size, self.hidden_size)
        return self.create_tensor(hidden, USE_GPU=USE_GPU)

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

    def create_tensor(self, tensor, USE_GPU):
        if USE_GPU:
            device = torch.device('cuda:0')
            tensor = tensor.to(device)
        return tensor
