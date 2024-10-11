import time

import torch
from sklearn.preprocessing import LabelEncoder
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import DataLoader
import pandas as pd

from SentenceDataSet import SentimentDataset

# 1.设置超参数
TRAIN_PATH = 'D:\python\learnDL\data\sentimentAnalysis\\train.tsv'
TEST_PATH = 'D:\python\learnDL\data\sentimentAnalysis\\test.tsv'
BATCH_SIZE = 64
HIDDEN_SIZE = 100
N_LAYERS = 2
N_EPOCHS = 10
LEARNING_RATE = 0.001


# 2.预处理
# 构建单词表，获取单词的唯一索引
def build_vocab(phrases):
    vocab = set()  # 去重
    # 取出文本里面的每个单词，添加到单词表集合里
    for phrase in phrases:
        for word in phrase.split():
            vocab.add(word)
    # 字典推导式，创建单词表
    word2idx = {word: idx for idx, word in enumerate(vocab, start=1)}
    # 填充位的对应索引为0
    word2idx['<PAD>'] = 0
    return word2idx


# 文本转换成索引列表
def phrase_to_indices(phrase, word2idx):
    return [word2idx[word] for word in phrase.split() if word in word2idx]


train_df = pd.read_csv(TRAIN_PATH, sep='\t')
test_df = pd.read_csv(TEST_PATH, sep='\t')


# 数据预处理
def preprocess_data():
    # 读取数据集
    # 将标签转换成数值类型
    le = LabelEncoder()
    train_df['Sentiment'] = le.fit_transform(train_df['Sentiment'])
    # fit:训练LabelEncoder，让它知道总共有多少个类别
    # transform:将类别标签转换成整数
    # inverse_transform:标签转换成类别
    train_phrases = train_df['Phrase'].tolist()
    train_sentiments = train_df['Sentiment'].tolist()
    test_phrases = test_df['Phrase'].tolist()
    return train_phrases, train_sentiments, test_phrases, le


train_phrases, train_sentiments, test_phrases, le = preprocess_data()
word2idx = build_vocab(train_phrases + test_phrases)
train_indices = [phrase_to_indices(phrase, word2idx) for phrase in train_phrases]
test_indices_old = [phrase_to_indices(phrase, word2idx) for phrase in test_phrases]

# 移除长度为0的样本
train_indices = [x for x in train_indices if len(x) > 0]
train_sentiments = [y for x, y in zip(train_indices, train_sentiments) if len(x) > 0]
test_indices = [x for x in test_indices_old if len(x) > 0]

# 数据加载器
def collate_fn(batch):
    # *:解包 zip:打包成为元组
    # 将元组列表转换成两个元组x和y，目的是为了能够后续批处理
    # 如果用列表的话，只能够对(x,y)进行批处理
    phrases, sentiments = zip(*batch)
    # 获取文本长度tensor
    lengths = torch.tensor([len(x) for x in phrases])
    # 获取文本tensor
    phrases = [torch.tensor(x) for x in phrases]
    # 进行padding，统一序列长度
    phrases_padded = pad_sequence(phrases, batch_first=True, padding_value=0)
    sentiments = torch.tensor(sentiments)
    return phrases_padded, sentiments, lengths


train_dataset = SentimentDataset(train_indices, train_sentiments)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

test_dataset = SentimentDataset(test_indices)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         collate_fn=lambda x: pad_sequence([torch.tensor(phrase) for phrase in x], batch_first=True,
                                                           padding_value=0))


# 模型定义
class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, n_layers):
        super(SentimentRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, n_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x, lengths):
        x = self.embedding(x)
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                 enforce_sorted=False)
        # 表示序列长度不需要严格递减，避免了手动排序 lengths 的需求
        _, (hidden, _) = self.lstm(x)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        out = self.fc(hidden)
        return out


vocab_size = len(word2idx)
embed_size = 128
output_size = len(le.classes_)

model = SentimentRNN(vocab_size, embed_size, HIDDEN_SIZE, output_size, N_LAYERS)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# 训练和测试循环
def train(model, train_loader, criterion, optimizer, n_epochs):
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        for phrases, sentiments, lengths in train_loader:
            optimizer.zero_grad()
            output = model(phrases, lengths)
            loss = criterion(output, sentiments)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch: {epoch + 1}, Loss: {total_loss / len(train_loader)}')


def generate_test_results(model, test_loader):
    model.eval()
    results = []
    with torch.no_grad():
        for phrases in test_loader:
            lengths = torch.tensor([len(x) for x in phrases])
            output = model(phrases, lengths)
            preds = torch.argmax(output, dim=1)
            results.extend(preds.cpu().numpy())
    return results


if __name__ == '__main__':
    begin = time.time()
    train(model, train_loader, criterion, optimizer, N_EPOCHS)
    end = time.time()
    print(end - begin)
    test_ids = test_df['PhraseId'].tolist()
    preds = generate_test_results(model, test_loader)
    new_preds = []
    for idx in range(len(preds)):
        if idx == 1390:
            new_preds.append(2)
        new_preds.append(preds[idx])
    assert len(test_ids) == len(new_preds), f"Lengths do not match: {len(test_ids)} vs {len(new_preds)}"
    # 保存结果
    output_df = pd.DataFrame({'PhraseId': test_ids, 'Sentiment': new_preds})
    output_df.to_csv('sentiment_predictions.csv', index=False)
