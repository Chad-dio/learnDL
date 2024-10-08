import torch
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import pandas as pd

from SentenceDataSet import SentimentDataset

# 1.设置超参数
TRAIN_PATH = 'D:\python\learnDL\data\sentimentAnalysis\\train.tsv'
TEST_PATH = 'D:\python\learnDL\data\sentimentAnalysis\\test.tsv'
BATCH_SIZE = 256


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


# 数据预处理
def preprocess_data():
    # 读取数据集
    train_df = pd.read_csv(TRAIN_PATH, sep='\t')
    test_df = pd.read_csv(TEST_PATH, sep='\t')
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
test_indices = [phrase_to_indices(phrase, word2idx) for phrase in test_phrases]

# 移除长度为0的样本
train_indices = [x for x in train_indices if len(x) > 0]
train_sentiments = [y for x, y in zip(train_indices, train_sentiments) if len(x) > 0]
test_indices = [x for x in test_indices if len(x) > 0]


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

if __name__ == '__main__':
    pass
    # for i, (phrase, sentiment) in enumerate(train_loader, 1):
    #     if i > 10:
    #         break
    #     print(phrase)
    #     print(sentiment)
