import re

import pandas as pd
import jieba
from opencc import OpenCC

data = pd.read_csv("train.tsv", sep="\t")
print(data.head(10))

data['word_count'] = data['text_a'].apply(lambda x: len(jieba.lcut(x)))
print(data[['text_a', 'word_count']].head())

data['char_count'] = data['text_a'].str.len()
print(data[['text_a', 'char_count']].head())


def avg_word(review):
    words = jieba.lcut(review)
    return sum(len(word) for word in words) / len(words)


data['avg_word'] = data['text_a'].apply(lambda x: avg_word(x))
print(data[['text_a', 'avg_word']].head())

# 转换繁体为简体
cc = OpenCC('t2s')
data['text_a'] = data['text_a'].apply(lambda x: cc.convert(x))
print(data['text_a'].head())

# 去除标点符号
data['text_a'] = data['text_a'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

print(data['text_a'].head())

# 去除停用词
# 1.读入停用词集合
with open("chinese_stopwords.txt", "r", encoding="utf-8") as f:
    stopwords = set(f.read().splitlines())
# 2.使用jieba分词并过滤停用词
data['text_a'] = data['text_a'].apply(
    lambda x: " ".join(word for word in jieba.lcut(x) if word not in stopwords)
)

print(data['text_a'].head())

all_words = pd.Series([word
                       for text in data['text_a']
                       for word in jieba.lcut(text)])
low_freq = all_words.value_counts()[-10:]
print(low_freq)
