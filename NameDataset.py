import os

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class NameDataSet(Dataset):
    def __init__(self, train=True, test_size=0.2, random_state=42):
        self.data = []
        self.labels = []
        self.inputs = []
        folder_path = "data/names/"
        for filename in os.listdir(folder_path):
            label = filename.replace(".txt", "")
            self.labels.append(label)
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = []
                for line in file:
                    line = line.strip()
                    content.append(line)
                    self.data.append((line, label))
                self.inputs.append(content)
        # 使用train_test_split来划分数据集
        train_data, test_data = train_test_split(self.data, test_size=test_size, random_state=random_state)
        # 根据train参数选择使用训练集或测试集
        self.data = train_data if train else test_data

        self.country = self.get_countries_dict()

    def __getitem__(self, item):
        x, y = self.data[item]
        return x, self.country[y]

    def __len__(self):
        return len(self.data)

    def getAll(self):
        for t in self.data:
            print(t)

    def get_countries_dict(self):
        # 根据国家名对应序号
        country_dict = dict()
        for idx, country_name in enumerate(self.labels):
            country_dict[country_name] = idx
        return country_dict

    def idx2country(self, index):
        # 根据索引返回国家名字
        return self.labels[index]

    def get_countries_num(self):
        # 返回国家名个数（分类的总个数）
        return len(self.labels)
