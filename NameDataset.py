import os

from torch.utils.data import Dataset

class NameDataSet(Dataset):
    def __init__(self):
        self.labels = []
        self.inputs = []
        folder_path = "data/names/"
        for filename in os.listdir(folder_path):
            label = filename.replace(".txt", "")
            self.labels.append(label)
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = [line.strip() for line in file]
                self.inputs.append(content)

    def __getitem__(self, item):
        return self.inputs[item], self.labels[item]

    def __len__(self):
        return len(self.inputs)

    def getAll(self):
        for x, y in zip(self.inputs, self.labels):
            print(x, y, sep=" ")

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

ds = NameDataSet()

ds.getAll()