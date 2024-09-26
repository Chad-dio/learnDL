from NameDataset import NameDataSet
from torch.utils.data import DataLoader

# 1.设置超参数
HIDDEN_SIZE = 100
BATCH_SIZE = 256
N_LAYER = 2  # RNN的层数
N_EPOCHS = 100
N_CHARS = 128  # ASCII码的个数
USE_GPU = False

# 2.数据集相关
train_set = NameDataSet()
train_loader = DataLoader(train_set, shuffle=True, batch_size=BATCH_SIZE, num_workers=2)
N_COUNTRY = train_set.get_countries_num()
