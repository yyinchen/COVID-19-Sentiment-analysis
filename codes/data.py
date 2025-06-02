from functools import partial

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


# Make MyDataset
class MyDataset(Dataset):
    # self 可以在一个大类当中使用，引用其他语句
    def __init__(self, sentences, labels, method_name, model_name):
        self.sentences = sentences  # 读取文本
        self.labels = labels   # 读取labels行数据
        self.method_name = method_name  # 方法名称lstm
        self.model_name = model_name    # 模型名称bert
        dataset = list()  # 定义dataset 为list
        index = 0
        for data in sentences:
            tokens = str(data).split(' ')  # 用空格对一条文本进行分割
            labels_id = labels[index]
            index += 1
            dataset.append((tokens, labels_id))    # 将文字分割转换成数字
        self._dataset = dataset

    def __getitem__(self, index):
        return self._dataset[index]

    def __len__(self):
        return len(self.sentences)


# Make tokens for every batch
def my_collate(batch, tokenizer):
    tokens, label_ids = map(list, zip(*batch))   # x与y一一对应
    """
    # padding：给序列补全到一定长度，True or ‘longest’: 是补全到batch中的最长长度，max_length’:补到给定max-length或没给定时，补到模型能接受的最长长度。
    # return_tensors：返回数据的类型，可选’tf’，‘pt’， ‘np’ ，分别表示tf.constant, torch.Tensor或np.ndarray类型
    # is_split_into_words参数来告诉标记器,我们的输入序列已经被分割成单词:
    tokenizer会将tokens变成数字，作为输入到模型中。就是模型的字典。
    """
    text_ids = tokenizer(tokens,
                         # 当句子长度大于max_length时,截断
                         truncation=True,
                         # 一律补pad到max_length长度
                         padding=True,
                         max_length=150,
                         is_split_into_words=True,
                         add_special_tokens=True,
                         return_tensors='pt')
    return text_ids, torch.tensor(label_ids)


# Load dataset
def load_dataset(tokenizer, train_batch_size, test_batch_size, model_name, method_name, workers):
    data = pd.read_excel(r'C:\Users\Chen\Desktop\论文数据\新能源汽车\数据分析\car_sentiment2.xlsx', header=0)

    len1 = int(len(list(data['labels'])) * 1)
    labels = list(data['labels'])[0:len1]
    sentences = list(data['sentences'])[0:len1]
    # split train_set and test_set
    tr_sen, te_sen, tr_lab, te_lab = train_test_split(sentences, labels, train_size=31)

    # Dataset
    train_set = MyDataset(tr_sen, tr_lab, method_name, model_name)
    test_set = MyDataset(te_sen, te_lab, method_name, model_name)
    # DataLoader
    collate_fn = partial(my_collate, tokenizer=tokenizer)
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=workers,
                              collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=True, num_workers=workers,
                             collate_fn=collate_fn, pin_memory=True)
    return train_loader, test_loader