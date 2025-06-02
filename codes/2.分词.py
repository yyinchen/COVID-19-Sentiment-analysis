import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import re
import jieba

plt.rcParams['font.sans-serif'] = ['KaiTi']  #指定默认字体 SimHei黑体
plt.rcParams['axes.unicode_minus'] = False   #解决保存图像是负号'

# 加载停用词表，目录就是停用词表所在的目录
stopwords = [line.strip() for line in open(r'\stopwords.txt', 'r', encoding='UTF-8')]
papers = pd.read_excel(r'C:\Users\Chen\Desktop\评审意见\论文评阅结果.xlsx').astype(str)

jieba.load_userdict(r'C:\Users\Chen\Desktop\评审意见\user.txt')
# 分词
def segment_text(text):
    words = jieba.cut(text)
    return [word for word in words if len(word) > 1]

papers['new-comment'] = papers['comment1'].apply(segment_text)

# 去除停用词
papers['new-comment'] =papers['new-comment'].apply(lambda words: [word for word in words if word not in stopwords and len(word) > 0])

#写入数据
papers.to_excel(r'.xlsx', index=False)