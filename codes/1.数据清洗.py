import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import re
plt.rcParams['font.sans-serif'] = ['KaiTi']  #指定默认字体 SimHei黑体
plt.rcParams['axes.unicode_minus'] = False   #解决保存图像是负号'

# # 读取文件中所有数据
data_set = pd.read_excel()
data_blog = data_set[['comment']].values
data_blog = data_blog.tolist()  # 把数据存储格式转化成list

def textParse(str_doc):
    # 目的是将文本中特殊符号、标点等过滤
    str_doc = re.sub(r'[0-9]', '', str_doc)
    r1 = '[’!"#$%&\'()*+,-./:：;；|<=>?@，—。?★、…【】《》？“”‘’！[\\]^_`{|}~/~]+'
    str_doc = re.sub(r1, '', str_doc)
    return str_doc
# outlist=[]
for i in range(len(data_blog)):
    # 正则表达式对字符串清洗
    data_blog[i] = textParse(str(data_blog[i]))
#     # print(data_blog[i])

data_set['comment1'] = data_blog
data_set.to_excel(）