import pyLDAvis
import pandas as pd
from gensim import corpora, models
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# 读取Excel文件
data = pd.read_excel(r'.xlsx',names=['new-comment']).astype(str)

# # 创建语料库
texts = data['new-comment'].apply(lambda x: x.split())  # 将评论文本分词
# print(texts)

dictionary = corpora.Dictionary(texts)  # 创建词典
corpus = [dictionary.doc2bow(text) for text in texts]  # 创建语料库


lda_model = models.LdaModel(corpus, num_topics=3, id2word=dictionary,eval_every=None, passes=40, iterations=100)


# 创建DataFrame来存储结果
topic_word_freq_data = []
for i in range(3):  # 假设有4个主题
    topic_words = lda_model.show_topic(i, topn= 8)
    topic_word_freq = [(word, freq) for word, freq in topic_words]
    for word, freq in topic_word_freq:
        topic_word_freq_data.append([f"Topic {i + 1}", word, freq])

df = pd.DataFrame(topic_word_freq_data, columns=["Topic", "Word", "Frequency"])


# 指定存储位置并将DataFrame写入Excel文件
file_path = r'topic.xlsx'  # 你想要存储的位置
df.to_excel(file_path, index=False)