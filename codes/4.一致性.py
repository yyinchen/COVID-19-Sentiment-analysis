import gensim
import pyLDAvis.gensim as gensimvis
from gensim import corpora, models
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import warnings
import pyLDAvis.gensim
import re
import pandas as pd
import numpy as np
import jieba
import matplotlib.pyplot as plt
import jieba.posseg as jp, jieba
from snownlp import seg
from snownlp import SnowNLP
from snownlp import sentiment
from gensim.models import CoherenceModel
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

# 读取Excel文件
data = pd.read_excel(r'keywords只含中文.xlsx',names=['New_keywords'])

# # 创建语料库
texts = data['New_keywords'].apply(lambda x: x.split())  # 将评论文本分词
# print(texts)

id2word = corpora.Dictionary(texts)  # 创建词典
corpus = [id2word.doc2bow(text) for text in texts]  # 创建语料库

coherence_values = []
perplexity_values = []
model_list = []

if __name__ == '__main__':
    for topic in range(15):
        lda_model = gensim.models.LdaMulticore(corpus=corpus, num_topics=topic + 1, id2word = id2word, random_state=100,
                                           chunksize=100, passes=50, per_word_topics=True)

        model_list.append(lda_model)
        coherencemodel = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word, coherence='c_v')
        coherence_values.append(round(coherencemodel.get_coherence(), 3))

    x = range(1,16)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()




