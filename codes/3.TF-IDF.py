from jieba import analyse
import pandas as pd
import xlrd
import jieba
import xlwt
from gensim import corpora, models
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pyLDAvis

data_set = pd.read_excel(r'.xlsx')
data_blog = data_set[['new-comment']].values

keywords = []
content_text = ""
for i in range(len(data_blog)):
    text = data_blog[i]
    tfidf =  analyse.extract_tags
    keyword = tfidf(str(text), topK=100, withWeight=False, allowPOS=())
    content_text = " ".join(keywords)
    strkeyword = ''
    for word in keyword:
        strkeyword = strkeyword + str(word) + ' '
    keywords.append(strkeyword)


data_set['new-comment1'] = keywords

data_set.to_excel(r'keywords.xlsx')
