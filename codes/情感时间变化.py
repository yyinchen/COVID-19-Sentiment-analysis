import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#read excel file
excel_path = r'C:\Users\Chen\Desktop\论文数据\论文数据库\时间表.xlsx'
data = pd.read_excel(excel_path,sheet_name=1)


x = data['date']
y1 = data['all emotion']

plt.plot(x, y1)
plt.title('微博总体均值情感时间变化图')  # 折线图标题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字
plt.xlabel('date')  # x轴标题
plt.ylabel('all emotion')  # y轴标题


plt.show()  # 显示折线图

