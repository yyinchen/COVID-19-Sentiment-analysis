import pandas as pd
import numpy as np
from gensim import corpora
from gensim.models import LdaSeqModel
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 加载数据
df = pd.read_excel(r'C:\Users\Administrator\Desktop\论文数据\官方2.xlsx', sheet_name='Sheet1')

# 2. 文本预处理
texts = [str(text).split() for text in df['text']]

# 3. 创建词典和语料库
dictionary = corpora.Dictionary(texts)
dictionary.filter_extremes(no_below=2, no_above=0.9)
corpus = [dictionary.doc2bow(text) for text in texts]


# 4. 时间分片
# time_labels = df['time'].astype(float).values
num_time_slices = [6079, 1967, 2570]
# time_bins = pd.cut(time_labels, bins=num_time_slices, labels=False)
# time_slice_counts = np.bincount(time_bins)

# 5. 训练DTM模型
num_topics = 6
ldaseq = LdaSeqModel(
    corpus=corpus,
    id2word=dictionary,
    time_slice=num_time_slices,
    num_topics=num_topics,
    passes=5,
    random_state=50
)

# 6. 打印主题词
for t in range(3):
    print(f"\n时间片 {t+1}:")
    for topic_id in range(num_topics):
        topic_words = ldaseq.print_topic(topic_id, time=t)
        print(f"  主题 {topic_id+1}: {topic_words[:10]}")

# ========== 2. 修复后：DTM主题热力图绘制（核心修改：适配print_topic方法） ==========
# 全局配置：Windows中文显示+高清图设置（无需修改）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
sns.set_style('whitegrid')

# 关键参数（可按需修改）
top_n_words = 10  # 每个主题提取TOP10关键词
heatmap_data = []  # 热力图数值矩阵
x_labels = []  # x轴：TOP关键词
y_labels = []  # y轴：时间片-主题

# 核心修复：从DTM的print_topic结果中解析「关键词-权重」（适配其字符串输出格式）
for t in range(4):
    for topic_id in range(num_topics):
        # 1. 提取DTM主题词字符串，按空格分割为「词+权重」列表
        topic_words_str = ldaseq.print_topic(topic_id, time=t)
        word_weight_pairs = topic_words_str.strip().split()  # 分割为[词1, 权重1, 词2, 权重2, ...]

        # 2. 解析为(词, 权重)元组，过滤无效值
        topic_word_weights = []
        for i in range(0, len(word_weight_pairs) - 1, 2):  # 步长2，依次取词和权重
            try:
                word = word_weight_pairs[i]
                weight = float(word_weight_pairs[i + 1])
                topic_word_weights.append((word, weight))
            except:
                continue  # 跳过解析失败的无效项

        # 3. 取TOPn关键词权重，构建热力图数据
        top_weights = [round(weight, 4) for _, weight in topic_word_weights[:top_n_words]]
        # 补全空值（若主题词不足10个，用0填充，保证矩阵维度一致）
        if len(top_weights) < top_n_words:
            top_weights += [0.0] * (top_n_words - len(top_weights))
        heatmap_data.append(top_weights)

        # 4. 构建坐标轴标签
        if t == 0:  # 仅首次遍历添加x轴关键词（避免重复）
            x_labels = [word for word, _ in topic_word_weights[:top_n_words]]
            # 补全x轴标签（与权重数匹配）
            if len(x_labels) < top_n_words:
                x_labels += [f"无词{len(x_labels) + 1}"] * (top_n_words - len(x_labels))
        y_labels.append(f"时间片{t + 1}-主题{topic_id + 1}")  # y轴：时间片-主题

# 转换为矩阵，适配seaborn热力图
heatmap_matrix = np.array(heatmap_data)

# ========== 3. 绘制热力图（样式优化，直接用于论文） ==========
fig, ax = plt.subplots(figsize=(top_n_words + 3, num_topics * num_time_slices + 2))
sns.heatmap(
    heatmap_matrix,
    cmap='Blues',  # 蓝色系：颜色越深，关键词权重越高
    annot=True,  # 显示权重数值，便于精准分析
    fmt='.4f',  # 数值保留4位小数
    cbar=True,
    cbar_kws={'label': '关键词权重', 'shrink': 0.8},  # 颜色条设置
    ax=ax,
    xticklabels=x_labels,
    yticklabels=y_labels
)

# 图表标注优化
ax.set_xlabel('主题TOP关键词', fontsize=12, fontweight='bold', labelpad=10)
ax.set_ylabel('时间片-主题', fontsize=12, fontweight='bold', labelpad=10)
ax.set_title(f'DTM动态主题建模-关键词权重热力图（各主题TOP{top_n_words}词）',
             fontsize=14, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right')  # 旋转x轴标签，避免重叠
plt.yticks(fontsize=10)
plt.tight_layout()  # 自动调整布局，避免标签裁剪

# 保存热力图（与原数据同路径，高清300dpi）
save_path = r'C:\Users\Administrator\Desktop\论文数据\DTM主题热力图_修复版.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"\n✅ 热力图绘制完成！已保存至：{save_path}")
print(f"✅ 热力图维度：{len(y_labels)}（时间片-主题）×{len(x_labels)}（TOP关键词）")