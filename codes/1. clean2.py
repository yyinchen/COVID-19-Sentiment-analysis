import pandas as pd
import numpy as np
import ast
import re
import jieba
from collections import Counter

# 读取数据
df = pd.read_excel("/mnt/dataset1_processed_个人.xlsx")
print(f"原始数据规模：{len(df)}条文本")
print(f"数据列名：{list(df.columns)}")
print("\n前3条文本预览：")
for i in range(3):
    print(f"\n第{i+1}条：")
    print(f"原始文本：{df.iloc[i]['blog'][:100]}...")
    print(f"已分词文本：{df.iloc[i]['processed_words'][:50]}...")

# -------------------------- 核心：快递员相关关键词库构建 --------------------------
# 基于快递员职业场景分类的关键词（覆盖工作内容、工具、场景、问题等）
courier_keywords = {
    # 1. 核心职业行为
    "action": ["派件", "收件", "配送", "送件", "取件", "投递", "签收", "派送", "揽件", "交接"],
    # 2. 职业身份标识
    "identity": ["快递员", "速递员", "配送员", "快递小哥", "快递师傅", "快递员师傅", "信使", "邮差"],
    # 3. 工作载体/工具
    "tool": ["快递车", "三轮车", "电瓶车", "快递箱", "扫描枪", "巴枪", "快递袋", "快递盒", "包裹", "快件"],
    # 4. 工作场景/机构
    "scene": ["快递站", "站点", "网点", "快递点", "驿站", "菜鸟驿站", "快递柜", "丰巢", "快递中心", "分拨中心"],
    # 5. 工作问题/需求
    "problem": ["超时", "延误", "丢失", "破损", "投诉", "差评", "罚款", "加班", "工作量", "工资", "待遇"],
    # 6. 关联主体
    "related": ["客户", "收件人", "寄件人", "商家", "快递公司", "顺丰", "中通", "圆通", "申通", "韵达", "京东物流"]
}

# 合并为总关键词列表（用于快速匹配）
all_courier_keywords = [word for sublist in courier_keywords.values() for word in sublist]
print(f"\n构建快递员相关关键词共 {len(all_courier_keywords)} 个")


# -------------------------- 1. 文本过滤函数（核心逻辑） --------------------------
def filter_courier_related(text_words, raw_text, keyword_list, min_match=1):
    """
    过滤快递员相关文本
    :param text_words: 已分词的文本列表
    :param raw_text: 原始文本（用于二次语义校验）
    :param keyword_list: 快递员相关关键词库
    :param min_match: 最少匹配关键词数量（阈值）
    :return: (是否保留, 匹配的关键词, 优化后的分词列表)
    """
    # 步骤1：关键词匹配（统计匹配数量）
    matched_keywords = [word for word in text_words if word in keyword_list]

    # 步骤2：排除歧义文本（如“东风快递”属于军事术语，需剔除）
    ambiguous_patterns = ["东风快递", "快递单号查询", "快递费", "快递价格"]  # 非快递员相关的歧义关键词
    has_ambiguity = any(pattern in raw_text for pattern in ambiguous_patterns)

    # 步骤3：判断是否保留（满足匹配数量且无歧义）
    if len(matched_keywords) >= min_match and not has_ambiguity:
        # 优化分词：保留与快递员相关的核心词，剔除无关词
        irrelevant_words = ["我爸", "肉类", "鱼尾", "老人", "孩子", "偷走", "摔跤", "石墩子"]  # 基于数据样本总结的无关词
        optimized_words = [word for word in text_words if word not in irrelevant_words or word in keyword_list]
        return True, matched_keywords, optimized_words
    return False, [], []


# -------------------------- 2. 执行过滤与预处理 --------------------------
# 先处理processed_words列（字符串转列表）
def str_to_word_list(text_str):
    try:
        return ast.literal_eval(text_str)  # 解析字符串格式的列表
    except:
        return jieba.lcut(text_str)  # 异常时重新分词


df["word_list"] = df["processed_words"].apply(str_to_word_list)

# 批量过滤文本
filter_results = df.apply(
    lambda x: filter_courier_related(
        text_words=x["word_list"],
        raw_text=x["blog"],
        keyword_list=all_courier_keywords,
        min_match=1  # 至少匹配1个快递员相关关键词
    ),
    axis=1
)

# 提取过滤结果到DataFrame
df["is_courier_related"] = [res[0] for res in filter_results]
df["matched_keywords"] = [res[1] for res in filter_results]
df["optimized_words"] = [res[2] for res in filter_results]

# -------------------------- 3. 过滤结果统计 --------------------------
# 筛选出快递员相关文本
courier_df = df[df["is_courier_related"]].copy()
print(f"过滤后结果：")
print(f"- 原始文本总数：{len(df)} 条")
print(f"- 快递员相关文本数：{len(courier_df)} 条")
print(f"- 保留比例：{len(courier_df) / len(df) * 100:.2f}%")

# 统计匹配关键词Top10（验证过滤准确性）
all_matched_words = [word for words in courier_df["matched_keywords"] for word in words]
top_keywords = Counter(all_matched_words).most_common(10)
print(f"\n匹配频率Top10的快递员相关关键词：")
for word, count in top_keywords:
    print(f"  {word}: {count} 次")

# 预览过滤后的文本示例
print(f"\n过滤后的快递员相关文本示例（3条）：")
for i, (idx, row) in enumerate(courier_df.head(3).iterrows()):
    print(f"\n第{i + 1}条：")
    print(f"原始文本：{row['blog'][:80]}...")
    print(f"优化后分词：{row['optimized_words']}")
    print(f"匹配关键词：{row['matched_keywords']}")
