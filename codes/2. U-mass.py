import tomotopy as tp
import pandas as pd
import numpy as np
import nltk
import re
import inspect
import os
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from gensim.models import CoherenceModel
import gensim.corpora as corpora

# 设置中文字体（适配Windows/macOS/Linux）
plt.rcParams['font.sans-serif'] = ['SimHei' if os.name == 'nt' else 'WenQuanYi Zen Hei']  # Windows用黑体，Linux用文泉驿
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 下载必要的NLTK停用词数据
nltk.download('stopwords', quiet=True)


# ---------------------- 1. 数据加载 ----------------------
def load_preprocessed_data(excel_path):
    """加载已预处理的Excel数据，确保时间ID范围严格为 0 ~ T-1"""
    try:
        df = pd.read_excel(excel_path, engine='openpyxl')
        print(f"✅ 成功读取文件：{excel_path}")
        print(f"📊 原始数据：{len(df)} 行 | 列名：{list(df.columns)}")
    except Exception as e:
        print(f"❌ 读取文件失败：{str(e)}")
        raise

    # 自动检测文本列和时间列
    possible_text_cols = [col for col in df.columns if 'token' in col.lower() or 'text' in col.lower()]
    possible_time_cols = [col for col in df.columns if
                          'time' in col.lower() or 'year' in col.lower() or 'date' in col.lower()]

    if not possible_text_cols:
        raise ValueError("❌ 未检测到预处理文本列（列名建议含'token'/'text'）")
    if not possible_time_cols:
        raise ValueError("❌ 未检测到时间列（列名建议含'time'/'year'/'date'）")

    text_col = possible_text_cols[0]
    time_col = possible_time_cols[0]
    print(f"\n🔍 自动匹配列：")
    print(f"  - 预处理文本列：{text_col}")
    print(f"  - 时间列：{time_col}")

    # 文本解析函数
    def parse_tokens(token_input):
        if pd.isna(token_input):
            return []
        if isinstance(token_input, list):
            return [str(t).strip() for t in token_input if str(t).strip()]
        token_str = str(token_input).strip()
        if token_str.startswith('[') and token_str.endswith(']'):
            clean_str = re.sub(r'[\[\]\'"]', '', token_str)
            return [t.strip() for t in clean_str.split(',') if t.strip()]
        elif ' ' in token_str:
            return [t.strip() for t in token_str.split() if t.strip()]
        return []

    # 过滤空文本
    df['processed_tokens'] = df[text_col].apply(parse_tokens)
    df_filtered = df[df['processed_tokens'].map(len) > 0].reset_index(drop=True)

    if len(df_filtered) == 0:
        raise ValueError("❌ 过滤后无有效文本数据！")

    print(f"\n✅ 文本过滤完成：有效文本行数 = {len(df_filtered)}")

    # 时间ID编码（严格保证 0 ~ T-1）
    time_doc_count = df_filtered[time_col].value_counts().sort_index()
    valid_time_vals = [time_val for time_val, count in time_doc_count.items() if count > 0]
    df_filtered = df_filtered[df_filtered[time_col].isin(valid_time_vals)].reset_index(drop=True)

    valid_time_vals_sorted = sorted(valid_time_vals)
    T = len(valid_time_vals_sorted)
    time_ids = {time_val: idx for idx, time_val in enumerate(valid_time_vals_sorted)}
    df_filtered['time_id'] = df_filtered[time_col].map(time_ids)

    # 校验时间ID范围
    max_time_id = df_filtered['time_id'].max()
    if max_time_id >= T:
        df_filtered = df_filtered[df_filtered['time_id'] < T].reset_index(drop=True)
        max_time_id = df_filtered['time_id'].max()
        print(f"⚠️ 修正超出范围的时间ID！原最大ID={max_time_id + 1} → 修正后={max_time_id}")

    print(f"\n✅ 时间编码完成：")
    print(f"  - 模型时间片数量 T = {T}")
    print(f"  - 合法时间ID范围 = 0 ~ {max_time_id}（{max_time_id} < {T} ✔️）")

    return df_filtered, time_ids, T, text_col, time_col


# ---------------------- 2. 参数适配函数 ----------------------
def get_add_doc_time_param():
    """自动检测add_doc的时间参数名"""
    sig = inspect.signature(tp.DTModel.add_doc)
    params = list(sig.parameters.keys())
    if 'timepoint' in params:
        return 'timepoint'
    elif 'time' in params:
        return 'time'
    elif 't' in params:
        return 't'
    elif 'tid' in params:
        return 'tid'
    else:
        raise ValueError(f"❌ 不支持的add_doc参数列表：{params}")


# ---------------------- 3. 模型训练+评估 ----------------------
def evaluate_preprocessed_dtm(df, time_ids, T, k=5):
    """训练DTM模型，返回困惑度和U-mass一致性（修复nan问题）"""
    # 构建词典和语料库
    texts = df['processed_tokens'].tolist()
    gensim_dict = corpora.Dictionary(texts)
    # 核心修改：降低过滤阈值，保留更多词汇
    gensim_dict.filter_extremes(no_below=1, no_above=0.9)
    gensim_corpus = [gensim_dict.doc2bow(text) for text in texts]

    # 打印词典信息，排查问题
    print(f"\n📖 词典信息：总词数 = {len(gensim_dict)} | 语料库文档数 = {len(gensim_corpus)}")
    if len(gensim_dict) < 2:
        print(f"⚠️  词典词汇过少，可能导致U-mass计算失败！")

    # 初始化模型
    max_timepoint = df['time_id'].max() if len(df) > 0 else -1
    model_T = max(T, max_timepoint + 1)
    try:
        dtm_model = tp.DTModel(k=k, t=model_T, seed=42)
    except TypeError:
        dtm_model = tp.DTModel(k=k, seed=42)

    # 添加文档
    time_param = get_add_doc_time_param()
    added_docs = 0
    for idx, row in df.iterrows():
        timepoint = row['time_id']
        if timepoint < model_T:
            doc_kwargs = {'words': row['processed_tokens'], time_param: timepoint}
            dtm_model.add_doc(**doc_kwargs)
            added_docs += 1

    if added_docs == 0:
        raise ValueError("❌ 无合法文档可添加！")

    # 优化训练：增加迭代次数
    total_iter = 1000
    batch_iter = 150
    print(f"\n📈 模型训练中（总迭代{total_iter}次）...")
    for i in range(total_iter // batch_iter):
        dtm_model.train(batch_iter)
        if (i + 1) % 5 == 0:
            print(f"  进度：{(i + 1) * batch_iter}/{total_iter} | 困惑度：{dtm_model.perplexity:.4f}")

    # 计算困惑度
    perplexity = dtm_model.perplexity

    # 计算U-mass一致性（新增校验+兜底）
    topics = []
    valid_timepoints = [tp for tp in df['time_id'].unique() if tp < model_T]
    for topic_id in range(k):
        top_words = []
        for timepoint in valid_timepoints:
            try:
                words = dtm_model.get_topic_words(topic_id, timepoint=timepoint, top_n=10)
            except TypeError:
                try:
                    words = dtm_model.get_topic_words(topic_id, time=timepoint, top_n=10)
                except TypeError:
                    try:
                        words = dtm_model.get_topic_words(topic_id, t=timepoint, top_n=10)
                    except TypeError:
                        words = dtm_model.get_topic_words(topic_id, top_n=10)
            top_words.extend([word for word, prob in words])
        top_words = list(dict.fromkeys(top_words))[:10]
        print(f"📌 主题{topic_id}的top词：{top_words}")  # 打印主题词排查

        topic_word_ids = [gensim_dict.token2id[w] for w in top_words if w in gensim_dict.token2id]
        if len(topic_word_ids) > 0:
            topics.append(topic_word_ids)

    # 兜底逻辑：避免nan
    if len(topics) == 0:
        print(f"⚠️  无有效主题词，U-mass返回-1.0")
        u_mass_coherence = -1.0
    else:
        coherence_model = CoherenceModel(
            topics=topics, dictionary=gensim_dict, corpus=gensim_corpus, coherence='u_mass'
        )
        u_mass_coherence = coherence_model.get_coherence()
        if np.isnan(u_mass_coherence):
            print(f"⚠️  U-mass计算结果为nan，替换为-1.0")
            u_mass_coherence = -1.0

    return perplexity, u_mass_coherence


# ---------------------- 4. 可视化函数（核心修复：f-string语法错误） ----------------------
def plot_metrics(k_list, perplexity_list, u_mass_list, save_filename='metrics_visualization2-2.png'):
    """
    绘制U-mass一致性和困惑度折线图（跨平台适配+修复f-string语法）
    """
    # 获取当前脚本所在目录（跨平台）
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    save_path = os.path.join(current_dir, save_filename)

    # 创建图表
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 绘制困惑度（左y轴）
    ax1.set_xlabel('Topic', fontsize=12)
    ax1.set_ylabel('Coherence', fontsize=12)
    ax1.plot(k_list, u_mass_list, marker='o', linewidth=2.5, markersize=8, label='Coherence')
    ax1.tick_params(axis='y')
    ax1.grid(True, alpha=0.3)



    # 添加数值标签
    for k, ppl, umass in zip(k_list, perplexity_list, u_mass_list):
        if not np.isnan(ppl):
            ax1.annotate(f'{ppl:.2f}', (k, ppl), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10)

    # 设置标题和图例
    plt.title('Coherence（U-mass）', fontsize=14, fontweight='bold', pad=20)
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=11)

    # 保存图片（自动处理路径）
    try:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ 可视化图表已保存至：{save_path}")
        # 核心修复：将路径替换逻辑移到f-string外部，避免反斜杠错误
        if os.name == 'nt':
            # 先替换反斜杠，再拼接f-string
            web_path = save_path.replace(os.sep, '/')  # 使用os.sep适配系统分隔符，避免硬编码反斜杠
            print(f"   📂 Windows文件资源管理器路径：file:///{web_path}")
    except Exception as e:
        # 兜底方案：保存到桌面
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", save_filename)
        plt.savefig(desktop_path, dpi=300, bbox_inches='tight')
        print(f"\n⚠️  当前目录保存失败，已保存到桌面：{desktop_path}")
        # 同样修复桌面路径的显示
        if os.name == 'nt':
            web_desktop_path = desktop_path.replace(os.sep, '/')
            print(f"   📂 Windows文件资源管理器路径：file:///{web_desktop_path}")
        save_path = desktop_path

    plt.close()  # 关闭图表释放资源
    return save_path


# ---------------------- 5. 主执行流程 ----------------------
if __name__ == "__main__":
    # 配置参数（适配Windows路径）
    EXCEL_PATH = r'C:\Users\Administrator\Desktop\论文数据\dataset1_processed_final.xlsx'  # 改为当前目录的Excel文件
    TEST_K_LIST = [3, 4, 5, 6, 7, 8, 9, 10]  # 测试的主题数量
    perplexity_results = []  # 存储困惑度结果
    u_mass_results = []  # 存储U-mass结果

    # 加载数据
    try:
        df, time_ids, T, text_col, time_col = load_preprocessed_data(EXCEL_PATH)
    except ValueError as e:
        print(f"\n❌ 数据加载失败：{e}")
        exit(1)

    # 评估不同主题数量并收集结果
    print(f"\n===== 开始模型评估与数据收集 =====")
    for k in TEST_K_LIST:
        print(f"\n🔍 评估主题数量 k={k}")
        try:
            ppl, umass = evaluate_preprocessed_dtm(df, time_ids, T, k)
            perplexity_results.append(ppl)
            u_mass_results.append(umass)
            print(f"✅ 评估成功 | 困惑度：{ppl:.4f} | U-mass一致性：{umass:.4f}")
        except Exception as e:
            print(f"❌ 评估失败：{str(e)}")
            perplexity_results.append(np.nan)
            u_mass_results.append(np.nan)
            continue

    # 绘制可视化图表（仅当有有效数据时）
    valid_ppl = [x for x in perplexity_results if not np.isnan(x)]
    valid_umass = [x for x in u_mass_results if not np.isnan(x)]
    if valid_ppl and valid_umass:
        plot_path = plot_metrics(TEST_K_LIST, perplexity_results, u_mass_results)

        # 输出最优结果
        valid_indices = [i for i, (ppl, umass) in enumerate(zip(perplexity_results, u_mass_results)) if
                         not np.isnan(ppl) and not np.isnan(umass)]
        if valid_indices:
            best_idx = max(valid_indices, key=lambda i: u_mass_results[i])
            best_k = TEST_K_LIST[best_idx]
            best_ppl = perplexity_results[best_idx]
            best_umass = u_mass_results[best_idx]
            print(f"\n===== 最优模型结果 =====")
            print(f"最优主题数量：k={best_k}")
            print(f"困惑度（越低越好）：{best_ppl:.4f}")
            print(f"U-mass一致性（越接近0越好）：{best_umass:.4f}")
            print(f"可视化图表路径：{plot_path}")
    else:
        print("\n❌ 无有效评估结果，无法绘制可视化图表！")
