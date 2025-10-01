"""
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_and_preprocess_data():
    print(" 加载原始数据 ")
    df_raw = pd.read_excel('qingshandataforregression.xlsx')
    df_raw.columns = ['蒸发量', '降雨量', '径流量']
    print(f"数据形状: {df_raw.shape}")
    
    # 创建预处理数据副本
    df_processed = df_raw.copy()
    
    # 对数变换解决偏度问题
    print("\n 对数变换 ")
    df_processed['降雨量_log'] = np.log1p(df_processed['降雨量'])
    df_processed['径流量_log'] = np.log1p(df_processed['径流量'])
    
    # 检查变换效果
    print("变换前偏度:")
    print(f"降雨量: {df_raw['降雨量'].skew():.3f}")
    print(f"径流量: {df_raw['径流量'].skew():.3f}")
    
    print("\n变换后偏度:")
    print(f"降雨量_log: {df_processed['降雨量_log'].skew():.3f}")
    print(f"径流量_log: {df_processed['径流量_log'].skew():.3f}")
    
    # 标准化蒸发量
    scaler = StandardScaler()
    df_processed['蒸发量_std'] = scaler.fit_transform(df_processed[['蒸发量']])
    
    # 异常值处理（可选：温和处理）
    print("\n 异常值统计 ")
    for col in ['蒸发量', '降雨量', '径流量']:
        Q1 = df_processed[col].quantile(0.25)
        Q3 = df_processed[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df_processed[(df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)]
        print(f"{col}: {len(outliers)} 个异常值")
    
    return df_raw, df_processed, scaler

def visualize_preprocessing_effects(df_raw, df_processed):
    plt.figure(figsize=(15, 10))
    
    # 原始数据分布
    plt.subplot(2, 3, 1)
    sns.histplot(df_raw['降雨量'], kde=True, alpha=0.7)
    plt.title(f'原始降雨量分布 (偏度: {df_raw["降雨量"].skew():.2f})')
    
    plt.subplot(2, 3, 2)
    sns.histplot(df_raw['径流量'], kde=True, alpha=0.7)
    plt.title(f'原始径流量分布 (偏度: {df_raw["径流量"].skew():.2f})')
    
    plt.subplot(2, 3, 3)
    sns.scatterplot(data=df_raw, x='降雨量', y='径流量', alpha=0.6)
    plt.title(f'原始数据散点图 (相关性: {df_raw["降雨量"].corr(df_raw["径流量"]):.3f})')
    
    # 预处理后数据分布
    plt.subplot(2, 3, 4)
    sns.histplot(df_processed['降雨量_log'], kde=True, alpha=0.7, color='orange')
    plt.title(f'对数变换后降雨量 (偏度: {df_processed["降雨量_log"].skew():.2f})')
    
    plt.subplot(2, 3, 5)
    sns.histplot(df_processed['径流量_log'], kde=True, alpha=0.7, color='orange')
    plt.title(f'对数变换后径流量 (偏度: {df_processed["径流量_log"].skew():.2f})')
    
    plt.subplot(2, 3, 6)
    sns.scatterplot(data=df_processed, x='降雨量_log', y='径流量_log', alpha=0.6, color='orange')
    plt.title(f'变换后散点图 (相关性: {df_processed["降雨量_log"].corr(df_processed["径流量_log"]):.3f})')
    
    plt.tight_layout()
    plt.show()

def prepare_modeling_data(df_processed):
    
    # 准备不同的特征组合
    data_variants = {
        'original': {
            'X': df_processed[['蒸发量', '降雨量']],
            'y': df_processed['径流量'],
            'description': '原始数据'
        },
        'log_transformed': {
            'X': df_processed[['蒸发量_std', '降雨量_log']],
            'y': df_processed['径流量_log'],
            'description': '对数变换+标准化'
        },
        'rainfall_only': {
            'X': df_processed[['降雨量_log']],
            'y': df_processed['径流量_log'],
            'description': '仅降雨量（主要预测因子）'
        }
    }
    
    return data_variants

if __name__ == "__main__":
    # 执行数据预处理
    print("开始数据预处理...")
    df_raw, df_processed, scaler = load_and_preprocess_data()
    
    # 可视化效果
    print("\\n生成预处理效果图...")
    visualize_preprocessing_effects(df_raw, df_processed)
    
    # 准备建模数据
    print("\\n准备建模数据...")
    data_variants = prepare_modeling_data(df_processed)
    
    # 保存预处理后的数据
    df_processed.to_csv('preprocessed_data.csv', index=False, encoding='utf-8-sig')
    print("\\n预处理完成！数据已保存到 'preprocessed_data.csv'")
    
    # 输出数据摘要
    print("\\n 预处理数据摘要 ")
    print(df_processed.describe())
    
    print("\\n 建模数据准备完成 ")
    for name, data in data_variants.items():
        print(f"{name}: {data['description']}")
        print(f"  特征数: {data['X'].shape[1]}, 样本数: {data['X'].shape[0]}")