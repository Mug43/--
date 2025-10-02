"""
降雨量和径流量：进行对数变换校正右偏分布。
蒸发量：进行 Z-Score 标准化
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

file_path = 'qingshandataforregression.xlsx' 
df = pd.read_excel(file_path)
df.columns = ['蒸发量', '降雨量', '径流量']

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("--- 原始数据偏度 ---")
print(df.skew())

# 对降雨量和径流量进行 np.log1p(x) 变换
df_log_rain_run = df[['降雨量', '径流量']].apply(np.log1p)
df_log_rain_run.columns = ['ln_降雨量', 'ln_径流量']

# 蒸发量保留原始值，等待下一步标准化
df_combined = pd.concat([df['蒸发量'], df_log_rain_run], axis=1)

print("\n降雨量和径流量对数变换后数据偏度")
print(df_combined.skew())


# 标准化
X = df_combined[['蒸发量', 'ln_降雨量']] 
y = df_combined['ln_径流量']           
scaler = StandardScaler()

# 对 '蒸发量' 和 'ln_降雨量' 列进行标准化
X_scaled_array = scaler.fit_transform(X)
X_final = pd.DataFrame(X_scaled_array, columns=['scaled_蒸发量', 'scaled_ln_降雨量'])

print("\n最终特征 X 统计信息")
# 检查标准化后的均值接近0，标准差接近1
print(X_final.describe().loc[['mean', 'std']])

#划分训练集和测试集 (为后续模型训练做准备) ---

# 以80%训练集，20%测试集进行划分
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42 # 建议设置随机种子以确保结果可复现
)

print(f"\n训练集样本量: {len(X_train)}")
print(f"测试集样本量: {len(X_test)}")

# X_train, X_test, y_train, y_test 即可用于模型训练