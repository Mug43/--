import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据文件
file_path = 'qingshandataforregression.xlsx' # 将文件保存到同目录下的地址
df = pd.read_excel(file_path)

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签（黑体）
plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号

# 重命名源文件的列名在 DataFrame 中的列标签
df.columns = ['蒸发量', '降雨量', '径流量']

# 查看数据的基本信息，包括列名、非空值数量和数据类型
print("--- 数据概览 ---")
print(df.info())

# 打印数据的前5行，快速了解数据格式
print("\n--- 数据前5行 ---")
print(df.head())

# 一、缺失值检查
print("\n--- 缺失值检查 ---")
print(df.isnull().sum())


# 二、异常值检测（使用箱线图）
print("\n--- 异常值检测 ---")
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.boxplot(y=df['蒸发量'])
plt.title('蒸发量')
plt.ylabel('蒸发量数值')

plt.subplot(1, 3, 2)
sns.boxplot(y=df['降雨量'])
plt.title('降雨量')
plt.ylabel('降雨量数值')

plt.subplot(1, 3, 3)
sns.boxplot(y=df['径流量'])
plt.title('径流量')
plt.ylabel('径流量数值')

plt.tight_layout()
plt.show()

# 三、偏度与正态性检查（为数据变换做准备）
print("\n 偏度检查 ")
print(df.skew())

# 可视化数据分布，使用直方图和核密度估计曲线
plt.figure(figsize=(15, 5), dpi=600)

plt.subplot(1, 3, 1)
sns.histplot(df['蒸发量'], kde=True)
plt.title('蒸发量分布')

plt.subplot(1, 3, 2)
sns.histplot(df['降雨量'], kde=True)
plt.title('降雨量分布')

plt.subplot(1, 3, 3)
sns.histplot(df['径流量'], kde=True)
plt.title('径流量分布')

plt.tight_layout()
plt.show()

print("\n 结果总结 ")
for col in df.columns:
    skewness = df[col].skew()
    if abs(skewness) > 1:
        print(f"{col.capitalize()}: 偏度为 {skewness:.2f}，表明数据存在显著的右偏。可能需要进行对数或幂变换来改善分布，以提升回归模型性能。")
    else:
        print(f"{col.capitalize()}: 偏度为 {skewness:.2f}，接近于0。分布相对对称，可能不需要进行变换。")

