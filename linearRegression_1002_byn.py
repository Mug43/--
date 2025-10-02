# 导包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取划分好的数据集
# 训练集
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")
# 测试集
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")

# 构建线性回归模型
Model_OLR = LinearRegression()
Model_OLR.fit(X_train, y_train)

intercept = Model_OLR.intercept_[0]
coefficients = Model_OLR.coef_[0]

feature_names = X_train.columns.tolist()
target_name = y_train.columns[0]

print(f"\n截距 (Intercept): b = {intercept:.2f}")

print(f"回归系数 (Coefficients):")
for i, (name, coef) in enumerate(zip(feature_names, coefficients)):
    print(f" \u03B2{i+1} ({name}): {coef:.2f}")

equation_parts = [f"{intercept:.2f}"]
for name, coef in zip(feature_names, coefficients):
    sign = "+" if coef >= 0 else "-"
    equation_parts.append(f"{sign} {abs(coef):.2f} × {name}")

equation_str = " ".join(equation_parts)
print(f"\n{target_name} = {equation_str}")

# 模型预测（对数空间）
y_train_pred_log = Model_OLR.predict(X_train)
y_test_pred_log = Model_OLR.predict(X_test)

# 反变换到原始空间（实际径流量）
y_train_original = np.expm1(y_train.values.flatten())
y_test_original = np.expm1(y_test.values.flatten())
y_train_pred = np.expm1(y_train_pred_log.flatten())
y_test_pred = np.expm1(y_test_pred_log.flatten())

# 计算评估指标
def calculate_nse(y_true, y_pred):
    """计算Nash-Sutcliffe效率系数 (NSE)"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    nse = 1 - (numerator / denominator)
    return nse

# 训练集评估（原始空间）
train_r2 = r2_score(y_train_original, y_train_pred)
train_mse = mean_squared_error(y_train_original, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train_original, y_train_pred)
train_nse = calculate_nse(y_train_original, y_train_pred)

# 测试集评估（原始空间）
test_r2 = r2_score(y_test_original, y_test_pred)
test_mse = mean_squared_error(y_test_original, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test_original, y_test_pred)
test_nse = calculate_nse(y_test_original, y_test_pred)

# 打印评估结果
print("\n" + "="*60)
print("模型评估指标")
print("="*60)
print(f"\n{'指标':<15} {'训练集':<20} {'测试集':<20}")
print("-"*60)
print(f"{'R^2':<15} {train_r2:<20.6f} {test_r2:<20.6f}")
print(f"{'MSE':<15} {train_mse:<20.6f} {test_mse:<20.6f}")
print(f"{'RMSE':<15} {train_rmse:<20.6f} {test_rmse:<20.6f}")
print(f"{'MAE':<15} {train_mae:<20.6f} {test_mae:<20.6f}")
print(f"{'NSE':<15} {train_nse:<20.6f} {test_nse:<20.6f}")
print("="*60)

# 可视化预测曲线
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. 训练集：实际值 vs 预测值（散点图）
axes[0, 0].scatter(y_train_original, y_train_pred, alpha=0.6, edgecolors='k', linewidths=0.5)
axes[0, 0].plot([y_train_original.min(), y_train_original.max()], 
                [y_train_original.min(), y_train_original.max()], 
                'r--', lw=2, label='理想预测线')
axes[0, 0].set_xlabel('实际径流量', fontsize=12)
axes[0, 0].set_ylabel('预测径流量', fontsize=12)
axes[0, 0].set_title(f'训练集: 实际值 vs 预测值（原始空间）\nR²={train_r2:.4f}, RMSE={train_rmse:.4f}', 
                     fontsize=13, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. 测试集：实际值 vs 预测值（散点图）
axes[0, 1].scatter(y_test_original, y_test_pred, alpha=0.6, color='orange', 
                   edgecolors='k', linewidths=0.5)
axes[0, 1].plot([y_test_original.min(), y_test_original.max()], 
                [y_test_original.min(), y_test_original.max()], 
                'r--', lw=2, label='理想预测线')
axes[0, 1].set_xlabel('实际径流量', fontsize=12)
axes[0, 1].set_ylabel('预测径流量', fontsize=12)
axes[0, 1].set_title(f'测试集: 实际值 vs 预测值（原始空间）\nR²={test_r2:.4f}, RMSE={test_rmse:.4f}', 
                     fontsize=13, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. 训练集：预测曲线（时间序列）
train_indices = range(len(y_train_original))
axes[1, 0].plot(train_indices, y_train_original, 'b-o', 
                label='实际值', markersize=4, linewidth=1.5)
axes[1, 0].plot(train_indices, y_train_pred, 'r-s', 
                label='预测值', markersize=3, linewidth=1.5, alpha=0.7)
axes[1, 0].set_xlabel('样本索引', fontsize=12)
axes[1, 0].set_ylabel('径流量', fontsize=12)
axes[1, 0].set_title(f'训练集预测曲线（原始空间）\nNSE={train_nse:.4f}, MAE={train_mae:.4f}', 
                     fontsize=13, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. 测试集：预测曲线（时间序列）
test_indices = range(len(y_test_original))
axes[1, 1].plot(test_indices, y_test_original, 'b-o', 
                label='实际值', markersize=4, linewidth=1.5)
axes[1, 1].plot(test_indices, y_test_pred, 'r-s', 
                label='预测值', markersize=3, linewidth=1.5, alpha=0.7)
axes[1, 1].set_xlabel('样本索引', fontsize=12)
axes[1, 1].set_ylabel('径流量', fontsize=12)
axes[1, 1].set_title(f'测试集预测曲线（原始空间）\nNSE={test_nse:.4f}, MAE={test_mae:.4f}', 
                     fontsize=13, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('线性回归模型预测结果.png', dpi=300, bbox_inches='tight')
plt.savefig('线性回归模型预测结果.svg', format='svg', bbox_inches='tight')
print("\n图表已保存为 '线性回归模型预测结果.png' 和 '线性回归模型预测结果.svg'")
plt.show()

# 残差分析图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 训练集残差（原始空间）
train_residuals = y_train_original - y_train_pred
axes[0].scatter(y_train_pred, train_residuals, alpha=0.6, edgecolors='k', linewidths=0.5)
axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0].set_xlabel('预测径流量', fontsize=12)
axes[0].set_ylabel('残差（实际值 - 预测值）', fontsize=12)
axes[0].set_title('训练集残差图（原始空间）', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# 测试集残差（原始空间）
test_residuals = y_test_original - y_test_pred
axes[1].scatter(y_test_pred, test_residuals, alpha=0.6, color='orange', 
                edgecolors='k', linewidths=0.5)
axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1].set_xlabel('预测径流量', fontsize=12)
axes[1].set_ylabel('残差（实际值 - 预测值）', fontsize=12)
axes[1].set_title('测试集残差图（原始空间）', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('残差分析图.png', dpi=300, bbox_inches='tight')
plt.savefig('残差分析图.svg', format='svg', bbox_inches='tight')
print("残差分析图已保存为 '残差分析图.png' 和 '残差分析图.svg'")
plt.show()


