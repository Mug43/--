# 导包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

# Matplotlib 配置
plt.rcParams['font.sans-serif'] = ['SimHei'] # 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

# --- 1. 数据加载与预处理 ---

# 假设文件已上传
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")

# 对数空间的y值（用于模型训练）
y_train_log = y_train.values.flatten()
y_test_log = y_test.values.flatten()

# 原始空间的y值（用于评估）
y_train_original = np.expm1(y_train_log)
y_test_original = np.expm1(y_test_log)

target_name = y_train.columns[0]

print("="*70)
print("数据加载成功，假设特征已完成标准化。")
print("="*70)

# --- 2. 评估指标函数 ---

def calculate_nse(y_true, y_pred):
    """计算Nash-Sutcliffe效率系数 (NSE)"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    if denominator == 0:
        return -999.0 
    
    numerator = np.sum((y_true - y_pred) ** 2)
    nse = 1 - (numerator / denominator)
    return nse

# --- 3. 定义模型与超参数搜索空间 ---

base_sgd = SGDRegressor(
    learning_rate='invscaling',
    eta0=0.01, 
    max_iter=2000, # 增加迭代次数以确保收敛
    tol=1e-4, # 提高容忍度
    random_state=42,
    fit_intercept=True # 确保拟合截距
)

# 定义模型列表和超参数搜索空间
# 注意：OLS模型不需要搜索 alpha，但需要包含在循环中
models_to_tune = {
    'SGD_OLS (None)': {
        'penalty': [None],  # 修正：使用 None 而不是字符串
        'alpha': [0.0001], 
    },
    'SGD_Ridge (L2)': {
        'penalty': ['l2'],
        'alpha': np.logspace(-6, 1, 15), # 扩大搜索范围和精度
    },
    'SGD_Lasso (L1)': {
        'penalty': ['l1'],
        'alpha': np.logspace(-6, 1, 15),
    },
    'SGD_ElasticNet': {
        'penalty': ['elasticnet'],
        'alpha': np.logspace(-6, 1, 10),
        'l1_ratio': [0.1, 0.5, 0.9],
    }
}

best_models = {}
best_params_dict = {}
alpha_search_results = {} # 存储超参数搜索结果用于可视化
training_history = {} # 存储训练过程中的损失函数变化

print("开始进行 SGD 梯度下降模型的超参数 Grid Search 优化...")
print("="*70)

# --- 4. 超参数寻找 (Grid Search) ---

for name, params in models_to_tune.items():
    
    grid_search = GridSearchCV(
        base_sgd, 
        params, 
        cv=5, 
        scoring='r2', 
        n_jobs=-1, 
        verbose=0,
        return_train_score=True  # 返回训练分数用于分析
    )
    grid_search.fit(X_train, y_train_log)

    best_models[name] = grid_search.best_estimator_
    best_params_dict[name] = grid_search.best_params_

    print(f"\n模型: {name}")
    print(f"  最优 R² (CV): {grid_search.best_score_:.6f}")
    print(f"  最优参数: {grid_search.best_params_}")
    print("-" * 30)
    
    # 提取超参数搜索结果
    if name != 'SGD_OLS (None)':
        cv_results = pd.DataFrame(grid_search.cv_results_)
        
        # 对于 ElasticNet，固定 l1_ratio=0.5 来绘制一条曲线
        if name == 'SGD_ElasticNet':
            df_plot = cv_results[cv_results['param_l1_ratio'] == 0.5]
            alpha_values = df_plot['param_alpha'].values
            mean_scores = df_plot['mean_test_score'].values
            std_scores = df_plot['std_test_score'].values
        else:
            alpha_values = cv_results['param_alpha'].values
            mean_scores = cv_results['mean_test_score'].values
            std_scores = cv_results['std_test_score'].values
            
        alpha_search_results[name] = {
            'alpha': alpha_values,
            'r2_score': mean_scores,
            'std_score': std_scores,
            'best_alpha': grid_search.best_params_['alpha']
        }
    
    # 使用最优参数重新训练，记录训练过程（使用partial_fit模拟）
    best_model = best_models[name]
    # 注意：SGDRegressor 不直接提供损失历史，我们使用重新训练来获取
    losses = []
    epochs = []
    
    # 创建一个新的模型用于记录训练过程
    tracking_model = SGDRegressor(
        penalty=best_params_dict[name]['penalty'],
        alpha=best_params_dict[name]['alpha'],
        learning_rate='invscaling',
        eta0=0.01,
        max_iter=1,  # 每次只训练一个epoch
        tol=None,
        warm_start=True,  # 允许增量训练
        random_state=42,
        fit_intercept=True
    )
    
    # 如果是 ElasticNet，需要添加 l1_ratio
    if name == 'SGD_ElasticNet':
        tracking_model.l1_ratio = best_params_dict[name]['l1_ratio']
    
    # 训练多个epoch并记录损失
    losses_log = []  # 对数空间的损失
    losses_original = []  # 原始空间的损失
    for epoch in range(100):
        tracking_model.fit(X_train, y_train_log)
        y_pred_log = tracking_model.predict(X_train)
        
        # 对数空间的损失
        loss_log = mean_squared_error(y_train_log, y_pred_log)
        losses_log.append(loss_log)
        
        # 原始空间的损失
        y_pred_original = np.expm1(y_pred_log)
        loss_original = mean_squared_error(y_train_original, y_pred_original)
        losses_original.append(loss_original)
        
        epochs.append(epoch + 1)
    
    training_history[name] = {
        'epochs': epochs,
        'losses_log': losses_log,
        'losses_original': losses_original
    }

print("="*70)

# --- 5. 可视化超参数寻找过程 ---

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 左图：超参数 alpha 对 R² 的影响
colors_map = {'SGD_Ridge (L2)': 'red', 'SGD_Lasso (L1)': 'green', 'SGD_ElasticNet': 'purple'}

for name, data in alpha_search_results.items():
    color = colors_map.get(name, 'blue')
    ax1.plot(data['alpha'], data['r2_score'], label=name, marker='o', 
             markersize=5, linewidth=2, color=color, alpha=0.7)
    
    # 添加置信区间（标准差）
    ax1.fill_between(data['alpha'], 
                     data['r2_score'] - data['std_score'], 
                     data['r2_score'] + data['std_score'], 
                     alpha=0.2, color=color)
    
    # 标记最优 alpha
    best_alpha = data['best_alpha']
    best_idx = np.where(data['alpha'] == best_alpha)[0][0]
    best_r2 = data['r2_score'][best_idx]
    ax1.scatter([best_alpha], [best_r2], color=color, s=200, zorder=5, 
               marker='*', edgecolors='black', linewidths=2)
    ax1.annotate(f'α={best_alpha:.1e}\nR²={best_r2:.4f}', 
                xy=(best_alpha, best_r2), 
                xytext=(10, 10), textcoords='offset points',
                fontsize=9, ha='left',
                bbox=dict(boxstyle='round,pad=0.5', fc=color, alpha=0.3),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

ax1.set_xscale('log')
ax1.set_xlabel('正则化参数 $\\alpha$ (对数尺度)', fontsize=13, fontweight='bold')
ax1.set_ylabel('交叉验证平均 $R^2$ 得分', fontsize=13, fontweight='bold')
ax1.set_title('超参数 $\\alpha$ 对模型性能的影响', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11, loc='lower right')
ax1.grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.7)

# 右图：训练过程中损失函数的变化（原始空间）
for name, data in training_history.items():
    color = colors_map.get(name, 'blue')
    ax2.plot(data['epochs'], data['losses_original'], label=name, 
             linewidth=2, color=color, alpha=0.7)

ax2.set_xlabel('训练轮数 (Epochs)', fontsize=13, fontweight='bold')
ax2.set_ylabel('均方误差 (MSE) - 原始空间', fontsize=13, fontweight='bold')
ax2.set_title('训练过程中损失函数的变化\n（在实际径流量空间计算）', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11, loc='upper right')
ax2.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
ax2.set_yscale('log')  # 使用对数尺度更好地观察变化

plt.tight_layout()
plt.savefig('超参数寻找与训练过程可视化.png', dpi=300, bbox_inches='tight')
plt.savefig('超参数寻找与训练过程可视化.svg', format='svg', bbox_inches='tight')
print("\n[成功] 超参数寻找与训练过程可视化已保存为 '超参数寻找与训练过程可视化.png' 和 .svg 文件")
plt.show()


# --- 6. 评估四种模型在训练集和测试集上的性能 ---

train_results = {}
test_results = {}

# 评估所有模型
for name, model in best_models.items():
    # === 训练集评估 ===
    y_train_pred_log = model.predict(X_train)
    y_train_pred = np.expm1(y_train_pred_log)
    
    train_r2 = r2_score(y_train_original, y_train_pred)
    train_mse = mean_squared_error(y_train_original, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train_original, y_train_pred)
    train_nse = calculate_nse(y_train_original, y_train_pred)
    
    train_results[name] = {
        'R^2': train_r2, 
        'RMSE': train_rmse, 
        'MAE': train_mae, 
        'NSE': train_nse
    }
    
    # === 测试集评估 ===
    y_test_pred_log = model.predict(X_test)
    y_test_pred = np.expm1(y_test_pred_log)
    
    test_r2 = r2_score(y_test_original, y_test_pred)
    test_mse = mean_squared_error(y_test_original, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test_original, y_test_pred)
    test_nse = calculate_nse(y_test_original, y_test_pred)
    
    test_results[name] = {
        'R^2': test_r2, 
        'RMSE': test_rmse, 
        'MAE': test_mae, 
        'NSE': test_nse
    }

# 打印训练集评估结果
print("\n" + "="*70)
print("模型评估结果 - 训练集 (原始空间)")
print("="*70)
print(f"{'模型':<20} {'R^2':<10} {'RMSE':<10} {'MAE':<10} {'NSE':<10}")
print("-" * 70)

for name in best_models.keys():
    r2 = train_results[name]['R^2']
    rmse = train_results[name]['RMSE']
    mae = train_results[name]['MAE']
    nse = train_results[name]['NSE']
    print(f"{name:<20} {r2:.4f} {rmse:.4f} {mae:.4f} {nse:.4f}")

print("="*70)

# 打印测试集评估结果
print("\n" + "="*70)
print("模型评估结果 - 测试集 (原始空间)")
print("="*70)
print(f"{'模型':<20} {'R^2':<10} {'RMSE':<10} {'MAE':<10} {'NSE':<10}")
print("-" * 70)

for name in best_models.keys():
    r2 = test_results[name]['R^2']
    rmse = test_results[name]['RMSE']
    mae = test_results[name]['MAE']
    nse = test_results[name]['NSE']
    print(f"{name:<20} {r2:.4f} {rmse:.4f} {mae:.4f} {nse:.4f}")

print("="*70)

# --- 7. 梯度下降过程的额外可视化 ---
print("\n" + "="*70)
print("生成梯度下降过程的详细可视化...")
print("="*70)

colors_map = {'SGD_OLS (None)': '#1f77b4', 'SGD_Ridge (L2)': '#ff7f0e', 
              'SGD_Lasso (L1)': '#2ca02c', 'SGD_ElasticNet': '#d62728'}

# === 7.1 学习曲线（训练集 vs 验证集性能随epoch变化）===
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, (name, model) in enumerate(best_models.items()):
    # 重新训练以获取每个epoch的训练和验证分数
    from sklearn.model_selection import train_test_split
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train_log, test_size=0.2, random_state=42
    )
    
    train_scores = []
    val_scores = []
    epochs_list = range(1, 101)
    
    # 创建临时模型用于增量训练
    temp_model = SGDRegressor(
        penalty=best_params_dict[name]['penalty'],
        alpha=best_params_dict[name]['alpha'],
        learning_rate='invscaling',
        eta0=0.01,
        max_iter=1,
        tol=None,
        warm_start=True,
        random_state=42,
        fit_intercept=True
    )
    
    if name == 'SGD_ElasticNet':
        temp_model.l1_ratio = best_params_dict[name]['l1_ratio']
    
    for epoch in epochs_list:
        temp_model.fit(X_train_split, y_train_split)
        
        # 训练集R²
        train_pred = temp_model.predict(X_train_split)
        train_r2 = r2_score(y_train_split, train_pred)
        train_scores.append(train_r2)
        
        # 验证集R²
        val_pred = temp_model.predict(X_val_split)
        val_r2 = r2_score(y_val_split, val_pred)
        val_scores.append(val_r2)
    
    # 绘制学习曲线
    axes[idx].plot(epochs_list, train_scores, label='训练集 R²', 
                   linewidth=2, color=colors_map.get(name, 'blue'), alpha=0.8)
    axes[idx].plot(epochs_list, val_scores, label='验证集 R²', 
                   linewidth=2, linestyle='--', color=colors_map.get(name, 'blue'), alpha=0.8)
    axes[idx].set_xlabel('训练轮数 (Epochs)', fontsize=11, fontweight='bold')
    axes[idx].set_ylabel('R² 得分', fontsize=11, fontweight='bold')
    axes[idx].set_title(f'{name}\n学习曲线', fontsize=12, fontweight='bold')
    axes[idx].legend(fontsize=10)
    axes[idx].grid(True, alpha=0.3)

plt.suptitle('四种SGD模型的学习曲线对比\n(观察过拟合/欠拟合)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('SGD模型学习曲线.png', dpi=300, bbox_inches='tight')
plt.savefig('SGD模型学习曲线.svg', format='svg', bbox_inches='tight')
print("[成功] 学习曲线已保存为 'SGD模型学习曲线.png' 和 .svg 文件")
plt.show()

# === 7.2 梯度下降收敛速度对比（对数空间和原始空间的损失）===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

for name, data in training_history.items():
    color = colors_map.get(name, 'blue')
    
    # 左图：对数空间的损失
    ax1.plot(data['epochs'], data['losses_log'], label=name, 
             linewidth=2, color=color, alpha=0.7)
    
    # 右图：原始空间的损失
    ax2.plot(data['epochs'], data['losses_original'], label=name, 
             linewidth=2, color=color, alpha=0.7)

ax1.set_xlabel('训练轮数 (Epochs)', fontsize=13, fontweight='bold')
ax1.set_ylabel('MSE - 对数空间', fontsize=13, fontweight='bold')
ax1.set_title('梯度下降收敛过程\n(对数空间)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

ax2.set_xlabel('训练轮数 (Epochs)', fontsize=13, fontweight='bold')
ax2.set_ylabel('MSE - 原始空间', fontsize=13, fontweight='bold')
ax2.set_title('梯度下降收敛过程\n(原始径流量空间)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

plt.tight_layout()
plt.savefig('SGD梯度下降收敛对比.png', dpi=300, bbox_inches='tight')
plt.savefig('SGD梯度下降收敛对比.svg', format='svg', bbox_inches='tight')
print("[成功] 梯度下降收敛图已保存为 'SGD梯度下降收敛对比.png' 和 .svg 文件")
plt.show()

# === 7.3 模型性能条形图对比 ===
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

metrics = ['R^2', 'RMSE', 'MAE', 'NSE']
model_names = list(best_models.keys())
x_pos = np.arange(len(model_names))
width = 0.35

for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    
    train_values = [train_results[name][metric] for name in model_names]
    test_values = [test_results[name][metric] for name in model_names]
    
    bars1 = ax.bar(x_pos - width/2, train_values, width, label='训练集', alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, test_values, width, label='测试集', alpha=0.8)
    
    ax.set_xlabel('模型', fontsize=11, fontweight='bold')
    ax.set_ylabel(metric, fontsize=11, fontweight='bold')
    ax.set_title(f'{metric} 对比', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([name.replace(' (', '\n(') for name in model_names], 
                       fontsize=9, rotation=0)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 在柱子上添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

plt.suptitle('四种SGD模型性能指标对比\n(训练集 vs 测试集)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('SGD模型性能条形图对比.png', dpi=300, bbox_inches='tight')
plt.savefig('SGD模型性能条形图对比.svg', format='svg', bbox_inches='tight')
print("[成功] 性能条形图已保存为 'SGD模型性能条形图对比.png' 和 .svg 文件")
plt.show()

# --- 8. 为最佳模型（SGD_OLS）绘制详细可视化 ---
print("\n" + "="*70)
print("开始为最佳模型 SGD_OLS (None) 生成详细可视化...")
print("="*70)

best_model_name = 'SGD_OLS (None)'
best_model = best_models[best_model_name]

# 获取预测结果
y_train_pred_log = best_model.predict(X_train)
y_train_pred = np.expm1(y_train_pred_log)
y_test_pred_log = best_model.predict(X_test)
y_test_pred = np.expm1(y_test_pred_log)

# === 9.1 预测结果散点图（实际值 vs 预测值）===
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 训练集散点图
axes[0].scatter(y_train_original, y_train_pred, alpha=0.6, edgecolors='k', linewidths=0.5, s=50)
axes[0].plot([y_train_original.min(), y_train_original.max()], 
             [y_train_original.min(), y_train_original.max()], 
             'r--', lw=2, label='理想预测线')
axes[0].set_xlabel('实际径流量', fontsize=13, fontweight='bold')
axes[0].set_ylabel('预测径流量', fontsize=13, fontweight='bold')
train_r2 = train_results[best_model_name]['R^2']
train_rmse = train_results[best_model_name]['RMSE']
axes[0].set_title(f'训练集: 实际值 vs 预测值\nR²={train_r2:.4f}, RMSE={train_rmse:.2f}', 
                  fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# 测试集散点图
axes[1].scatter(y_test_original, y_test_pred, alpha=0.6, color='orange', 
                edgecolors='k', linewidths=0.5, s=50)
axes[1].plot([y_test_original.min(), y_test_original.max()], 
             [y_test_original.min(), y_test_original.max()], 
             'r--', lw=2, label='理想预测线')
axes[1].set_xlabel('实际径流量', fontsize=13, fontweight='bold')
axes[1].set_ylabel('预测径流量', fontsize=13, fontweight='bold')
test_r2 = test_results[best_model_name]['R^2']
test_rmse = test_results[best_model_name]['RMSE']
axes[1].set_title(f'测试集: 实际值 vs 预测值\nR²={test_r2:.4f}, RMSE={test_rmse:.2f}', 
                  fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('SGD_OLS预测散点图.png', dpi=300, bbox_inches='tight')
plt.savefig('SGD_OLS预测散点图.svg', format='svg', bbox_inches='tight')
print("\n[成功] 预测散点图已保存为 'SGD_OLS预测散点图.png' 和 .svg 文件")
plt.show()

# === 9.2 预测曲线时间序列图 ===
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# 训练集时间序列
train_indices = range(len(y_train_original))
axes[0].plot(train_indices, y_train_original, 'b-o', 
             label='实际值', markersize=4, linewidth=1.5, alpha=0.8)
axes[0].plot(train_indices, y_train_pred, 'r-s', 
             label='预测值', markersize=3, linewidth=1.5, alpha=0.7)
axes[0].set_xlabel('样本索引', fontsize=13, fontweight='bold')
axes[0].set_ylabel('径流量', fontsize=13, fontweight='bold')
train_nse = train_results[best_model_name]['NSE']
train_mae = train_results[best_model_name]['MAE']
axes[0].set_title(f'训练集预测曲线\nNSE={train_nse:.4f}, MAE={train_mae:.2f}', 
                  fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11, loc='upper right')
axes[0].grid(True, alpha=0.3)

# 测试集时间序列
test_indices = range(len(y_test_original))
axes[1].plot(test_indices, y_test_original, 'b-o', 
             label='实际值', markersize=4, linewidth=1.5, alpha=0.8)
axes[1].plot(test_indices, y_test_pred, 'r-s', 
             label='预测值', markersize=3, linewidth=1.5, alpha=0.7)
axes[1].set_xlabel('样本索引', fontsize=13, fontweight='bold')
axes[1].set_ylabel('径流量', fontsize=13, fontweight='bold')
test_nse = test_results[best_model_name]['NSE']
test_mae = test_results[best_model_name]['MAE']
axes[1].set_title(f'测试集预测曲线\nNSE={test_nse:.4f}, MAE={test_mae:.2f}', 
                  fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11, loc='upper right')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('SGD_OLS预测曲线时间序列.png', dpi=300, bbox_inches='tight')
plt.savefig('SGD_OLS预测曲线时间序列.svg', format='svg', bbox_inches='tight')
print("[成功] 预测曲线时间序列图已保存为 'SGD_OLS预测曲线时间序列.png' 和 .svg 文件")
plt.show()

# === 9.3 残差分析图 ===
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 训练集残差散点图
train_residuals = y_train_original - y_train_pred
axes[0, 0].scatter(y_train_pred, train_residuals, alpha=0.6, edgecolors='k', linewidths=0.5, s=40)
axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('预测径流量', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('残差（实际值 - 预测值）', fontsize=12, fontweight='bold')
axes[0, 0].set_title('训练集残差图', fontsize=13, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# 测试集残差散点图
test_residuals = y_test_original - y_test_pred
axes[0, 1].scatter(y_test_pred, test_residuals, alpha=0.6, color='orange', 
                   edgecolors='k', linewidths=0.5, s=40)
axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('预测径流量', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('残差（实际值 - 预测值）', fontsize=12, fontweight='bold')
axes[0, 1].set_title('测试集残差图', fontsize=13, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# 训练集残差直方图
axes[1, 0].hist(train_residuals, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('残差', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('频数', fontsize=12, fontweight='bold')
axes[1, 0].set_title(f'训练集残差分布\n均值={train_residuals.mean():.2f}, 标准差={train_residuals.std():.2f}', 
                     fontsize=13, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# 测试集残差直方图
axes[1, 1].hist(test_residuals, bins=30, edgecolor='black', alpha=0.7, color='lightcoral')
axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('残差', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('频数', fontsize=12, fontweight='bold')
axes[1, 1].set_title(f'测试集残差分布\n均值={test_residuals.mean():.2f}, 标准差={test_residuals.std():.2f}', 
                     fontsize=13, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('SGD_OLS残差分析图.png', dpi=300, bbox_inches='tight')
plt.savefig('SGD_OLS残差分析图.svg', format='svg', bbox_inches='tight')
print("[成功] 残差分析图已保存为 'SGD_OLS残差分析图.png' 和 .svg 文件")
plt.show()

# === 9.4 四模型对比的预测散点图（在一张图上）===
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for idx, (name, model) in enumerate(best_models.items()):
    # 获取测试集预测
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    
    # 绘制散点图
    axes[idx].scatter(y_test_original, y_pred, alpha=0.6, 
                     color=colors_list[idx], edgecolors='k', linewidths=0.5, s=50)
    axes[idx].plot([y_test_original.min(), y_test_original.max()], 
                   [y_test_original.min(), y_test_original.max()], 
                   'r--', lw=2, label='理想预测线')
    
    axes[idx].set_xlabel('实际径流量', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('预测径流量', fontsize=12, fontweight='bold')
    
    r2 = test_results[name]['R^2']
    rmse = test_results[name]['RMSE']
    axes[idx].set_title(f'{name}\nR²={r2:.4f}, RMSE={rmse:.2f}', 
                       fontsize=13, fontweight='bold')
    axes[idx].legend(fontsize=10)
    axes[idx].grid(True, alpha=0.3)

plt.suptitle('四种SGD模型测试集预测对比', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('四种SGD模型预测散点图对比.png', dpi=300, bbox_inches='tight')
plt.savefig('四种SGD模型预测散点图对比.svg', format='svg', bbox_inches='tight')
print("[成功] 四模型预测散点图对比已保存为 '四种SGD模型预测散点图对比.png' 和 .svg 文件")
plt.show()

print("\n" + "="*70)
print("开始导出所有绘图数据...")
print("="*70)

# === 10. 数据导出 ===
import os
export_dir = '绘图数据导出'
if not os.path.exists(export_dir):
    os.makedirs(export_dir)

# --- 10.1 导出训练过程数据（对应：超参数寻找与训练过程可视化.png）---
print("\n[1/9] 导出训练过程数据...")
for name, data in training_history.items():
    df_training = pd.DataFrame({
        'Epoch': data['epochs'],
        'MSE_对数空间': data['losses_log'],
        'MSE_原始空间': data['losses_original']
    })
    filename = f"{export_dir}/01_训练过程_{name.replace(' ', '_').replace('(', '').replace(')', '')}.csv"
    df_training.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"   ✓ {filename}")

# --- 10.2 导出超参数搜索结果（对应：超参数寻找与训练过程可视化.png 左图）---
print("\n[2/9] 导出超参数搜索数据...")
for name, data in alpha_search_results.items():
    df_alpha = pd.DataFrame({
        'Alpha': data['alpha'],
        'R2_Score_Mean': data['r2_score'],
        'R2_Score_Std': data['std_score'],
        'Best_Alpha': [data['best_alpha']] * len(data['alpha'])
    })
    filename = f"{export_dir}/02_超参数搜索_{name.replace(' ', '_').replace('(', '').replace(')', '')}.csv"
    df_alpha.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"   ✓ {filename}")

# --- 10.3 导出模型评估指标（对应：SGD模型性能条形图对比.png）---
print("\n[3/9] 导出模型评估指标...")
metrics_data = []
for name in best_models.keys():
    metrics_data.append({
        '模型': name,
        '训练集_R2': train_results[name]['R^2'],
        '训练集_RMSE': train_results[name]['RMSE'],
        '训练集_MAE': train_results[name]['MAE'],
        '训练集_NSE': train_results[name]['NSE'],
        '测试集_R2': test_results[name]['R^2'],
        '测试集_RMSE': test_results[name]['RMSE'],
        '测试集_MAE': test_results[name]['MAE'],
        '测试集_NSE': test_results[name]['NSE']
    })
df_metrics = pd.DataFrame(metrics_data)
filename = f"{export_dir}/03_模型评估指标汇总.csv"
df_metrics.to_csv(filename, index=False, encoding='utf-8-sig')
print(f"   ✓ {filename}")

# --- 10.4 导出学习曲线数据（对应：SGD模型学习曲线.png）---
print("\n[4/9] 导出学习曲线数据...")
print("   (注：学习曲线数据已在第1步中导出)")

# --- 10.5 导出SGD_OLS预测结果（对应：SGD_OLS预测散点图.png）---
print("\n[5/9] 导出SGD_OLS预测结果...")
# 训练集预测
df_train_pred = pd.DataFrame({
    '样本索引': range(len(y_train_original)),
    '实际径流量': y_train_original,
    '预测径流量': y_train_pred,
    '残差': train_residuals
})
filename = f"{export_dir}/05_SGD_OLS训练集预测结果.csv"
df_train_pred.to_csv(filename, index=False, encoding='utf-8-sig')
print(f"   ✓ {filename}")

# 测试集预测
df_test_pred = pd.DataFrame({
    '样本索引': range(len(y_test_original)),
    '实际径流量': y_test_original,
    '预测径流量': y_test_pred,
    '残差': test_residuals
})
filename = f"{export_dir}/06_SGD_OLS测试集预测结果.csv"
df_test_pred.to_csv(filename, index=False, encoding='utf-8-sig')
print(f"   ✓ {filename}")

# --- 10.6 导出残差统计信息（对应：SGD_OLS残差分析图.png）---
print("\n[6/9] 导出残差统计信息...")
residuals_stats = pd.DataFrame({
    '数据集': ['训练集', '测试集'],
    '残差均值': [train_residuals.mean(), test_residuals.mean()],
    '残差标准差': [train_residuals.std(), test_residuals.std()],
    '残差最小值': [train_residuals.min(), test_residuals.min()],
    '残差最大值': [train_residuals.max(), test_residuals.max()],
    '残差中位数': [np.median(train_residuals), np.median(test_residuals)]
})
filename = f"{export_dir}/07_SGD_OLS残差统计.csv"
residuals_stats.to_csv(filename, index=False, encoding='utf-8-sig')
print(f"   ✓ {filename}")

# --- 10.7 导出四模型测试集预测结果（对应：四种SGD模型预测散点图对比.png）---
print("\n[7/9] 导出四模型测试集预测结果...")
for name, model in best_models.items():
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    
    df_model_pred = pd.DataFrame({
        '样本索引': range(len(y_test_original)),
        '实际径流量': y_test_original,
        '预测径流量': y_pred,
        '残差': y_test_original - y_pred,
        'R2': [test_results[name]['R^2']] * len(y_test_original),
        'RMSE': [test_results[name]['RMSE']] * len(y_test_original)
    })
    filename = f"{export_dir}/08_测试集预测_{name.replace(' ', '_').replace('(', '').replace(')', '')}.csv"
    df_model_pred.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"   ✓ {filename}")

# --- 10.8 导出最优超参数配置---
print("\n[8/9] 导出最优超参数配置...")
best_params_data = []
for name, params in best_params_dict.items():
    row = {'模型': name}
    row.update(params)
    best_params_data.append(row)
df_best_params = pd.DataFrame(best_params_data)
filename = f"{export_dir}/09_最优超参数配置.csv"
df_best_params.to_csv(filename, index=False, encoding='utf-8-sig')
print(f"   ✓ {filename}")

# --- 10.9 创建数据说明文件---
print("\n[9/9] 创建数据说明文档...")
readme_content = """# SGD回归模型绘图数据说明

本文件夹包含所有可视化图表对应的原始数据。

## 文件清单及对应图表

### 1. 训练过程数据（01_训练过程_*.csv）
**对应图表**: 超参数寻找与训练过程可视化.png（右图）
**内容**: 
- Epoch: 训练轮次
- MSE_对数空间: 对数空间的均方误差
- MSE_原始空间: 原始径流量空间的均方误差
**说明**: 展示四种模型在训练过程中损失函数的变化

### 2. 超参数搜索数据（02_超参数搜索_*.csv）
**对应图表**: 超参数寻找与训练过程可视化.png（左图）
**内容**:
- Alpha: 正则化参数α
- R2_Score_Mean: 交叉验证R²得分均值
- R2_Score_Std: 交叉验证R²得分标准差
- Best_Alpha: 最优α值
**说明**: Grid Search超参数搜索结果

### 3. 模型评估指标汇总（03_模型评估指标汇总.csv）
**对应图表**: SGD模型性能条形图对比.png
**内容**: 四种模型在训练集和测试集上的R²、RMSE、MAE、NSE指标
**说明**: 所有评估指标均在原始径流量空间计算

### 4. 学习曲线数据
**对应图表**: SGD模型学习曲线.png
**说明**: 数据包含在"01_训练过程_*.csv"文件中

### 5. SGD_OLS训练集预测结果（05_SGD_OLS训练集预测结果.csv）
**对应图表**: 
- SGD_OLS预测散点图.png（左图）
- SGD_OLS预测曲线时间序列.png（上图）
**内容**:
- 样本索引: 样本序号
- 实际径流量: 真实径流量值
- 预测径流量: 模型预测值
- 残差: 实际值 - 预测值

### 6. SGD_OLS测试集预测结果（06_SGD_OLS测试集预测结果.csv）
**对应图表**: 
- SGD_OLS预测散点图.png（右图）
- SGD_OLS预测曲线时间序列.png（下图）
**内容**: 同上

### 7. SGD_OLS残差统计（07_SGD_OLS残差统计.csv）
**对应图表**: SGD_OLS残差分析图.png
**内容**: 残差的均值、标准差、最小值、最大值、中位数
**说明**: 残差应接近正态分布，均值接近0

### 8. 四模型测试集预测（08_测试集预测_*.csv）
**对应图表**: 四种SGD模型预测散点图对比.png
**内容**: 四种模型各自在测试集上的预测结果和残差

### 9. 最优超参数配置（09_最优超参数配置.csv）
**内容**: Grid Search找到的每个模型的最优超参数
**说明**: 这些参数用于训练最终模型

## 数据单位说明
- 径流量: 原始径流量单位（已通过np.expm1反变换）
- MSE/RMSE: 径流量单位的平方/径流量单位
- MAE: 径流量单位
- R²/NSE: 无量纲，范围[-∞, 1]，越接近1越好

## 数据生成时间
{}

## 注意事项
1. 所有评估指标均在**原始空间**（实际径流量）计算
2. 训练过程中的损失同时记录了对数空间和原始空间的值
3. 残差 = 实际值 - 预测值，正值表示低估，负值表示高估
""".format(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'))

filename = f"{export_dir}/README.txt"
with open(filename, 'w', encoding='utf-8') as f:
    f.write(readme_content)
print(f"   ✓ {filename}")

print("\n" + "="*70)
print(f"数据导出完成！所有文件已保存到文件夹: {export_dir}/")
print("="*70)

print("\n" + "="*70)
print("所有可视化图表生成完成！")
print("="*70)