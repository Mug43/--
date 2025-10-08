"""
========================================
径流预测 - SGD随机梯度下降回归模型分析
========================================
本程序实现了四种SGD回归模型的训练、评估与可视化：
  1. SGD_OLS (无正则化)
  2. SGD_Ridge (L2正则化)
  3. SGD_Lasso (L1正则化)
  4. SGD_ElasticNet (L1+L2混合正则化)

主要功能：
  - 超参数网格搜索优化
  - 模型训练过程可视化
  - 多指标性能评估（R², RMSE, MAE, NSE）
  - 预测结果与残差分析
  
作者: byn
日期: 2024-10-02
"""

# ============================================================================
# 1. 导入必要的库
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from matplotlib.ticker import LogFormatter

# 忽略警告信息
warnings.filterwarnings('ignore')

# ============================================================================
# 2. Matplotlib 绘图配置
# ============================================================================

plt.rcParams['font.sans-serif'] = ['SimHei']      # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False         # 正常显示负号
plt.rcParams['xtick.direction'] = 'in'             # x轴刻度线向内
plt.rcParams['ytick.direction'] = 'in'             # y轴刻度线向内

# ============================================================================
# 3. 数据加载与预处理
# ============================================================================

# 加载训练集和测试集数据（特征已标准化）
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")

# 对数空间的目标变量（用于模型训练，已进行log1p变换）
y_train_log = y_train.values.flatten()
y_test_log = y_test.values.flatten()

# 原始空间的目标变量（用于模型评估，通过expm1反变换得到）
y_train_original = np.expm1(y_train_log)
y_test_original = np.expm1(y_test_log)

# 目标变量名称
target_name = y_train.columns[0]

print("="*70)
print("数据加载成功，特征已完成标准化处理。")
print("="*70)

# ============================================================================
# 4. 自定义评估指标函数
# ============================================================================

def calculate_nse(y_true, y_pred):
    """
    计算Nash-Sutcliffe效率系数 (NSE)
    
    NSE是水文学中常用的模型评估指标，范围从-∞到1
    NSE = 1 表示完美预测
    NSE = 0 表示模型预测效果与均值相当
    NSE < 0 表示模型预测效果不如均值
    
    参数：
        y_true: 实际观测值
        y_pred: 模型预测值
    
    返回：
        float: NSE系数值
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # 计算分母：实际值与均值的平方差之和
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    if denominator == 0:
        return -999.0  # 避免除零错误
    
    # 计算分子：预测值与实际值的平方差之和
    numerator = np.sum((y_true - y_pred) ** 2)
    nse = 1 - (numerator / denominator)
    return nse

# ============================================================================
# 5. 定义基础模型与超参数搜索空间
# ============================================================================

# 创建基础SGD回归器（所有模型共享的基础配置）
base_sgd = SGDRegressor(
    learning_rate='invscaling',    # 使用逆比例缩放学习率
    eta0=0.01,                     # 初始学习率
    max_iter=2000,                 # 最大迭代次数（确保收敛）
    tol=1e-4,                      # 收敛容忍度
    random_state=42,               # 随机种子（保证结果可复现）
    fit_intercept=True             # 拟合截距项
)

# 定义四种模型的超参数搜索空间
models_to_tune = {
    'SGD_OLS (None)': {
        'penalty': [None],              # 无正则化
        'alpha': [0.0001],              # alpha值对OLS无实际影响
    },
    'SGD_Ridge (L2)': {
        'penalty': ['l2'],              # L2正则化（岭回归）
        'alpha': np.logspace(-6, 1, 15), # 正则化强度：10^-6 到 10^1
    },
    'SGD_Lasso (L1)': {
        'penalty': ['l1'],              # L1正则化（Lasso回归）
        'alpha': np.logspace(-6, 1, 15), # 正则化强度：10^-6 到 10^1
    },
    'SGD_ElasticNet': {
        'penalty': ['elasticnet'],      # L1+L2混合正则化
        'alpha': np.logspace(-6, 1, 10), # 正则化强度：10^-6 到 10^1
        'l1_ratio': [0.1, 0.5, 0.9],    # L1与L2的混合比例
    }
}

# 初始化存储字典
best_models = {}              # 存储每个模型的最优估计器
best_params_dict = {}         # 存储每个模型的最优超参数
alpha_search_results = {}     # 存储超参数搜索结果（用于可视化）
training_history = {}         # 存储训练过程中的损失函数变化

print("开始进行 SGD 梯度下降模型的超参数网格搜索优化...")
print("="*70)

# ============================================================================
# 6. 超参数网格搜索与最优模型训练
# ============================================================================

for name, params in models_to_tune.items():
    
    # 创建网格搜索对象
    grid_search = GridSearchCV(
        estimator=base_sgd,              # 基础估计器
        param_grid=params,               # 超参数搜索空间
        cv=5,                            # 5折交叉验证
        scoring='r2',                    # 评分标准：R²系数
        n_jobs=-1,                       # 使用所有CPU核心并行计算
        verbose=0,                       # 不显示详细信息
        return_train_score=True          # 返回训练分数（用于过拟合分析）
    )
    
    # 执行网格搜索
    grid_search.fit(X_train, y_train_log)

    # 保存最优模型和最优参数
    best_models[name] = grid_search.best_estimator_
    best_params_dict[name] = grid_search.best_params_

    # 输出最优结果
    print(f"\n模型: {name}")
    print(f"  最优 R² (交叉验证): {grid_search.best_score_:.6f}")
    print(f"  最优参数: {grid_search.best_params_}")
    print("-" * 30)
    
    # 提取超参数搜索结果（用于后续可视化）
    if name != 'SGD_OLS (None)':
        cv_results = pd.DataFrame(grid_search.cv_results_)
        
        # 针对ElasticNet：固定l1_ratio=0.5来绘制一条曲线
        if name == 'SGD_ElasticNet':
            df_plot = cv_results[cv_results['param_l1_ratio'] == 0.5]
            alpha_values = df_plot['param_alpha'].values
            mean_scores = df_plot['mean_test_score'].values
            std_scores = df_plot['std_test_score'].values
        else:
            alpha_values = cv_results['param_alpha'].values
            mean_scores = cv_results['mean_test_score'].values
            std_scores = cv_results['std_test_score'].values
        
        # 保存搜索结果
        alpha_search_results[name] = {
            'alpha': alpha_values,
            'r2_score': mean_scores,
            'std_score': std_scores,
            'best_alpha': grid_search.best_params_['alpha']
        }
    
    # -----------------------------------------------------------------------
    # 使用最优参数重新训练，记录训练过程（用于绘制收敛曲线）
    # -----------------------------------------------------------------------
    
    # 创建跟踪模型（使用最优参数配置）
    tracking_model = SGDRegressor(
        penalty=best_params_dict[name]['penalty'],
        alpha=best_params_dict[name]['alpha'],
        learning_rate='invscaling',
        eta0=0.01,
        max_iter=1,                # 每次只训练一个epoch
        tol=None,                  # 不使用提前停止
        warm_start=True,           # 允许增量训练（保留上次训练状态）
        random_state=42,
        fit_intercept=True
    )
    
    # 针对ElasticNet：设置L1与L2混合比例
    if name == 'SGD_ElasticNet':
        tracking_model.l1_ratio = best_params_dict[name]['l1_ratio']
    
    # 逐轮训练并记录损失函数值
    losses_log = []         # 对数空间的MSE损失
    losses_original = []    # 原始空间的MSE损失
    epochs = []             # 训练轮数
    
    for epoch in range(100):
        # 训练一个epoch
        tracking_model.fit(X_train, y_train_log)
        y_pred_log = tracking_model.predict(X_train)
        
        # 计算对数空间的损失
        loss_log = mean_squared_error(y_train_log, y_pred_log)
        losses_log.append(loss_log)
        
        # 计算原始空间的损失（反变换后）
        y_pred_original = np.expm1(y_pred_log)
        loss_original = mean_squared_error(y_train_original, y_pred_original)
        losses_original.append(loss_original)
        
        epochs.append(epoch + 1)
    
    # 保存训练历史
    training_history[name] = {
        'epochs': epochs,
        'losses_log': losses_log,
        'losses_original': losses_original
    }

print("="*70)

# ============================================================================
# 7. 可视化超参数搜索过程与训练历史
# ============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# ---------------------------------------------------------------------------
# 左图：超参数 alpha 对 R² 的影响
# ---------------------------------------------------------------------------

# 定义各模型的颜色
colors_map = {'SGD_Ridge (L2)': 'red', 'SGD_Lasso (L1)': 'green', 'SGD_ElasticNet': 'purple'}

# 收集最优点信息（用于后续放大视图）
best_alphas = []
best_r2s = []

for name, data in alpha_search_results.items():
    color = colors_map.get(name, 'blue')
    
    # 绘制主曲线
    ax1.plot(data['alpha'], data['r2_score'], label=name, marker='o', 
             markersize=5, linewidth=2, color=color, alpha=0.7)
    
    # 添加置信区间（±1倍标准差）
    ax1.fill_between(data['alpha'], 
                     data['r2_score'] - data['std_score'], 
                     data['r2_score'] + data['std_score'], 
                     alpha=0.2, color=color)
    
    # 标记最优alpha点（用五角星标记）
    best_alpha = data['best_alpha']
    best_idx = np.where(data['alpha'] == best_alpha)[0][0]
    best_r2 = data['r2_score'][best_idx]
    ax1.scatter([best_alpha], [best_r2], color=color, s=200, zorder=5, 
               marker='*', edgecolors='black', linewidths=2)
    
    best_alphas.append(best_alpha)
    best_r2s.append(best_r2)

# 设置坐标轴和标题
ax1.set_xscale('log')
ax1.set_xlabel('正则化参数 $\\alpha$ (对数尺度)', fontsize=13, fontweight='bold')
ax1.set_ylabel('交叉验证平均 $R^2$ 得分', fontsize=13, fontweight='bold')
ax1.set_title('超参数 $\\alpha$ 对模型性能的影响', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11, loc='lower right')
ax1.grid(False)




# 添加局部放大视图（插入子图，聚焦最优点区域）
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# 确定放大区域的范围（围绕最优点）
alpha_min = min(best_alphas) / 5
alpha_max = max(best_alphas) * 5
r2_min = min(best_r2s) - 0.02
r2_max = max(best_r2s) + 0.02

# 创建嵌入的放大子图（位于左下角）
axins = inset_axes(ax1, width="40%", height="35%", loc='lower left',
                   bbox_to_anchor=(0.05, 0.05, 1, 1), bbox_transform=ax1.transAxes)

# 在放大子图中绘制相同的数据
for name, data in alpha_search_results.items():
    color = colors_map.get(name, 'blue')
    axins.plot(data['alpha'], data['r2_score'], marker='o', 
               markersize=4, linewidth=1.5, color=color, alpha=0.7)
    axins.fill_between(data['alpha'], 
                       data['r2_score'] - data['std_score'], 
                       data['r2_score'] + data['std_score'], 
                       alpha=0.2, color=color)
    
    # 在放大视图中标记最优点
    best_alpha = data['best_alpha']
    best_idx = np.where(data['alpha'] == best_alpha)[0][0]
    best_r2 = data['r2_score'][best_idx]
    axins.scatter([best_alpha], [best_r2], color=color, s=100, zorder=5, 
                  marker='*', edgecolors='black', linewidths=1.5)

# 设置放大区域的坐标范围
axins.set_xlim(alpha_min, alpha_max)
axins.set_ylim(r2_min, r2_max)
axins.set_xscale('log')
axins.grid(False)
axins.tick_params(labelsize=8)

# 设置对数坐标轴格式化器
axins.xaxis.set_major_formatter(LogFormatter(labelOnlyBase=False, linthresh=1e-6))
ax1.xaxis.set_major_formatter(LogFormatter(labelOnlyBase=False, linthresh=1e-6))

# 绘制连接线，指示放大区域
mark_inset(ax1, axins, loc1=2, loc2=4, fc="none", ec="0.5", linestyle='--', linewidth=1)

# ---------------------------------------------------------------------------
# 右图：训练过程中损失函数的变化（原始空间）
# ---------------------------------------------------------------------------

for name, data in training_history.items():
    color = colors_map.get(name, 'blue')
    ax2.plot(data['epochs'], data['losses_original'], label=name, 
             linewidth=2, color=color, alpha=0.7)

# 设置坐标轴和标题
ax2.set_xlabel('训练轮数 (Epochs)', fontsize=13, fontweight='bold')
ax2.set_ylabel('均方误差 (MSE) - 原始空间', fontsize=13, fontweight='bold')
ax2.set_title('训练过程中损失函数的变化\n（在实际径流量空间计算）', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11, loc='upper right')
ax2.set_yscale('log')  # 使用对数尺度更清晰地观察收敛过程
ax2.grid(False)

# 保存并显示图形
plt.tight_layout()
plt.savefig('超参数寻找与训练过程可视化.svg', format='svg', bbox_inches='tight')
print("\n[成功] 超参数寻找与训练过程可视化已保存为 '超参数寻找与训练过程可视化.svg' 文件")
plt.show()


# ============================================================================
# 8. 评估四种模型在训练集和测试集上的性能
# ============================================================================

train_results = {}  # 存储训练集评估结果
test_results = {}   # 存储测试集评估结果

# 遍历所有模型进行评估
for name, model in best_models.items():
    
    # -----------------------------------------------------------------------
    # 训练集评估
    # -----------------------------------------------------------------------
    y_train_pred_log = model.predict(X_train)
    y_train_pred = np.expm1(y_train_pred_log)  # 反变换到原始空间
    
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
    
    # -----------------------------------------------------------------------
    # 测试集评估
    # -----------------------------------------------------------------------
    y_test_pred_log = model.predict(X_test)
    y_test_pred = np.expm1(y_test_pred_log)  # 反变换到原始空间
    
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

# ---------------------------------------------------------------------------
# 输出训练集评估结果
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("模型评估结果 - 训练集 (原始空间)")
print("="*70)
print(f"{'模型':<20} {'R²':<10} {'RMSE':<10} {'MAE':<10} {'NSE':<10}")
print("-" * 70)

for name in best_models.keys():
    r2 = train_results[name]['R^2']
    rmse = train_results[name]['RMSE']
    mae = train_results[name]['MAE']
    nse = train_results[name]['NSE']
    print(f"{name:<20} {r2:<10.4f} {rmse:<10.4f} {mae:<10.4f} {nse:<10.4f}")

print("="*70)

# ---------------------------------------------------------------------------
# 输出测试集评估结果
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("模型评估结果 - 测试集 (原始空间)")
print("="*70)
print(f"{'模型':<20} {'R²':<10} {'RMSE':<10} {'MAE':<10} {'NSE':<10}")
print("-" * 70)

for name in best_models.keys():
    r2 = test_results[name]['R^2']
    rmse = test_results[name]['RMSE']
    mae = test_results[name]['MAE']
    nse = test_results[name]['NSE']
    print(f"{name:<20} {r2:<10.4f} {rmse:<10.4f} {mae:<10.4f} {nse:<10.4f}")

print("="*70)

# ============================================================================
# 9. 梯度下降过程的详细可视化
# ============================================================================

print("\n" + "="*70)
print("生成梯度下降过程的详细可视化...")
print("="*70)

# 定义四种模型的颜色（用于后续所有可视化）
colors_map = {'SGD_OLS (None)': '#1f77b4', 'SGD_Ridge (L2)': '#ff7f0e', 
              'SGD_Lasso (L1)': '#2ca02c', 'SGD_ElasticNet': '#d62728'}

# ---------------------------------------------------------------------------
# 9.1 学习曲线（训练集 vs 验证集性能随epoch变化）
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, (name, model) in enumerate(best_models.items()):
    
    # 划分训练集和验证集（用于学习曲线绘制）
    from sklearn.model_selection import train_test_split
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train_log, test_size=0.2, random_state=42
    )
    
    train_scores = []   # 训练集R²得分
    val_scores = []     # 验证集R²得分
    epochs_list = range(1, 101)  # 训练100个epochs
    
    # 创建临时模型用于增量训练
    temp_model = SGDRegressor(
        penalty=best_params_dict[name]['penalty'],
        alpha=best_params_dict[name]['alpha'],
        learning_rate='invscaling',
        eta0=0.01,
        max_iter=1,        # 每次训练一个epoch
        tol=None,
        warm_start=True,   # 保留之前的训练状态
        random_state=42,
        fit_intercept=True
    )
    
    # 针对ElasticNet：设置L1与L2混合比例
    if name == 'SGD_ElasticNet':
        temp_model.l1_ratio = best_params_dict[name]['l1_ratio']
    
    # 逐轮训练并记录R²得分
    for epoch in epochs_list:
        temp_model.fit(X_train_split, y_train_split)
        
        # 计算训练集R²
        train_pred = temp_model.predict(X_train_split)
        train_r2 = r2_score(y_train_split, train_pred)
        train_scores.append(train_r2)
        
        # 计算验证集R²
        val_pred = temp_model.predict(X_val_split)
        val_r2 = r2_score(y_val_split, val_pred)
        val_scores.append(val_r2)
    
    # 绘制学习曲线
    axes[idx].plot(epochs_list, train_scores, label='训练集 $R^2$', 
                   linewidth=2, color=colors_map.get(name, 'blue'), alpha=0.8)
    axes[idx].plot(epochs_list, val_scores, label='验证集 $R^2$', 
                   linewidth=2, linestyle='--', color=colors_map.get(name, 'blue'), alpha=0.8)
    axes[idx].set_xlabel('训练轮数 (Epochs)', fontsize=11, fontweight='bold')
    axes[idx].set_ylabel('$R^2$ 得分', fontsize=11, fontweight='bold')
    axes[idx].set_title(f'{name}\n学习曲线', fontsize=12, fontweight='bold')
    axes[idx].legend(fontsize=10, loc='lower right')
    axes[idx].grid(False)
    
    # 为每个子图添加放大视图（右上角，聚焦第3到第10个epochs的关键区域）
    axins = inset_axes(axes[idx], width="40%", height="35%", loc='upper right',
                       bbox_to_anchor=(0.0, 0.0, 0.95, 0.95), bbox_transform=axes[idx].transAxes)

    # 在放大子图中绘制第3到第10个epochs的数据
    epochs_subset = list(epochs_list)[2:10]  # 索引2-9，对应第3-10个epoch
    train_subset = train_scores[2:10]
    val_subset = val_scores[2:10]
    
    axins.plot(epochs_subset, train_subset, 
               linewidth=1.5, color=colors_map.get(name, 'blue'), alpha=0.8)
    axins.plot(epochs_subset, val_subset, 
               linewidth=1.5, linestyle='--', color=colors_map.get(name, 'blue'), alpha=0.8)
    
    # 设置放大区域的坐标范围
    axins.set_xlim(2, 10)
    y_min_subset = min(min(train_subset), min(val_subset)) - 0.02
    y_max_subset = max(max(train_subset), max(val_subset)) + 0.02
    axins.set_ylim(y_min_subset, y_max_subset)
    axins.grid(False)
    axins.tick_params(labelsize=8)
    axins.set_xlabel('Epochs', fontsize=8)
    axins.set_ylabel('$R^2$', fontsize=8)
    
    # 绘制连接线，指示放大区域
    mark_inset(axes[idx], axins, loc1=2, loc2=4, fc="none", ec="0.5", linestyle='--', linewidth=1)

# 设置总标题并保存
plt.suptitle('四种SGD模型的学习曲线对比\n(观察过拟合/欠拟合)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('SGD模型学习曲线.svg', format='svg', bbox_inches='tight')
print("[成功] 学习曲线已保存为 'SGD模型学习曲线.svg' 文件")
plt.show()

# ---------------------------------------------------------------------------
# 9.2 梯度下降收敛速度对比（对数空间和原始空间的损失）
# ---------------------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

for name, data in training_history.items():
    color = colors_map.get(name, 'blue')
    
    # 左图：对数空间的MSE损失
    ax1.plot(data['epochs'], data['losses_log'], label=name, 
             linewidth=2, color=color, alpha=0.7)
    
    # 右图：原始空间的MSE损失
    ax2.plot(data['epochs'], data['losses_original'], label=name, 
             linewidth=2, color=color, alpha=0.7)

# 左图设置
ax1.set_xlabel('训练轮数 (Epochs)', fontsize=13, fontweight='bold')
ax1.set_ylabel('MSE - 对数空间', fontsize=13, fontweight='bold')
ax1.set_title('梯度下降收敛过程\n(对数空间)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(False)
ax1.set_yscale('log')  # 使用对数尺度更好地观察收敛

# 右图设置
ax2.set_xlabel('训练轮数 (Epochs)', fontsize=13, fontweight='bold')
ax2.set_ylabel('MSE - 原始空间', fontsize=13, fontweight='bold')
ax2.set_title('梯度下降收敛过程\n(原始径流量空间)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(False)
ax2.set_yscale('log')  # 使用对数尺度更好地观察收敛

plt.tight_layout()
plt.savefig('SGD梯度下降收敛对比.svg', format='svg', bbox_inches='tight')
print("[成功] 梯度下降收敛图已保存为 'SGD梯度下降收敛对比.svg' 文件")
plt.show()

# ---------------------------------------------------------------------------
# 9.3 模型性能条形图对比
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

metrics = ['R^2', 'RMSE', 'MAE', 'NSE']  # 四个评估指标
model_names = list(best_models.keys())
x_pos = np.arange(len(model_names))
width = 0.35  # 柱子宽度

for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    
    # 提取训练集和测试集的指标值
    train_values = [train_results[name][metric] for name in model_names]
    test_values = [test_results[name][metric] for name in model_names]
    
    # 绘制条形图
    bars1 = ax.bar(x_pos - width/2, train_values, width, label='训练集', alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, test_values, width, label='测试集', alpha=0.8)
    
    # 设置坐标轴和标题
    ax.set_xlabel('模型', fontsize=11, fontweight='bold')
    ax.set_ylabel(metric, fontsize=11, fontweight='bold')
    ax.set_title(f'{metric} 对比', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([name.replace(' (', '\n(') for name in model_names], 
                       fontsize=9, rotation=0)
    ax.legend(fontsize=10)
    ax.grid(False)
    
    # 在柱子上方添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

# 设置总标题并保存
plt.suptitle('四种SGD模型性能指标对比\n(训练集 vs 测试集)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('SGD模型性能条形图对比.svg', format='svg', bbox_inches='tight')
print("[成功] 性能条形图已保存为 'SGD模型性能条形图对比.svg' 文件")
plt.show()

# ============================================================================
# 10. 为最佳模型（SGD_OLS）绘制详细可视化
# ============================================================================

print("\n" + "="*70)
print("开始为最佳模型 SGD_OLS (None) 生成详细可视化...")
print("="*70)

best_model_name = 'SGD_OLS (None)'
best_model = best_models[best_model_name]

# 获取训练集和测试集的预测结果
y_train_pred_log = best_model.predict(X_train)
y_train_pred = np.expm1(y_train_pred_log)  # 反变换到原始空间
y_test_pred_log = best_model.predict(X_test)
y_test_pred = np.expm1(y_test_pred_log)    # 反变换到原始空间

# ---------------------------------------------------------------------------
# 10.1 预测结果散点图（实际值 vs 预测值）
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 训练集散点图
axes[0].scatter(y_train_original, y_train_pred, alpha=0.6, edgecolors='k', linewidths=0.5, s=50)
axes[0].set_xlabel('实际径流量', fontsize=13, fontweight='bold')
axes[0].set_ylabel('预测径流量', fontsize=13, fontweight='bold')
train_r2 = train_results[best_model_name]['R^2']
train_rmse = train_results[best_model_name]['RMSE']
axes[0].set_title(f'训练集: 实际值 vs 预测值\n$R^2$={train_r2:.4f}, RMSE={train_rmse:.2f}', 
                  fontsize=14, fontweight='bold')
axes[0].grid(False)

# 测试集散点图
axes[1].scatter(y_test_original, y_test_pred, alpha=0.6, color='orange', 
                edgecolors='k', linewidths=0.5, s=50)
axes[1].set_xlabel('实际径流量', fontsize=13, fontweight='bold')
axes[1].set_ylabel('预测径流量', fontsize=13, fontweight='bold')
test_r2 = test_results[best_model_name]['R^2']
test_rmse = test_results[best_model_name]['RMSE']
axes[1].set_title(f'测试集: 实际值 vs 预测值\n$R^2$={test_r2:.4f}, RMSE={test_rmse:.2f}', 
                  fontsize=14, fontweight='bold')
axes[1].grid(False)

# 添加 X=Y 参考线（在两个子图上），范围基于所有实际与预测值的最小/最大值
xy_min = min(np.min(y_train_original), np.min(y_train_pred), np.min(y_test_original), np.min(y_test_pred))
xy_max = max(np.max(y_train_original), np.max(y_train_pred), np.max(y_test_original), np.max(y_test_pred))
for ax in axes:
    ax.plot([xy_min, xy_max], [xy_min, xy_max], color='gray', linestyle='--', linewidth=1.5, label='y = x')
    ax.legend(fontsize=10)

# 保存并显示
plt.tight_layout()
plt.savefig('SGD_OLS预测散点图.svg', format='svg', bbox_inches='tight')
print("\n[成功] 预测散点图已保存为 'SGD_OLS预测散点图.svg' 文件")
plt.show()

# ---------------------------------------------------------------------------
# 10.2 残差分析图（原10.3节）
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 计算残差
train_residuals = y_train_original - y_train_pred
test_residuals = y_test_original - y_test_pred

# 训练集残差散点图
axes[0, 0].scatter(y_train_pred, train_residuals, alpha=0.6, edgecolors='k', linewidths=0.5, s=40)
axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)  # 添加零线
axes[0, 0].set_xlabel('预测径流量', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('残差（实际值 - 预测值）', fontsize=12, fontweight='bold')
axes[0, 0].set_title('训练集残差图', fontsize=13, fontweight='bold')
axes[0, 0].grid(False)

# 测试集残差散点图
axes[0, 1].scatter(y_test_pred, test_residuals, alpha=0.6, color='orange', 
                   edgecolors='k', linewidths=0.5, s=40)
axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)  # 添加零线
axes[0, 1].set_xlabel('预测径流量', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('残差（实际值 - 预测值）', fontsize=12, fontweight='bold')
axes[0, 1].set_title('测试集残差图', fontsize=13, fontweight='bold')
axes[0, 1].grid(False)

# 训练集残差直方图
axes[1, 0].hist(train_residuals, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
axes[1, 0].set_xlabel('残差', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('频数', fontsize=12, fontweight='bold')
axes[1, 0].set_title(f'训练集残差分布\n均值={train_residuals.mean():.2f}, 标准差={train_residuals.std():.2f}', 
                     fontsize=13, fontweight='bold')
axes[1, 0].grid(False)

# 测试集残差直方图
axes[1, 1].hist(test_residuals, bins=30, edgecolor='black', alpha=0.7, color='lightcoral')
axes[1, 1].set_xlabel('残差', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('频数', fontsize=12, fontweight='bold')
axes[1, 1].set_title(f'测试集残差分布\n均值={test_residuals.mean():.2f}, 标准差={test_residuals.std():.2f}', 
                     fontsize=13, fontweight='bold')
axes[1, 1].grid(False)

# 保存并显示
plt.tight_layout()
plt.savefig('SGD_OLS残差分析图.svg', format='svg', bbox_inches='tight')
print("[成功] 残差分析图已保存为 'SGD_OLS残差分析图.svg' 文件")
plt.show()

# ---------------------------------------------------------------------------
# 10.3 四模型对比的预测散点图（测试集，原10.4节）
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for idx, (name, model) in enumerate(best_models.items()):
    # 获取测试集预测结果
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)  # 反变换到原始空间
    
    # 绘制散点图
    axes[idx].scatter(y_test_original, y_pred, alpha=0.6, 
                     color=colors_list[idx], edgecolors='k', linewidths=0.5, s=50)
    
    # 设置坐标轴和标题
    axes[idx].set_xlabel('实际径流量', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('预测径流量', fontsize=12, fontweight='bold')
    
    r2 = test_results[name]['R^2']
    rmse = test_results[name]['RMSE']
    axes[idx].set_title(f'{name}\n$R^2$={r2:.4f}, RMSE={rmse:.2f}', 
                       fontsize=13, fontweight='bold')
    axes[idx].grid(False)

# 设置总标题并保存
plt.suptitle('四种SGD模型测试集预测对比', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('四种SGD模型预测散点图对比.svg', format='svg', bbox_inches='tight')
print("[成功] 四模型预测散点图对比已保存为 '四种SGD模型预测散点图对比.svg' 文件")
plt.show()

# ============================================================================
# 程序执行完成
# ============================================================================

print("\n" + "="*70)
print("所有可视化图表生成完成！")
print("="*70)