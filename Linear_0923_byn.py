import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_preprocessed_data():
    """加载预处理后的数据"""
    try:
        df = pd.read_csv('preprocessed_data.csv', encoding='utf-8-sig')
        print("成功加载预处理数据")
        return df
    except FileNotFoundError:
        print("未找到预处理数据，请先运行 dataPreprocessing_0926_byn.py")
        return None

def build_linear_models(df):
    """构建多种线性回归模型"""
    
    # 定义不同的模型配置
    model_configs = {
        'model1_original': {
            'X': df[['蒸发量', '降雨量']],
            'y': df['径流量'],
            'name': '原始数据线性回归',
            'features': ['蒸发量', '降雨量']
        },
        'model2_log_transformed': {
            'X': df[['蒸发量_std', '降雨量_log']],
            'y': df['径流量_log'],
            'name': '对数变换线性回归',
            'features': ['蒸发量_std', '降雨量_log']
        },
        'model3_rainfall_only': {
            'X': df[['降雨量_log']].values.reshape(-1, 1),
            'y': df['径流量_log'],
            'name': '仅降雨量线性回归',
            'features': ['降雨量_log']
        }
    }
    
    results = {}
    
    for model_key, config in model_configs.items():
        print(f"\n=== {config['name']} ===")
        
        # 数据准备
        X = config['X']
        y = config['y']
        
        # 分割训练测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 构建模型
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # 预测
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # 评估指标
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        # 交叉验证
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        
        # 保存结果
        results[model_key] = {
            'model': model,
            'config': config,
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'y_train_pred': y_train_pred, 'y_test_pred': y_test_pred,
            'metrics': {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
        }
        
        # 打印结果
        print(f"训练集 R²: {train_r2:.4f}")
        print(f"测试集 R²: {test_r2:.4f}")
        print(f"交叉验证 R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print(f"测试集 RMSE: {test_rmse:.4f}")
        print(f"测试集 MAE: {test_mae:.4f}")
        
        # 输出回归系数
        if len(config['features']) > 1:
            print("回归系数:")
            for feature, coef in zip(config['features'], model.coef_):
                print(f"  {feature}: {coef:.4f}")
        else:
            print(f"回归系数: {model.coef_[0]:.4f}")
        print(f"截距: {model.intercept_:.4f}")
    
    return results

def visualize_results(results):
    """可视化模型结果"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (model_key, result) in enumerate(results.items()):
        # 预测 vs 实际值散点图
        ax = axes[i]
        
        # 训练集
        ax.scatter(result['y_train'], result['y_train_pred'], 
                  alpha=0.6, color='blue', label='训练集', s=30)
        # 测试集
        ax.scatter(result['y_test'], result['y_test_pred'], 
                  alpha=0.8, color='red', label='测试集', s=30)
        
        # 完美预测线
        min_val = min(result['y_train'].min(), result['y_test'].min())
        max_val = max(result['y_train'].max(), result['y_test'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        ax.set_xlabel('实际值')
        ax.set_ylabel('预测值')
        ax.set_title(f"{result['config']['name']}\\nR² = {result['metrics']['test_r2']:.3f}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for i in range(len(results), 3):
        if i < 3:
            axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def compare_models(results):
    """比较模型性能"""
    print("\\n" + "="*60)
    print("模型性能比较")
    print("="*60)
    
    comparison_data = []
    for model_key, result in results.items():
        comparison_data.append({
            '模型': result['config']['name'],
            '测试集R²': f"{result['metrics']['test_r2']:.4f}",
            '交叉验证R²': f"{result['metrics']['cv_mean']:.4f}±{result['metrics']['cv_std']:.4f}",
            'RMSE': f"{result['metrics']['test_rmse']:.4f}",
            'MAE': f"{result['metrics']['test_mae']:.4f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # 找出最佳模型
    best_model_key = max(results.keys(), 
                        key=lambda k: results[k]['metrics']['test_r2'])
    best_model = results[best_model_key]
    
    print(f"\\n🏆 最佳模型: {best_model['config']['name']}")
    print(f"   测试集 R²: {best_model['metrics']['test_r2']:.4f}")
    print(f"   交叉验证 R²: {best_model['metrics']['cv_mean']:.4f}")

if __name__ == "__main__":
    # 加载数据
    print("开始线性回归建模...")
    df = load_preprocessed_data()
    
    if df is not None:
        # 构建模型
        results = build_linear_models(df)
        
        # 可视化结果
        print("\\n生成模型结果图...")
        visualize_results(results)
        
        # 比较模型
        compare_models(results)
        
        print("\\n线性回归建模完成！")
    else:
        print("请先运行数据预处理脚本！")