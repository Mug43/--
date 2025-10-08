# 径流预测 - 线性回归与SGD模型应用

## 📋 项目简介

本项目基于青山水库的径流数据，使用多种线性回归模型（普通线性回归、Lasso、Ridge及其融合模型）和随机梯度下降（SGD）回归方法进行径流预测研究。通过超参数优化、模型对比分析和可视化评估，探索不同正则化方法在径流预测中的应用效果。

## 🎯 主要功能

- **数据预处理**: 数据清洗、格式转换和标准化
- **模型训练**: SGD回归模型（OLS、Ridge、Lasso、ElasticNet）
- **超参数优化**: 网格搜索自动寻找最优参数
- **性能评估**: 多指标评估（R²、RMSE、MAE、NSE）
- **结果可视化**: 训练过程、预测结果、残差分析等图表
- **文档输出**: 完整的研究报告（Word/PDF格式）

## 📂 项目结构

```
.
├── dataChecking_0923_byn.py              # 数据检查脚本
├── dataConversion_1001_byn.py            # 数据转换脚本
├── newSGDregressor_1002_byn.py           # SGD回归模型主程序（最终版本）
├── qingshandataforregression.xlsx        # 原始径流数据
├── X_train.csv                           # 训练集特征数据
├── X_test.csv                            # 测试集特征数据
├── y_train.csv                           # 训练集目标数据
├── y_test.csv                            # 测试集目标数据
├── 普通线性回归、Lasso与Ridge及其融合模型在径流预测中的应用与评估.docx
├── 普通线性回归、Lasso与Ridge及其融合模型在径流预测中的应用与评估.pdf
└── README.md                             # 项目说明文档
```

## 🚀 快速开始

### 环境要求

- Python 3.x
- 必需的Python包：
  ```
  pandas
  numpy
  matplotlib
  scikit-learn
  openpyxl
  ```

### 安装依赖

```bash
pip install pandas numpy matplotlib scikit-learn openpyxl
```

### 运行程序

1. **数据预处理**（如需重新处理数据）：
   ```bash
   python dataChecking_0923_byn.py      # 检查数据质量
   python dataConversion_1001_byn.py    # 转换数据格式
   ```

2. **模型训练与评估**：
   ```bash
   python newSGDregressor_1002_byn.py   # 运行SGD回归模型分析
   ```

## 📊 模型说明

本项目实现了四种SGD回归模型：

| 模型 | 正则化类型 | 特点 |
|------|-----------|------|
| **SGD_OLS** | 无正则化 | 标准线性回归，无惩罚项 |
| **SGD_Ridge** | L2正则化 | 收缩系数，防止过拟合 |
| **SGD_Lasso** | L1正则化 | 特征选择，稀疏解 |
| **SGD_ElasticNet** | L1+L2混合 | 结合两者优点 |

## 📈 评估指标

- **R² (决定系数)**: 模型拟合优度
- **RMSE (均方根误差)**: 预测误差的标准差
- **MAE (平均绝对误差)**: 预测误差的平均值
- **NSE (纳什效率系数)**: 水文模拟常用指标

## 📝 输出结果

程序运行后会自动生成：
- 模型性能对比图表
- 预测值与实际值对比散点图
- 残差分析图
- 训练过程收敛曲线
- 超参数搜索结果

## 👨‍💻 作者

**byn**

## 📅 更新日志

- **2024-10-02**: 完成SGD模型主程序开发
- **2024-10-01**: 数据转换脚本优化
- **2024-09-23**: 初始版本，数据检查功能

## 📄 许可证

本项目仅供学习和研究使用。

## 🔗 相关文档

详细的理论分析和实验结果请参考：
- `普通线性回归、Lasso与Ridge及其融合模型在径流预测中的应用与评估.pdf`

---

**注意**: 运行前请确保数据文件路径正确，所有CSV文件应与Python脚本位于同一目录下。
