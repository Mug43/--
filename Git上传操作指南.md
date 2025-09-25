# Git上传操作指南

## 📋 项目信息
- **项目名称**: 水文数据回归分析
- **仓库地址**: https://github.com/Mug43/--.git
- **分支**: main
- **创建日期**: 2025年9月25日

## 🚀 完整上传流程

### 1. 初始化Git仓库
```bash
# 初始化Git仓库（如果还未初始化）
git init

# 查看仓库状态
git status
```

### 2. 配置Git用户信息
```bash
# 设置用户名和邮箱（全局配置）
git config --global user.name "包一宁"
git config --global user.email "2775610414@qq.com"

# 查看配置
git config --global --list
```

### 3. 配置代理（重要！）
```bash
# 配置HTTP代理（端口7897）
git config --global http.proxy http://127.0.0.1:7897
git config --global https.proxy http://127.0.0.1:7897

# 验证代理配置
git config --global --get http.proxy
git config --global --get https.proxy
```

### 4. 添加文件到暂存区
```bash
# 添加所有Python文件和Excel文件
git add *.py *.xlsx

# 或者添加所有文件
git add .

# 查看暂存区状态
git status
```

### 5. 提交更改
```bash
# 提交到本地仓库
git commit -m "更新数据检查代码"

# 查看提交历史
git log --oneline
```

### 6. 连接远程仓库
```bash
# 添加远程仓库（HTTPS方式）
git remote add origin https://github.com/Mug43/--.git

# 查看远程仓库配置
git remote -v
```

### 7. 推送到远程仓库
```bash
# 推送到GitHub
git push origin main

# 如果是第一次推送，可能需要设置上游分支
git push -u origin main
```

## 🔧 网络问题解决方案

### 代理配置
```bash
# 设置代理
git config --global http.proxy http://127.0.0.1:7897
git config --global https.proxy http://127.0.0.1:7897

# 清除代理（如果不需要）
git config --global --unset http.proxy
git config --global --unset https.proxy
```

### HTTP优化设置
```bash
# 增加缓冲区大小
git config --global http.postBuffer 524288000

# 设置超时时间
git config --global http.timeout 60

# 设置HTTP版本
git config --global http.version HTTP/1.1
```

### DNS问题解决
```bash
# 清除DNS缓存
ipconfig /flushdns

# 测试GitHub连接
ping github.com

# 使用Google DNS查询GitHub IP
nslookup github.com 8.8.8.8
```

## 📁 已上传文件清单

### Python脚本文件
- `dataChecking_0923_byn.py` - 数据质量检查和探索性分析脚本
- `Liner_0923_byn.py` - 线性回归模型脚本（待开发）
- `Ridge_0923_byn.py` - Ridge回归模型脚本（待开发）
- `Lasso_0923_byn.py` - Lasso回归模型脚本（待开发）
- `Coupling_0924_byn.py` - 耦合模型脚本（待开发）

### 数据文件
- `qingshandataforregression.xlsx` - 青山水文数据（蒸发量、降雨量、径流量）

### 待上传文件
- `Origin/` - 原始文件夹
- `图/` - 图表文件夹

## 🔄 日常更新操作

### 更新现有文件
```bash
# 1. 修改文件后查看状态
git status

# 2. 添加修改的文件
git add 文件名.py

# 3. 提交更改
git commit -m "更新说明"

# 4. 推送到远程
git push origin main
```

### 添加新文件
```bash
# 1. 添加新文件
git add 新文件名.py

# 2. 提交
git commit -m "添加新文件：文件描述"

# 3. 推送
git push origin main
```

## 📊 项目结构
```
径流/
├── dataChecking_0923_byn.py      # 数据检查脚本
├── Liner_0923_byn.py             # 线性回归
├── Ridge_0923_byn.py             # Ridge回归
├── Lasso_0923_byn.py             # Lasso回归
├── Coupling_0924_byn.py          # 耦合模型
├── qingshandataforregression.xlsx # 数据文件
├── Origin/                       # 原始文件
├── 图/                          # 图表文件
└── Git上传操作指南.md            # 本文档
```

## ⚠️ 注意事项

### 1. 代理设置
- 确保代理软件正在运行
- 代理端口为7897
- 如果代理端口变更，需要重新配置Git

### 2. 文件管理
- 避免上传敏感信息
- 大文件建议使用Git LFS
- 定期清理不需要的文件

### 3. 提交规范
- 提交信息要清晰明确
- 单次提交包含相关的修改
- 避免提交临时文件

## 🚨 常见问题解决

### 问题1: 推送失败 - 连接超时
```bash
# 解决方案：检查代理设置
git config --global --get http.proxy
# 如果为空，重新设置代理
git config --global http.proxy http://127.0.0.1:7897
```

### 问题2: GitHub域名解析到127.0.0.1
```bash
# 解决方案：清除DNS缓存
ipconfig /flushdns
# 或检查hosts文件是否有GitHub条目
notepad C:\Windows\System32\drivers\etc\hosts
```

### 问题3: 认证失败
```bash
# 解决方案：确认GitHub用户名和令牌
# 或使用GitHub Desktop进行认证
```

## 📞 技术支持
- GitHub仓库：https://github.com/Mug43/--.git
- 最后更新：2025年9月25日
- 操作系统：Windows + PowerShell
- Git版本：建议使用最新版本

---
*本文档记录了水文数据回归分析项目的完整Git操作流程，包含网络配置和问题解决方案。*