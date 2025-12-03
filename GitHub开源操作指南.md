# ASA-DETR GitHub开源操作指南

## 📋 目录

1. [准备工作](#准备工作)
2. [创建GitHub仓库](#创建github仓库)
3. [上传项目到GitHub](#上传项目到github)
4. [完善项目内容](#完善项目内容)
5. [发布Release版本](#发布release版本)
6. [推广项目](#推广项目)

---

## 1️⃣ 准备工作

### 1.1 安装Git

如果还没有安装Git，请先安装：

**Windows:**
- 下载：https://git-scm.com/download/win
- 安装后打开Git Bash

**验证安装:**
```bash
git --version
```

### 1.2 配置Git

```bash
# 配置用户名和邮箱
git config --global user.name "你的名字"
git config --global user.email "你的邮箱@example.com"

# 查看配置
git config --list
```

### 1.3 注册GitHub账号

如果还没有GitHub账号：
1. 访问 https://github.com
2. 点击"Sign up"注册
3. 验证邮箱

---

## 2️⃣ 创建GitHub仓库

### 方法一：通过GitHub网页创建

1. **登录GitHub**
   - 访问 https://github.com
   - 登录你的账号

2. **创建新仓库**
   - 点击右上角的 "+" 号
   - 选择 "New repository"

3. **填写仓库信息**
   ```
   Repository name: ASA-DETR
   Description: Adaptive Sparse Attention Enhanced RT-DETR for Remote Sensing Landslide Detection
   
   ☑ Public (公开仓库)
   ☐ Add a README file (不勾选，我们已经有了)
   ☐ Add .gitignore (不勾选，我们已经有了)
   ☐ Choose a license (不勾选，我们已经有了)
   ```

4. **点击 "Create repository"**

5. **记录仓库地址**
   ```
   https://github.com/你的用户名/ASA-DETR.git
   ```

---

## 3️⃣ 上传项目到GitHub

### 3.1 打开命令行

**Windows:**
- 按 `Win + R`
- 输入 `cmd` 或 `powershell`
- 回车

或者：
- 在ASA-DETR文件夹中，按住Shift键，右键点击空白处
- 选择"在此处打开PowerShell窗口"或"在此处打开命令窗口"

### 3.2 进入项目目录

```bash
# 进入ASA-DETR文件夹
cd "C:\Users\宋汉雨\Downloads\新建文件夹 (20)\ASA-DETR"

# 确认当前目录
pwd
```

### 3.3 初始化Git仓库

```bash
# 初始化Git仓库
git init

# 查看状态
git status
```

### 3.4 添加所有文件

```bash
# 添加所有文件到暂存区
git add .

# 查看添加的文件
git status
```

### 3.5 提交到本地仓库

```bash
# 提交文件
git commit -m "Initial commit: ASA-DETR project for landslide detection"

# 查看提交历史
git log --oneline
```

### 3.6 连接到GitHub远程仓库

```bash
# 添加远程仓库（替换为你的GitHub用户名）
git remote add origin https://github.com/你的用户名/ASA-DETR.git

# 查看远程仓库
git remote -v
```

### 3.7 推送到GitHub

```bash
# 推送到GitHub（首次推送）
git push -u origin master

# 或者如果默认分支是main
git branch -M main
git push -u origin main
```

**如果需要输入用户名和密码：**
- 用户名：你的GitHub用户名
- 密码：使用Personal Access Token（不是GitHub密码）

**创建Personal Access Token：**
1. GitHub网页 → 右上角头像 → Settings
2. 左侧菜单 → Developer settings
3. Personal access tokens → Tokens (classic)
4. Generate new token
5. 勾选 `repo` 权限
6. 生成并复制token（只显示一次，请保存）

---

## 4️⃣ 完善项目内容

### 4.1 整合现有代码

将您的RTDETR-20251028文件夹中的实际代码整合到ASA-DETR项目中：

```bash
# 1. 复制模型实现代码
# 将RTDETR-20251028/RTDETR-main/ultralytics/中的相关代码
# 复制到ASA-DETR/models/对应位置

# 2. 复制训练脚本
# 将实际的train.py、val.py等替换模板文件

# 3. 复制配置文件
# 将实际使用的配置文件放入configs/目录
```

### 4.2 添加数据集（可选）

如果数据集较小且允许公开：
```bash
# 创建数据集示例
mkdir -p datasets/RSLD-2K/images/train
mkdir -p datasets/RSLD-2K/labels/train

# 添加几张示例图片和标注
```

如果数据集较大：
- 在README.md中提供下载链接
- 使用百度网盘、Google Drive等

### 4.3 添加模型权重

```bash
# 创建weights目录
mkdir -p weights

# 上传预训练权重到云盘
# 在weights/README.md中添加下载链接
```

### 4.4 添加实验结果图片

```bash
# 创建图片目录
mkdir -p docs/images

# 添加以下图片：
# - architecture.png (网络结构图)
# - detection_results.png (检测结果)
# - heatmap_comparison.png (热力图对比)
# - performance_chart.png (性能对比图)
```

### 4.5 提交更新

```bash
# 添加新文件
git add .

# 提交
git commit -m "Add complete implementation and results"

# 推送到GitHub
git push
```

---

## 5️⃣ 发布Release版本

### 5.1 创建Release

1. **进入GitHub仓库页面**
   - https://github.com/你的用户名/ASA-DETR

2. **点击右侧的 "Releases"**

3. **点击 "Create a new release"**

4. **填写Release信息**
   ```
   Tag version: v1.0.0
   Release title: ASA-DETR v1.0.0 - Initial Release
   
   Description:
   ## ASA-DETR v1.0.0
   
   ### 主要特性
   - ✅ LASAB轻量级自适应稀疏注意力主干网络
   - ✅ CSPMFOK多尺度频率感知全向卷积模块
   - ✅ HMSAF层次化多尺度注意力融合模块
   
   ### 性能指标
   - mAP@0.5: 73.2%
   - mAP@0.5:0.95: 52.5%
   - 参数量: 18.3M
   
   ### 下载
   - 模型权重: [链接]
   - RSLD-2K数据集: [链接]
   ```

5. **上传文件（可选）**
   - 上传模型权重文件
   - 上传数据集压缩包

6. **点击 "Publish release"**

---

## 6️⃣ 推广项目

### 6.1 完善README

确保README.md包含：
- ✅ 项目徽章（Python、PyTorch、License）
- ✅ 清晰的项目描述
- ✅ 快速开始指南
- ✅ 性能对比表格
- ✅ 可视化结果
- ✅ 引用信息

### 6.2 添加Topics标签

在GitHub仓库页面：
1. 点击右侧的 "⚙️ Settings"旁边的齿轮图标
2. 添加相关标签：
   ```
   deep-learning
   object-detection
   remote-sensing
   landslide-detection
   pytorch
   computer-vision
   rt-detr
   transformer
   ```

### 6.3 创建GitHub Pages（可选）

展示项目文档和演示：
1. Settings → Pages
2. Source: Deploy from a branch
3. Branch: main → /docs
4. Save

### 6.4 分享项目

**学术社区：**
- 在论文中添加GitHub链接
- 在ResearchGate、Google Scholar个人主页添加链接
- 在相关学术会议/期刊投稿时提及

**社交媒体：**
- Twitter/X: 发布项目介绍
- LinkedIn: 分享项目成果
- 知乎/CSDN: 撰写技术博客

**开发者社区：**
- Reddit (r/MachineLearning, r/computervision)
- Papers with Code: 提交论文和代码链接
- Awesome Lists: 提交PR到相关awesome列表

---

## 📝 常用Git命令速查

### 日常更新

```bash
# 查看状态
git status

# 添加文件
git add .
git add 文件名

# 提交
git commit -m "更新说明"

# 推送
git push

# 拉取最新代码
git pull
```

### 分支管理

```bash
# 创建新分支
git branch 分支名
git checkout -b 分支名

# 切换分支
git checkout 分支名

# 合并分支
git merge 分支名

# 删除分支
git branch -d 分支名
```

### 查看历史

```bash
# 查看提交历史
git log
git log --oneline
git log --graph

# 查看文件修改
git diff
```

---

## ⚠️ 注意事项

### 1. 不要上传的内容

- ❌ 大型数据集文件（>100MB）
- ❌ 模型权重文件（>100MB）
- ❌ 个人敏感信息
- ❌ API密钥和密码
- ❌ 临时文件和缓存

### 2. 使用.gitignore

已经创建的`.gitignore`文件会自动忽略：
- Python缓存文件
- 模型权重
- 数据集
- 日志文件
- IDE配置文件

### 3. 大文件处理

如果需要上传大文件（如模型权重）：

**方法一：使用Git LFS**
```bash
# 安装Git LFS
git lfs install

# 追踪大文件
git lfs track "*.pt"
git lfs track "*.pth"

# 提交
git add .gitattributes
git add weights/*.pt
git commit -m "Add model weights"
git push
```

**方法二：使用云盘**
- 上传到百度网盘/Google Drive
- 在README中提供下载链接

### 4. 保持更新

定期更新项目：
```bash
# 每次有改进时
git add .
git commit -m "描述更新内容"
git push
```

---

## 🆘 常见问题

### Q1: 推送时提示权限错误

**解决方案：**
使用Personal Access Token代替密码

### Q2: 文件太大无法推送

**解决方案：**
1. 使用Git LFS
2. 或将大文件上传到云盘，提供下载链接

### Q3: 如何删除已提交的敏感文件

**解决方案：**
```bash
# 从Git历史中删除文件
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch 文件路径" \
  --prune-empty --tag-name-filter cat -- --all

# 强制推送
git push origin --force --all
```

### Q4: 如何更新README

```bash
# 编辑README.md文件
# 然后提交
git add README.md
git commit -m "Update README"
git push
```

---

## 📧 需要帮助？

如果遇到问题：
1. 查看Git官方文档：https://git-scm.com/doc
2. 查看GitHub帮助：https://docs.github.com
3. 搜索Stack Overflow
4. 在项目Issues中提问

---

## ✅ 检查清单

上传前确认：

- [ ] 所有代码文件已添加
- [ ] README.md内容完整
- [ ] LICENSE文件存在
- [ ] .gitignore配置正确
- [ ] 没有敏感信息
- [ ] 大文件已处理
- [ ] 文档链接有效
- [ ] 联系方式已更新

上传后确认：

- [ ] GitHub页面显示正常
- [ ] README渲染正确
- [ ] 文件结构清晰
- [ ] 链接可以访问
- [ ] Topics标签已添加
- [ ] Release已发布

---

**祝您开源顺利！🎉**

如有任何问题，欢迎随时询问！