# Knowcl GitHub 项目库维护：新手日常可照抄完整版

> 适合你现在的情况：**第一次上传已经做过了**，以后主要是日常维护项目库。
>
> 你是做大模型训练的，所以重点是：**代码和文档进 GitHub，数据、模型权重、输出结果不要进 GitHub。**

---

# 0. 你只需要记住这一句话

```text
先进入代码目录 → git status → git diff → git add 指定文件 → git diff --staged → git commit → git push
```

每天维护项目库，基本就是这套流程。

---

# 1. 永远先进入这个目录

每次操作 Git 前，先执行：

```powershell
cd "G:\Knowcl\888-代码"
```

然后确认位置：

```powershell
git status
```

你应该只在这里维护 Git 仓库：

```text
G:\Knowcl\888-代码
```

不要在这里操作 Git：

```text
G:\Knowcl
```

因为 `G:\Knowcl` 里面可能有：

```text
777-模型权重
999-输出成果文件
真实数据
训练输出
临时文件
```

这些都不应该上传到 GitHub。

---

# 2. 每天开始工作前怎么做

每天开始继续项目时，先执行：

```powershell
cd "G:\Knowcl\888-代码"

git status

git pull
```

意思很简单：

```text
cd          进入代码目录
git status  看当前有没有乱的改动
git pull    从 GitHub 拉取最新版本
```

如果 `git pull` 报错，先不要乱试，把报错内容复制出来再处理。

---

# 3. 每次改完后怎么提交

这是最常用的日常模板。

你改完文件后，按顺序执行：

```powershell
git status

git diff

git add 文件名1 文件名2

git diff --staged

git commit -m "提交说明"

git push
```

例如你只改了两个文档：

```powershell
git status

git diff

git add docs/progress.md docs/task_plan.md

git diff --staged

git commit -m "文档：更新今日进度"

git push
```

这就是你最常用的一套。

---

# 4. 每个命令是什么意思

## 4.1 `git status`

查看当前项目状态。

```powershell
git status
```

它会告诉你：

```text
哪些文件改了
哪些文件是新建的
哪些文件已经准备提交
当前在哪个分支
```

你迷路的时候，先运行这个。

---

## 4.2 `git diff`

查看你具体改了什么。

```powershell
git diff
```

提交前最好看一眼，避免把乱改的内容提交上去。

---

## 4.3 `git add 文件名`

选择这次要提交的文件。

```powershell
git add docs/progress.md
```

多个文件可以这样：

```powershell
git add docs/progress.md scripts/train.py configs/e1.yaml
```

新手尽量不要直接用：

```powershell
git add .
```

因为它可能把你不想提交的文件也加进去。

---

## 4.4 `git diff --staged`

查看已经 add、即将提交的内容。

```powershell
git diff --staged
```

每次 commit 前都看一下。

重点确认没有这些东西：

```text
.env
*.csv
*.xlsx
*.pt
*.pth
*.ckpt
*.safetensors
outputs/
runs/
logs/
777-模型权重/
999-输出成果文件/
```

---

## 4.5 `git commit -m "说明"`

正式保存一次版本记录。

```powershell
git commit -m "文档：更新今日进度"
```

commit 可以理解为：

```text
给当前这次改动拍一张快照，并写一句说明。
```

---

## 4.6 `git push`

把本地提交上传到 GitHub。

```powershell
git push
```

建议：

```text
每次 commit 后就 push
每天结束前一定 push
换电脑前一定 push
```

---

# 5. 你这个大模型训练项目，哪些能提交，哪些不能提交

## 5.1 可以提交的内容

这些通常可以提交：

```text
README.md
.gitignore
.env.example
requirements.txt
configs/
docs/
scripts/
src/
notebooks/  但必须清理输出
```

简单理解：

```text
代码可以提交
说明文档可以提交
配置模板可以提交
实验记录可以提交
小型 notebook 可以提交，但要清理输出
```

---

## 5.2 绝对不要提交的内容

这些不要提交：

```text
.env
真实数据
训练数据
验证数据
测试数据
模型权重
checkpoint
训练输出
日志
中间结果
大表格
大图片
```

常见危险文件：

```text
*.csv
*.xlsx
*.jsonl
*.parquet
*.pkl
*.npy
*.npz
*.pt
*.pth
*.ckpt
*.bin
*.safetensors
*.onnx
```

常见危险目录：

```text
data/
datasets/
raw_data/
processed_data/
outputs/
runs/
logs/
wandb/
mlruns/
checkpoints/
weights/
777-模型权重/
999-输出成果文件/
```

简单判断：

```text
别人 clone 你的仓库后，应该知道怎么运行项目。
但不应该拿到你的真实数据、模型权重、密钥、训练输出。
```

---

# 6. 最常用的 8 个提交场景

## 场景 1：只更新进度文档

你改了：

```text
docs/progress.md
docs/task_plan.md
```

命令：

```powershell
git status

git diff

git add docs/progress.md docs/task_plan.md

git diff --staged

git commit -m "文档：更新今日进度"

git push
```

---

## 场景 2：更新实验记录

你改了：

```text
docs/experiment_log.md
docs/findings.md
```

命令：

```powershell
git status

git diff

git add docs/experiment_log.md docs/findings.md

git diff --staged

git commit -m "实验：记录第1版实验结果"

git push
```

---

## 场景 3：改了训练脚本

你改了：

```text
scripts/train.py
src/trainer.py
```

命令：

```powershell
git status

git diff

git add scripts/train.py src/trainer.py

git diff --staged

git commit -m "训练：更新训练流程"

git push
```

---

## 场景 4：改了数据处理脚本

你改了：

```text
scripts/process_data.py
src/dataset.py
```

命令：

```powershell
git status

git diff

git add scripts/process_data.py src/dataset.py

git diff --staged

git commit -m "数据：更新数据处理脚本"

git push
```

---

## 场景 5：改了模型结构

你改了：

```text
src/model.py
configs/model.yaml
```

命令：

```powershell
git status

git diff

git add src/model.py configs/model.yaml

git diff --staged

git commit -m "模型：更新模型结构"

git push
```

---

## 场景 6：修了一个 bug

你改了：

```text
src/load_label.py
```

命令：

```powershell
git status

git diff

git add src/load_label.py

git diff --staged

git commit -m "修复：解决标签读取问题"

git push
```

---

## 场景 7：新增一个脚本

你新增了：

```text
scripts/check_data.py
```

命令：

```powershell
git status

git diff

git add scripts/check_data.py

git diff --staged

git commit -m "新增：添加数据检查脚本"

git push
```

---

## 场景 8：更新配置文件

你改了：

```text
configs/e1.yaml
configs/train.yaml
```

命令：

```powershell
git status

git diff

git add configs/e1.yaml configs/train.yaml

git diff --staged

git commit -m "配置：更新第1版实验配置"

git push
```

---

# 7. Commit Message：普通人好懂版

你不用一开始就学复杂英文规范。

你可以直接用中文，格式简单一点：

```text
类别：这次做了什么
```

例如：

```text
文档：更新今日进度
实验：记录第1版baseline结果
训练：更新训练流程
数据：新增数据检查脚本
模型：调整模型结构
配置：更新第2版实验参数
修复：解决标签读取报错
清理：删除无用临时代码
```

---

# 8. 推荐你使用的中文类别

## 8.1 日常最够用的类别

```text
文档
实验
训练
数据
模型
配置
修复
新增
清理
重构
```

含义：

| 类别 | 什么时候用 |
|---|---|
| 文档 | 改 README、进度、计划、实验记录 |
| 实验 | 记录实验结果、实验版本、实验结论 |
| 训练 | 改训练脚本、训练流程 |
| 数据 | 改数据处理、数据检查、数据加载 |
| 模型 | 改模型结构、模型模块 |
| 配置 | 改 yaml、json、参数配置 |
| 修复 | 修 bug、修报错 |
| 新增 | 新加脚本、新加功能 |
| 清理 | 删除垃圾代码、整理文件 |
| 重构 | 调整代码结构，但功能大体不变 |

---

## 8.2 最推荐的 commit message 模板

### 文档类

```powershell
git commit -m "文档：更新今日进度"
```

```powershell
git commit -m "文档：更新任务计划"
```

```powershell
git commit -m "文档：补充实验结论"
```

---

### 实验类

```powershell
git commit -m "实验：新增第1版baseline配置"
```

```powershell
git commit -m "实验：记录第1版baseline结果"
```

```powershell
git commit -m "实验：更新第2版对比结果"
```

```powershell
git commit -m "实验：整理第3版失败原因"
```

---

### 训练类

```powershell
git commit -m "训练：更新训练入口脚本"
```

```powershell
git commit -m "训练：添加断点续训逻辑"
```

```powershell
git commit -m "训练：调整日志保存方式"
```

---

### 数据类

```powershell
git commit -m "数据：新增数据检查脚本"
```

```powershell
git commit -m "数据：更新数据加载逻辑"
```

```powershell
git commit -m "数据：修正标签读取流程"
```

---

### 模型类

```powershell
git commit -m "模型：新增baseline模型"
```

```powershell
git commit -m "模型：调整特征融合模块"
```

```powershell
git commit -m "模型：更新知识图谱编码模块"
```

---

### 配置类

```powershell
git commit -m "配置：新增第1版训练参数"
```

```powershell
git commit -m "配置：调整学习率和batch size"
```

```powershell
git commit -m "配置：更新实验输出路径模板"
```

---

### 修复类

```powershell
git commit -m "修复：解决GBK编码读取报错"
```

```powershell
git commit -m "修复：解决模型加载路径错误"
```

```powershell
git commit -m "修复：解决训练日志为空的问题"
```

---

### 清理类

```powershell
git commit -m "清理：删除无用调试代码"
```

```powershell
git commit -m "清理：整理实验文档结构"
```

```powershell
git commit -m "清理：移除临时打印信息"
```

---

# 9. 多个版本更新怎么写

你可能会有很多实验版本，例如：

```text
第1版 baseline
第2版 调参
第3版 加知识图谱
第4版 改损失函数
第5版 消融实验
```

推荐在 commit message 里写清楚版本：

```powershell
git commit -m "实验：新增第1版baseline配置"
```

```powershell
git commit -m "实验：记录第1版baseline结果"
```

```powershell
git commit -m "实验：更新第2版调参结果"
```

```powershell
git commit -m "模型：新增第3版知识图谱模块"
```

```powershell
git commit -m "实验：记录第4版损失函数对比"
```

```powershell
git commit -m "实验：整理第5版消融实验结果"
```

---

# 10. 更清楚的版本命名方式

如果你想更规范一点，可以使用：

```text
e1-v01
e1-v02
e1-v03
e2-v01
e2-v02
```

含义：

```text
e1     第1组实验
v01    第1版
e1-v02 第1组实验第2版
```

commit message 可以这样：

```powershell
git commit -m "实验：e1-v01 新增baseline配置"
```

```powershell
git commit -m "实验：e1-v01 记录baseline结果"
```

```powershell
git commit -m "实验：e1-v02 调整学习率和batch size"
```

```powershell
git commit -m "实验：e2-v01 新增知识图谱模型配置"
```

```powershell
git commit -m "实验：e2-v02 记录知识图谱模型结果"
```

这种方式很适合大模型训练项目。

---

# 11. 推荐你的实验版本写法

你可以这样定：

```text
e1-baseline        第1组：baseline实验
e2-tune            第2组：调参实验
e3-kg              第3组：知识图谱相关实验
e4-loss            第4组：损失函数实验
e5-ablation        第5组：消融实验
e6-final           第6组：最终模型整理
```

然后每一组下面再分版本：

```text
e1-v01
e1-v02
e1-v03

e2-v01
e2-v02

e3-v01
e3-v02
```

commit message 示例：

```powershell
git commit -m "实验：e1-v01 新增baseline配置"
```

```powershell
git commit -m "实验：e1-v02 调整训练参数"
```

```powershell
git commit -m "实验：e3-v01 新增知识图谱输入"
```

```powershell
git commit -m "实验：e5-v01 记录消融实验结果"
```

---

# 12. 大实验建议开分支

如果只是改文档、小脚本，可以直接在 `main` 提交。

如果是下面这些情况，建议开实验分支：

```text
要大改训练代码
要改模型结构
要试新的 loss
要试新的数据处理流程
不确定能不能跑通
可能会把 main 搞乱
```

原则：

```text
小改动直接 main
大实验开分支
```

---

# 13. 创建实验分支

例如你要做第 1 组 baseline 实验：

```powershell
git checkout main

git pull

git checkout -b exp/e1-baseline
```

意思是：

```text
先回到 main
拉取 GitHub 最新版本
创建一个叫 exp/e1-baseline 的实验分支
```

---

# 14. 在实验分支上提交

例如你改了：

```text
configs/e1.yaml
scripts/train_baseline.py
docs/experiment_log.md
```

命令：

```powershell
git status

git diff

git add configs/e1.yaml scripts/train_baseline.py docs/experiment_log.md

git diff --staged

git commit -m "实验：e1-v01 新增baseline配置"

git push -u origin exp/e1-baseline
```

第一次推送新分支时，用：

```powershell
git push -u origin exp/e1-baseline
```

以后在这个分支继续推送，直接：

```powershell
git push
```

---

# 15. 实验成功后合并回 main

如果实验分支已经验证成功，想合并回 `main`：

```powershell
git checkout main

git pull

git merge --squash exp/e1-baseline

git status

git diff --staged

git commit -m "实验：合并第1组baseline实验"

git push
```

为什么用 `--squash`？

因为实验分支上可能有很多零碎提交：

```text
试一下
修一下
改参数
再跑一次
修 bug
```

这些全部放进 main 会很乱。

`git merge --squash` 可以把它们压成一次干净提交。

---

# 16. 实验失败后怎么处理

如果实验失败，不想保留这个分支：

```powershell
git checkout main

git branch -D exp/e1-baseline
```

如果这个分支已经推送到 GitHub，还要删除远端分支：

```powershell
git push origin --delete exp/e1-baseline
```

注意：

```text
删除前确认这个实验真的不需要了。
```

---

# 17. 什么时候用 stash

`stash` 是临时保存改动。

适合这种情况：

```text
我代码改到一半，还不想 commit
但我现在要切分支
或者我要先 git pull
```

命令：

```powershell
git status

git stash -u
```

恢复：

```powershell
git stash pop
```

新手记住：

```text
临时保存用 stash
长期保存用 commit
大实验用分支
```

不要把 stash 当长期备份。

---

# 18. 不小心 add 错文件怎么办

比如你执行了：

```powershell
git add .
```

结果把数据文件也加进去了。

先不要 commit。

取消暂存：

```powershell
git restore --staged 文件名
```

例如：

```powershell
git restore --staged data/raw.csv
```

如果整个目录都加错了：

```powershell
git restore --staged data/
```

然后再检查：

```powershell
git status
```

---

# 19. 不小心 commit 了不该提交的文件，但还没 push

比如你不小心 commit 了：

```text
.env
```

但还没有 push。

先从 Git 里移除，但保留本地文件：

```powershell
git rm --cached .env
```

确认 `.gitignore` 里有：

```gitignore
.env
```

然后修改上一次提交：

```powershell
git commit --amend -m "清理：移除误提交的环境配置"
```

再检查：

```powershell
git status
```

---

# 20. 如果已经 push 了敏感信息

比如已经把下面内容 push 到 GitHub：

```text
.env
API key
token
密码
真实数据
```

不要只是在下一次 commit 删除。

因为旧历史里仍然有。

你应该：

```text
1. 立刻作废泄露的 key、token、密码
2. 从仓库里删除敏感文件
3. 清理 Git 历史
4. 强制推送
```

新手阶段最重要的是：

```text
一开始就不要把 .env、数据、权重、密钥提交进去。
```

---

# 21. `.gitignore` 写了但 Git 还在跟踪怎么办

原因：

```text
.gitignore 只对还没被 Git 跟踪的文件有效。
如果文件以前已经提交过，Git 会继续跟踪它。
```

例如 `.env` 已经被 Git 跟踪了：

```powershell
git rm --cached .env

git commit -m "清理：停止跟踪环境配置文件"

git push
```

如果是 data 目录：

```powershell
git rm -r --cached data/

git commit -m "清理：停止跟踪数据目录"

git push
```

---

# 22. notebook 怎么提交

notebook 可以提交，但一定要注意：

```text
不要带大输出
不要带真实路径
不要带 token
不要带数据内容
不要带模型输出大图
```

提交前在 Jupyter 里清理输出：

```text
Kernel → Restart Kernel and Clear Outputs
```

然后：

```powershell
git status

git diff

git add notebooks/analysis.ipynb docs/experiment_log.md

git diff --staged

git commit -m "文档：更新清理后的分析notebook"

git push
```

---

# 23. 推荐的 `.gitignore` 核心内容

你的项目里 `.gitignore` 至少应该包含这些：

```gitignore
# Python
__pycache__/
*.pyc
.pytest_cache/

# virtual env
.venv/
venv/
env/

# secrets
.env
.env.*
!.env.example
*.key
*.pem
*.token

# data
data/
datasets/
raw_data/
processed_data/
*.csv
*.xlsx
*.xls
*.jsonl
*.parquet
*.pkl
*.pickle
*.npy
*.npz

# model weights
*.pt
*.pth
*.ckpt
*.bin
*.safetensors
*.onnx
checkpoints/
weights/
models/

# outputs and logs
outputs/
runs/
logs/
wandb/
mlruns/
*.log

# notebooks cache
.ipynb_checkpoints/

# local Knowcl folders if copied by mistake
777-模型权重/
999-输出成果文件/
```

---

# 24. 每天结束前检查清单

结束工作前，照着看：

```text
1. 我是否在 G:\Knowcl\888-代码？
2. 我是否运行了 git status？
3. 我是否运行了 git diff？
4. 我是否只 add 了本次需要提交的文件？
5. 我是否运行了 git diff --staged？
6. 暂存内容里有没有数据、权重、输出、.env？
7. commit message 是否能让一个月后的自己看懂？
8. 我是否 git push 了？
```

---

# 25. 最推荐你的日常维护模板

你以后最常用的就是这个：

```powershell
cd "G:\Knowcl\888-代码"

git status

git diff

# 把下面的文件名换成你这次真正改的文件
git add docs/progress.md docs/task_plan.md

git diff --staged

git commit -m "文档：更新今日进度"

git push
```

---

# 26. 大模型训练项目最常用模板

## 26.1 只更新实验记录

```powershell
git status

git diff

git add docs/experiment_log.md docs/findings.md

git diff --staged

git commit -m "实验：记录第1版实验结果"

git push
```

---

## 26.2 新增训练配置

```powershell
git status

git diff

git add configs/e1.yaml docs/experiment_log.md

git diff --staged

git commit -m "配置：新增第1版训练参数"

git push
```

---

## 26.3 更新训练代码

```powershell
git status

git diff

git add scripts/train.py src/trainer.py

git diff --staged

git commit -m "训练：更新训练流程"

git push
```

---

## 26.4 更新模型代码

```powershell
git status

git diff

git add src/model.py configs/model.yaml

git diff --staged

git commit -m "模型：更新模型结构"

git push
```

---

## 26.5 更新数据处理代码

```powershell
git status

git diff

git add src/dataset.py scripts/process_data.py

git diff --staged

git commit -m "数据：更新数据处理流程"

git push
```

---

# 27. 什么时候不要 commit

下面情况先不要 commit：

```text
代码还完全不能跑
你不确定里面有没有数据或权重
你不确定有没有 .env
你只是临时试错
你还没看 git diff --staged
```

这种情况可以先：

```powershell
git status
```

如果只是临时保存：

```powershell
git stash -u
```

如果是大实验：

```powershell
git checkout -b exp/实验名
```

---

# 28. 什么时候必须 push

这些情况一定要 push：

```text
今天工作结束前
换电脑前
一个阶段实验记录完成后
修复了重要 bug 后
合并实验分支后
```

命令：

```powershell
git push
```

---

# 29. 查看历史记录

想看最近提交：

```powershell
git log --oneline
```

想看更清楚的分支图：

```powershell
git log --oneline --decorate --graph --all
```

---

# 30. 给重要版本打标签

如果你完成了一个重要阶段，比如 baseline 跑通了，可以打 tag。

例如：

```powershell
git tag -a v0.1-baseline -m "baseline finished"

git push --tags
```

推荐 tag：

```text
v0.1-baseline
v0.2-data-check
v0.3-kg-model
v0.4-ablation
v1.0-final
```

tag 不用每天打。

只在重要节点打。

---

# 31. 最安全的习惯

你以后就按这个习惯来：

```text
1. 不确定时先 git status
2. add 前先 git diff
3. commit 前先 git diff --staged
4. 不要无脑 git add .
5. 数据、权重、输出、.env 永远不要提交
6. commit message 写成人能看懂的话
7. 大实验开分支
8. 每天结束前 git push
```

---

# 32. 你可以直接复制的最终版流程

## 开始工作

```powershell
cd "G:\Knowcl\888-代码"

git status

git pull
```

## 修改完以后

```powershell
git status

git diff

# 换成你真正要提交的文件
git add docs/progress.md docs/task_plan.md

git diff --staged

git commit -m "文档：更新今日进度"

git push
```

## 跑实验版本

```powershell
git checkout main

git pull

git checkout -b exp/e1-baseline
```

提交实验配置和记录：

```powershell
git status

git diff

git add configs/e1.yaml docs/experiment_log.md

git diff --staged

git commit -m "实验：e1-v01 新增baseline配置"

git push -u origin exp/e1-baseline
```

实验成功后合并：

```powershell
git checkout main

git pull

git merge --squash exp/e1-baseline

git status

git diff --staged

git commit -m "实验：合并第1组baseline实验"

git push
```

---

# 33. 最后只记这 3 件事

## 第一件事

```text
Git 只管 G:\Knowcl\888-代码
```

## 第二件事

```text
代码、文档、配置模板可以提交
数据、权重、输出、.env 不要提交
```

## 第三件事

```text
每次提交就是：status → diff → add 文件 → diff --staged → commit → push
```

只要你长期照这个做，项目库就会干净、稳定、可回滚。

