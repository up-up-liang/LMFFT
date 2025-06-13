# LMFFT

## introduction


## 📁 项目结构 📁

```
.gitignore           # Git忽略文件配置
data.py              # 数据处理相关脚本
model.py             # 模型定义与相关代码
README.md            # 项目说明文档
run.py               # 主运行脚本
train_eval.py        # 训练与评估脚本
data/                # 存放数据集的文件夹
    Dataset1_test.tsv
    Dataset1_train.tsv
    Dataset2_test.tsv
    Dataset2_train.tsv
    Dataset3_test.tsv
    Dataset3_train.tsv
pkls/                # 存放模型权重文件的文件夹
util/                # 工具函数
    util_metric.py
```

## ⚙️ Setup  ⚙️
To run the code in this repository, you'll need the following dependencies:
要在本仓库中运行代码，您需要以下依赖项：
- python 3.10.14
- torch 2.2.2
- transformers 4.39.3


## 🤖 Download  🤖
Before training and testing, you need to download the dataset and place it in the ./data directory.
在训练和测试之前，您需要下载数据集并将其放置在 ./data 目录中。

Before executing the code, you need to download the pre-trained model [esm2_t33_650M_UR50D-finetuned-secondary-structure](https://huggingface.co/gaodrew/esm2_t33_650M_UR50D-finetuned-secondary-structure) , pubmedbert and place them in the ./pretrained_model directory.
在执行代码之前，您需要下载预训练模型[esm2_t33_650M_UR50D-finetuned-secondary-structure](https://huggingface.co/gaodrew/esm2_t33_650M_UR50D-finetuned-secondary-structure)，并将它们放置在 ./pretrained_model 目录中。

## ⚡️ Running the Code  ⚡️
- Model Training:  模型训练：
```
python run.py --do_train --dataset <dataset_num: 1, 2, 3>
```

- Model Testing:  模型测试：

Download the model and place it in the ./pkls directory.
下载模型并将其放置在 ./pkls 目录中。
```
python run.py --eval --dataset <dataset_num: 1, 2, 3>
```




