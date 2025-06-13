# LMFFT

## Introduction
This repository contains the code and data for the paper titled "Language model encoded multi-scale feature fusion and transformation for predicting protein-peptide binding sites".  The paper introduces a novel sequence-based end-to-end PPBS predictor using deep learning, named Language model encoded Multi-scale Feature Fusion and Transformation (LMFFT). The proposed model starts with a single protein language model for comprehensive multi-scale feature extraction, including residue, dipeptide, and fragment-level representations, which are implemented by the dipeptide embedding-based fragment fusion and further enhanced through the dipeptide contextual encoding. Moreover, multi-scale convolutional neural networks are applied to transform multi-scale features by capturing intricate interactions between local and global information. Our LMFFT achieves state-of-the-art performance across three benchmark datasets, outperforming existing sequence-based methods and demonstrating competitive advantages over certain structure-based baselines. This work provides a cost-effective and efficient solution for PPBS prediction, advancing revealing the sequence-function relationship of proteins.

## 📁 Project 📁
```markdown
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
要在本仓库中运行代码，您需要以下依赖项：<br>To run the code in this repository, you'll need the following dependencies:
- python 3.10.14
- torch 2.2.2
- transformers 4.39.3


## 🤖 Download  🤖
在训练和测试之前，您需要下载数据集并将其放置在 ./data 目录中。<br>Before training and testing, you need to download the dataset and place it in the ./data directory.

在执行代码之前，您需要下载预训练模型[esm2_t33_650M_UR50D-finetuned-secondary-structure](https://huggingface.co/gaodrew/esm2_t33_650M_UR50D-finetuned-secondary-structure)，并将它们放置在 ./pretrained_model 目录中。<br>Before executing the code, you need to download the pre-trained model [esm2_t33_650M_UR50D-finetuned-secondary-structure](https://huggingface.co/gaodrew/esm2_t33_650M_UR50D-finetuned-secondary-structure) , pubmedbert and place them in the ./pretrained_model directory.

## ⚡️ Running the Code  ⚡️
- Model Training:
```python
python run.py --do_train --dataset <dataset_num: 1, 2, 3>
```

- Model Testing:
下载[模型](https://drive.google.com/drive/folders/1Vrf7G1rzmW5sezYpwwTAHnXbLHT2R5P3?usp=drive_link)并将其放置在 ./pkls 目录中。<br>Download the [model](https://drive.google.com/drive/folders/1Vrf7G1rzmW5sezYpwwTAHnXbLHT2R5P3?usp=drive_link) and place it in the ./pkls directory.
```python
python run.py --eval --dataset <dataset_num: 1, 2, 3>
```




