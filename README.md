# LMFFT

## Introduction
This repository contains the code and data for the paper titled "Language model encoded multi-scale feature fusion and transformation for predicting protein-peptide binding sites".  The paper introduces a novel sequence-based end-to-end PPBS predictor using deep learning, named Language model encoded Multi-scale Feature Fusion and Transformation (LMFFT). The proposed model starts with a single protein language model for comprehensive multi-scale feature extraction, including residue, dipeptide, and fragment-level representations, which are implemented by the dipeptide embedding-based fragment fusion and further enhanced through the dipeptide contextual encoding. Moreover, multi-scale convolutional neural networks are applied to transform multi-scale features by capturing intricate interactions between local and global information. Our LMFFT achieves state-of-the-art performance across three benchmark datasets, outperforming existing sequence-based methods and demonstrating competitive advantages over certain structure-based baselines. This work provides a cost-effective and efficient solution for PPBS prediction, advancing revealing the sequence-function relationship of proteins.

## Prepare
Since the maximum length that esm2_t33_650M_UR50D-finetuned-secondary structure can handle is 1024, and there are some protein sequences in the dataset with lengths far exceeding 1024; To solve this problem, the way we refer to the https://kexue.fm/archives/7947, to deal with location coding.Reference position /Users/liang/miniconda3/envs/pytorch/lib/python3.11/site-packages/transformers/models/esm.

## 📁 Project 📁
```markdown
data.py              # data processing 
model.py             # model
run.py               # Main running script
train_eval.py        # train and eaval
data/                # dataset
    Dataset1_test.tsv
    Dataset1_train.tsv
    Dataset2_test.tsv
    Dataset2_train.tsv
    Dataset3_test.tsv
    Dataset3_train.tsv
pkls/                # Model weight file
util/                # tools
    util_metric.py
```

## ⚙️ Setup  ⚙️
To run the code in this repository, you'll need the following dependencies:
- python 3.10.14
- torch 2.2.2
- transformers 4.39.3


## 🤖 Download  🤖
Before executing the code, you need to download the pre-trained model [esm2_t33_650M_UR50D-finetuned-secondary-structure](https://huggingface.co/gaodrew/esm2_t33_650M_UR50D-finetuned-secondary-structure) and place it in the ./pretrained_model directory.

## ⚡️ Running the Code  ⚡️
- Model Training:
```python
python run.py --do_train --dataset <dataset_num: 1, 2, 3>
```

- Model Testing:
Download the [model](https://drive.google.com/drive/folders/1Vrf7G1rzmW5sezYpwwTAHnXbLHT2R5P3?usp=drive_link) and place it in the `./pkls` directory.
```python
python run.py --eval --dataset <dataset_num: 1, 2, 3>
```




