import pathlib
import torch
import time
import random
import torch.nn as nn
import pandas as pd
import numpy as np
from tabulate import tabulate
from tqdm import tqdm, trange
import torch.nn.functional as F
from transformers import AdamW
from rich.progress import track
from sklearn.model_selection import KFold
from util.util_metric import caculate_metric
import gc
from model import LMFFT
from data import DataIterator
import json
import pickle

rewrite_print = print

log_file = None


def get_bert_optimizer(args, model):
    optimizer = AdamW(model.parameters(), eps=1e-8, lr=args.learning_rate, weight_decay=1e-5)
    return optimizer


class FocalLoss(nn.Module):
    def __init__(self, gama=1.5, alpha=0.9375, weight=None, reduction="mean") -> None:
        super().__init__()
        self.loss_fcn = nn.CrossEntropyLoss(weight=weight, reduction=reduction, ignore_index=-1)
        self.gama = gama
        self.alpha = alpha

    def forward(self, pre, target):
        logp = self.loss_fcn(pre, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gama * self.alpha * logp
        return loss.sum()


class FGM():
    '''
    Example
    # 初始化
    fgm = FGM(model, epsilon=1, emb_name='word_embeddings.')
    for batch_input, batch_label in data:
        # 正常训练
        loss = model(batch_input, batch_label)
        loss.backward() # 反向传播，得到正常的grad
        # 对抗训练
        fgm.attack() # 在embedding上添加对抗扰动
        loss_adv = model(batch_input, batch_label)
        loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        fgm.restore() # 恢复embedding参数
        # 梯度下降，更新参数
        optimizer.step()
        model.zero_grad()
    '''
    def __init__(self, model, emb_name,epsilon=1.0):
        # emb_name这个参数要换成你模型中embedding的参数名
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm!=0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class FGSM():
    def __init__(self, model, emb_name='embedding', epsilon=1):
        self.model = model
        self.eps = epsilon
        self.emb_name = emb_name
        self.backup = {}

    def attack(self):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                r_at = self.eps * param.grad.sign()
                param.data.add_(r_at)

    def restore(self):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, para in self.model.named_parameters():
            if para.requires_grad and self.emb_name in name:
                assert name in self.backup
                para.data = self.backup[name]
        self.backup = {}

def get_loss(logits, label, criterion):
    b = 0.06
    loss = criterion(logits, label)
    loss = loss.float()
    loss = (loss - b).abs() + b
    return loss

def train(args, raw_datasets):
    global log_file
    log_file = pathlib.Path(args.log_file)
    print(f'time: { time.strftime("%Y - %m - %d %H : %M : %S") }')

    # 加载 train - test 数据集
    trn_input_ids = raw_datasets["train_input_ids"]
    trn_labels = raw_datasets["train_labels"]
    train_set = DataIterator(args, trn_input_ids, trn_labels)
    train_set.shuffle()
    print(f"train file size ： {train_set.sample_num}")
    
    tst_input_ids = raw_datasets["test_input_ids"]
    tst_labels = raw_datasets["test_labels"]
    test_set = DataIterator(args, tst_input_ids, tst_labels)
    
    best_mcc = 0
    best_metric = []

    model = LMFFT(args).to(args.device)
    model.model.embeddings.position_embeddings = None

    print(model, vis=False)
    print(args, vis=False)
    optimizer = get_bert_optimizer(args, model)
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1, args.weight])).to(args.device)
    
    train_start = time.time()
    gc.collect()
    if args.fgm:
        fgm = FGM(model=model, emb_name="word_embeddings", epsilon=1)
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        start_time = time.time()
        
        print(f"Training {epoch} epoch:")
        for i in track(range(train_set.sample_num)):
            input_ids, target = train_set.get_index(i)
            output_prob1 = model.forward(input_ids)
            
            target = target.reshape([-1]).long()

            # loss
            loss_ce = get_loss(output_prob1, target, criterion)
            loss_ce.backward()
            # FGM
            if args.fgm:
                fgm.attack()
                output_prob2 = model.forward(input_ids)
                loss_adv = get_loss(output_prob2, target, criterion)
                loss_adv.backward()
                fgm.restore()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss_ce.item()
             
        end_time = time.time()
        print("Epoch %d, Loss: %f, The time used is: %d (s)" % (epoch, epoch_loss, (end_time - start_time)))
        ACC, Precision, Sensitivity, Specificity, F1, AUC, MCC, tp, fp, tn, fn = eval(args, model, test_set)

        if MCC > best_mcc:
            best_mcc = MCC
            best_metric = [ACC, Precision, Sensitivity, Specificity, F1, AUC, MCC, tp, fp, tn, fn]
            if args.save_model_path is not None:
                torch.save(model.state_dict(), f"./pkls/{args.save_model_path}.pkl")

    print("Best indicator:")
    header = ["ACC", "Precision", "Sensitivity", "Specificity", "F1-score", "AUC", "MCC", "tp", "fp", "tn", "fn"]
    rows = [
        (best_metric[0], best_metric[1], best_metric[2], best_metric[3], best_metric[4], best_metric[5], best_metric[6],
         best_metric[7], best_metric[8], best_metric[9], best_metric[10]),
    ]
    print(tabulate(rows, headers=header))
    print(args, vis=False, file=True)
    print(tabulate(rows, headers=header), vis=False, file=True)
    print(f"The training takes {(time.time() - train_start) // 60} minutes\n\n")

def train_kfold(args, raw_datasets):
    global log_file
    
    log_file = pathlib.Path(args.log_file)
    print(f'time: { time.strftime("%Y - %m - %d %H : %M : %S") }')

    kf = KFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1, args.weight])).to(args.device)

    print(args, vis=False)
    for fold, (train_index, val_index) in enumerate(kf.split(raw_datasets['train_input_ids'])):
        print("\n========== Fold " + str(fold + 1) + " ==========")

        trn_input_ids = [raw_datasets["train_input_ids"][_] for _ in train_index]
        trn_labels = [raw_datasets["train_labels"][_] for _ in train_index]
        train_set = DataIterator(args, trn_input_ids, trn_labels)

        valid_input_ids = [raw_datasets["train_input_ids"][_] for _ in val_index]
        valid_labels = [raw_datasets["train_labels"][_] for _ in val_index]
        valid_set = DataIterator(args, valid_input_ids, valid_labels)

        best_mcc = 0
        best_metric = []

        model = LMFFT(args).to(args.device)

        model.model.embeddings.position_embeddings = None

        train_start = time.time()
        optimizer = get_bert_optimizer(args, model)
        
        gc.collect()
        
        if args.fgm:
            fgm = FGM(model=model, emb_name="word_embeddings", epsilon=1.0)
        for epoch in track(range(1, args.epochs + 1)):
            model.train()
            epoch_loss = 0

            print(f"Training {epoch} epoch:")
            for i in range(train_set.sample_num):
                input_ids, target = train_set.get_index(i)
                output_prob = model.forward(input_ids)
                target = target.reshape([-1]).long()
                # loss
                loss = get_loss(output_prob, target, criterion)
                # upgrade
                loss.backward()
                # 对抗网络
                if args.fgm:
                    fgm.attack()
                    output_prob = model.forward(input_ids)
                    loss_adv = get_loss(output_prob, target, criterion)
                    loss_adv.backward()
                    fgm.restore()
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += loss.item()
            ACC, Precision, Sensitivity, Specificity, F1, AUC, MCC, tp, fp, tn, fn = eval(args, model, valid_set)

            if MCC > best_mcc:
                best_mcc = MCC
                best_metric = [ACC, Precision, Sensitivity, Specificity, F1, AUC, MCC, tp, fp, tn, fn]
                if args.save_model_path is not None:
                    torch.save(model.state_dict(), f"./pkls/hyper_experiment/fold_{args.save_model_path}_{fold}.pkl")

        print("test:")
        header = ["ACC", "Precision", "Sensitivity", "Specificity", "F1-score", "AUC", "MCC", "tp", "fp", "tn", "fn"]
        rows = [
            (best_metric[0], best_metric[1], best_metric[2], best_metric[3], best_metric[4], best_metric[5], best_metric[6],
            best_metric[7], best_metric[8], best_metric[9], best_metric[10]),
        ]
        print(tabulate(rows, headers=header))
        print(tabulate(rows, headers=header), vis=False, file=True)
        print(f"The training takes {(time.time() - train_start) // 60} minutes\n\n")

    label_pred = []
    pred_prob = []
    targets = []
    for fold, (train_index, val_index) in enumerate(kf.split(raw_datasets['train_input_ids'])):
        print("\n========== Fold " + str(fold + 1) + " ==========")

        valid_input_ids = [raw_datasets["train_input_ids"][_] for _ in val_index]
        valid_labels = [raw_datasets["train_labels"][_] for _ in val_index]
        valid_set = DataIterator(args, valid_input_ids, valid_labels)

        model = LMFFT(args).to(args.device)
        model.load_state_dict()       # 加载训练好的模型pkl文件

        model.model.embeddings.position_embeddings = None
        
        for i in range(valid_set.sample_num):
            input_ids, target = valid_set.get_index(i)
            out_pred = model.forward(input_ids)

            if args.logit == 2:
                out_pred = F.softmax(out_pred, dim=-1)
                out_pred = out_pred.reshape([-1, 2])
                label_p = torch.argmax(out_pred, dim=-1).squeeze(0).cpu().detach().numpy()
                out_pred = out_pred[:, 1].cpu().detach().numpy()
            else:
                out_pred = out_pred.reshape([-1, 1])
                label_p = out_pred.sigmoid() > 0.5

            target = target.reshape([-1]).long().cpu().detach().numpy()

            label_pred.extend(label_p)
            pred_prob.extend(out_pred)
            targets.extend(target)

            input_ids, target = None, None

    metric, roc_data, prc_data = caculate_metric(label_pred, targets, pred_prob)
    ACC, Precision, Sensitivity, Specificity, F1, AUC, MCC, tp, fp, tn, fn = metric

    print("test:")
    header = ["ACC", "Precision", "Sensitivity", "Specificity", "F1-score", "AUC", "MCC", "tp", "fp", "tn", "fn"]
    rows = [
        (best_metric[0], best_metric[1], best_metric[2], best_metric[3], best_metric[4], best_metric[5], best_metric[6],
        best_metric[7], best_metric[8], best_metric[9], best_metric[10]),
    ]
    print(tabulate(rows, headers=header))
    print(tabulate(rows, headers=header), vis=False, file=True)
    print(f"The training takes {(time.time() - train_start) // 60} minutes\n\n")


def eval(args, model, test_set):
    model.eval()

    label_pred = []
    pred_prob = []
    targets = []
    for i in range(test_set.sample_num):
        input_ids, target = test_set.get_index(i)
        out_pred = model.forward(input_ids)

        if args.logit == 2:
            out_pred = F.softmax(out_pred, dim=-1)
            out_pred = out_pred.reshape([-1, 2])
            label_p = torch.argmax(out_pred, dim=-1).squeeze(0).cpu().detach().numpy()
            out_pred = out_pred[:, 1].cpu().detach().numpy()
        else:
            out_pred = out_pred.reshape([-1, 1])
            label_p = out_pred.sigmoid() > 0.5

        target = target.reshape([-1]).long().cpu().detach().numpy()

        label_pred.extend(label_p)
        pred_prob.extend(out_pred)
        targets.extend(target)

        input_ids, target = None, None

    metric, roc_data, prc_data = caculate_metric(label_pred, targets, pred_prob)
    ACC, Precision, Sensitivity, Specificity, F1, AUC, MCC, tp, fp, tn, fn = metric

    header = ["ACC", "Precision", "Sensitivity", "Specificity", "F1-score", "AUC", "MCC", "tp", "fp", "tn", "fn"]
    rows = [
        (ACC, Precision, Sensitivity, Specificity, F1, AUC, MCC, tp, fp, tn, fn),
    ]
    print(tabulate(rows, headers=header))
    print("\n")

    return ACC, Precision, Sensitivity, Specificity, F1, AUC, MCC, tp, fp, tn, fn


def print(content, vis=True, file=False):
    global log_file

    if file == False:
        file = log_file
    else:
        file = pathlib.Path(f"./logs/best_Metric.txt")

    if vis:
        rewrite_print(content) # 打印到控制台

    with open(file, "a", encoding='utf-8') as f:
        rewrite_print(content, file=f)