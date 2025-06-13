import torch
import time
import torch.nn as nn
from tabulate import tabulate
from tqdm import tqdm, trange
import torch.nn.functional as F
from transformers import AdamW
from rich.progress import track
from util.util_metric import caculate_metric
from model import DL
from data import DataIterator
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, accuracy_score, recall_score, precision_score


def get_bert_optimizer(args, model):
    # Prepare optimizer and schedule (linear warmup and decay)
    # no_decay = ['bias', 'LayerNorm.weight']
    # diff_part = ["bert.embeddings", "bert.encoder"]
    # optimizer_grouped_parameters = [
    #     {
    #         "params": [p for n, p in model.named_parameters() if
    #                 not any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
    #         "weight_decay": 0.0,
    #         "lr": 2e-5
    #     },
    #     {
    #         "params": [p for n, p in model.named_parameters() if
    #                 any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
    #         "weight_decay": 0.0,
    #         "lr": 2e-5
    #     },
    #     {
    #         "params": [p for n, p in model.named_parameters() if
    #                 not any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
    #         "weight_decay": 0.0,
    #         "lr": 1e-3
    #     },
    #     {
    #         "params": [p for n, p in model.named_parameters() if
    #                 any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
    #         "weight_decay": 0.0,
    #         "lr": 1e-3
    #     },
    # ]
    # optimizer = AdamW(optimizer_grouped_parameters, eps=1e-8)
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


# class FocalLoss(nn.Module):
#     def __init__(self, gamma=2, weight=None):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.weight = weight
#
#     def forward(self, inputs, targets):
#         ce_loss = nn.CrossEntropyLoss(weight=self.weight, reduction="mean", ignore_index=-1)(inputs, targets)  # 使用交叉熵损失函数计算基础损失
#         pt = torch.exp(-ce_loss)  # 计算预测的概率
#         focal_loss = (1 - pt) ** self.gamma * ce_loss  # 根据Focal Loss公式计算Focal Loss
#         return focal_loss


def get_loss(logits, label, criterion):
    b = 0.06
    loss = criterion(logits, label)
    loss = loss.float()
    # flooding method
    loss = (loss - b).abs() + b

    # multi-sense loss
    # alpha = -0.1
    # loss_dist = alpha * cal_loss_dist_by_cosine(model)
    # loss += loss_dist
    return loss


def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize
        z = F.normalize(z, dim=1) # l2-normalize
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return 1 - F.cosine_similarity(p, z, dim=-1)
    else:
        raise Exception


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # euclidean_distance: [128]
        # euclidean_distance = F.pairwise_distance(output1, output2)
        cos_distance = D(output1, output2)
        # print("ED",euclidean_distance)
        loss_contrastive = torch.mean((1 - label) * torch.pow(cos_distance, 2) +  # calmp夹断用法
                                      (label) * torch.pow(torch.clamp(self.margin - cos_distance, min=0.0), 3))

        return loss_contrastive


def train(args, raw_datasets):
    trn_input_ids = raw_datasets["train_input_ids"]
    trn_labels = raw_datasets["train_labels"]
    train_set = DataIterator(args, trn_input_ids, trn_labels)

    epochs = 30

    model = DL(args).to("cuda")
    optimizer = get_bert_optimizer(args, model)
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1, 17])).to("cuda")
    contras_criterion = ContrastiveLoss().to("cuda")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        start_time = time.time()

        current_index = 0
        # prob
        prob_batch = []
        target_batch = []
        # represent
        repr_batch1 = []
        repr_batch2 = []
        contras_batch = []
        # for i in track(train_set.sample_num, desc="Training %d epoch" % (epoch,)):
        print(f"Training {epoch} epoch", end=": ")
        for i in track(range(train_set.sample_num)):
            input_ids, target = train_set.get_index(i)
            output_repr = model.forward(input_ids)[0, 1:-1]
            output_prob = model(input_ids)[0, 1:-1]

            output_prob = F.softmax(output_prob, dim=-1)
            target = target.reshape([-1]).long()

            # prob
            prob_batch.append(output_prob)
            target_batch.append(target)
            # represent
            repr_batch1.append(output_repr[:-1])
            repr_batch2.append(output_repr[1:])
            for i in range(target.shape[0]):
                xor_label = (target[i] ^ target[i+1])
                contras_batch.append(xor_label.unsqueeze(0))

            current_index += 1
            if current_index % args.batch_size == 0:
                prob_batch = torch.cat(prob_batch, dim=0)
                target_batch = torch.cat(target_batch, dim=0)

                repr_batch1 = torch.cat(repr_batch1, dim=0)
                repr_batch2 = torch.cat(repr_batch2, dim=0)
                contras_batch = torch.cat(contras_batch, dim=0)

                # loss
                ce_loss = get_loss(prob_batch, target_batch, criterion)
                contras_loss = contras_criterion(repr_batch1, repr_batch2, contras_batch)
                loss = ce_loss + contras_loss

                epoch_loss += loss
                # upgrade
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                prob_batch = []
                target_batch = []

        end_time = time.time()
        print("Epoch %d, Loss: %f, The time used is: %d (s)" % (epoch, epoch_loss, (end_time - start_time)))

        eval(args, model, raw_datasets)


def eval(args, model, raw_datasets):
    tst_input_ids = raw_datasets["test_input_ids"]
    tst_labels = raw_datasets["test_labels"]
    test_set = DataIterator(args, tst_input_ids, tst_labels)

    model.eval()

    label_pred = []
    pred_prob = []
    targets = []
    # for i in track(range(test_set.sample_num)):
    for i in range(test_set.sample_num):
        # get input and target
        input_ids, target = test_set.get_index(i)
        # model process
        pred = model(input_ids)[0, 1:-1]

        pred = F.softmax(pred, dim=-1)

        out_pred = F.softmax(pred, dim=-1)
        out_pred = out_pred.reshape([-1, pred.shape[-1]])
        label_p = torch.argmax(pred, dim=-1).squeeze(0).cpu().detach().numpy()
        pred = out_pred[:, 1].cpu().detach().numpy()
        target = target.reshape([-1]).long().cpu().detach().numpy()

        label_pred.extend(label_p)
        pred_prob.extend(pred)
        targets.extend(target)
    # print(label_pred)
    # print(pred_prob)
    # print(targets)
    metric, roc_data, prc_data = caculate_metric(label_pred, targets, pred_prob)
    ACC, Precision, Sensitivity, Specificity, F1, AUC, MCC, tp, fp, tn, fn = metric

    header = ["ACC", "Precision", "Sensitivity", "Specificity", "F1-score", "AUC", "MCC", "tp", "fp", "tn", "fn"]
    rows = [
        (ACC, Precision, Sensitivity, Specificity, F1, AUC, MCC, tp, fp, tn, fn),

    ]
    print(tabulate(rows, headers=header))