# -*- coding: utf-8 -*-
import time
import os
import sys

from typing import Optional
from dataclasses import dataclass, field
from transformers import HfArgumentParser, set_seed

from util.data import *
from train_eval import *
import torch
from model import LMFFT

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
model_name_or_path = "./pretrained_model/esm2_t33_650M_UR50D-finetuned-secondary-structure"

@dataclass
class Arguments:
    """
    Arguments pertraining to which task to perfrom
    """
    """
    config/tokenizer
    """
    model_name_or_path: str = field(
        default=model_name_or_path,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"})

    """
    train learning argument
    """
    weight_decay: Optional[int] = field(
        default=0.3, metadata={"help": "Syn type number"})
    learning_rate: Optional[float] = field(
        default=0.00001, metadata={"help": "Syn type number"})
    dropout: Optional[float] = field(
        default=0.4, metadata={"help": "dropout layer"})
    device: Optional[str] = field(
        default="cuda", metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"})

    """
    Arguments for training and eval.
    """
    dataset: Optional[int] = field(
        default=1, metadata={"help": "Dataset number, 1 for Dataset1, 2 for Dataset2, 3 for Dataset3"})
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."})
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."}, )

    dipeptide_file: Optional[str] = field(
        default="./pkls/dipeptide.pkl", metadata={"help": ""})
    log_file: Optional[str] = field(
        default=None, metadata={"help": ""})
    model: Optional[str] = field(
        default=None, metadata={"help": ""})

    do_reverse: bool = field(
        default=False, metadata={"help": ""})

    batch_size: int = field(
        default=1, metadata={"help": "batch size"})
    model_dim: int = field(
        default=1024, metadata={"help": "the dimension of model output"})
    max_seq_len: int = field(
        default=1500, metadata={"help": "max seq length"})
    epochs: int = field(
        default=100, metadata={"help": "epochs size"})
    do_train: Optional[bool] = field(
        default=False, metadata={"help": "do train"})
    do_eval: Optional[bool] = field(
        default=False, metadata={"help": "do eval"})
    do_predict: Optional[bool] = field(
        default=False, metadata={"help": "do predict"})
    threshold: Optional[float] = field(
        default=None, metadata={"help": "do predict"})
    seed: Optional[int] = field(
        default=2000, metadata={"help": "seed"})
    ws: Optional[int] = field(
        default=3, metadata={"help": "window size"})
    
    weight: Optional[int] = field(
        default=17, metadata={"help": "window size"})
    fgm: Optional[bool] = field(
        default=17, metadata={"help": ""})
    kfold: Optional[int] = field(
        default=None, metadata={"help": "window size"})
    logit: Optional[int] = field(
        default=2, metadata={"help": ""})
    save_fature: Optional[bool] = field(
        default=False, metadata={"help": ""})
    save_feature_name: Optional[str] = field(
        default=None, metadata={"help": ""})
    conv_mode: Optional[int] = field(
        default=1, metadata={"help": ""})
    dim: int = field(
        default=32, metadata={"help": ""})

    ablation_mode: Optional[int] = field(
        default=1, metadata={"help": ""})

def main():
    parser = HfArgumentParser(Arguments)

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))[0]
    else:
        # 从命令行中分配参数
        args = parser.parse_args_into_dataclasses()[0]

    args.train_file = f"./data/Dataset{args.dataset}_train.tsv"
    args.test_file = f"./data/Dataset{args.dataset}_test.tsv"

    set_seed(args.seed)
    raw_datasets = process(args)

    # train
    if args.do_train:
        if args.kfold == None:
            train(args, raw_datasets)
        else:
            train_kfold(args, raw_datasets)

    # eval
    if args.do_eval:
        # 创建模型对象
        model = LMFFT(args).to(args.device)
        model.model.embeddings.position_embeddings = None
        state_dict = torch.load(f"./pkls/save_model_Dataset{args.dataset}.pkl")
        model.load_state_dict(state_dict)
        model.eval()
        eval(args, model, raw_datasets["test"])

if __name__ == "__main__":
    main()
