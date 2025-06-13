from data import *
from train_eval import *
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import time
from transformers import HfArgumentParser, set_seed

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
model_name_or_path = "../model_file/esm2_t33_650M_UR50D-finetuned-secondary-structure"

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
    save_model_path: str = field(
        default=None, metadata={"help": ""})

    """
    train learning argument
    """
    bert_lr: Optional[int] = field(
        default=0.001, metadata={"help": "Syn type number"})
    weight_decay: Optional[int] = field(
        default=0.3, metadata={"help": "Syn type number"})
    learning_rate: Optional[float] = field(
        default=0.00001, metadata={"help": "Syn type number"})
    dropout: Optional[float] = field(
        default=0.4, metadata={"help": "dropout layer"})
    device: Optional[str] = field(
        default="cuda", metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"})

    dataset: Optional[int] = field(
        default=1, metadata={"help": ""})

    """
    Arguments for training and eval.
    """
    # dataset 1
    train_file: Optional[str] = field(
        default="../data/Dataset1_train.tsv", metadata={"help": "The input training data file (a csv or JSON file)."})
    test_file: Optional[str] = field(
        default="../data/Dataset1_test.tsv",
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
        default=True, metadata={"help": "do train"})
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
    best_model_result: Optional[bool] = field(
        default=False, metadata={"help": ""})
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

    if args.dataset == 1:
        # args.train_file = "../data/Dataset1_train.tsv"
        args.train_file = "../data/Dataset1_train_cut.tsv"
        args.test_file = "../data/Dataset1_test.tsv"
    elif args.dataset == 2:
        args.train_file = "../data/Dataset2_train.tsv"
        args.test_file = "../data/Dataset2_test.tsv"
    elif args.dataset == 3:
        args.train_file = "../data/Dataset1_train.tsv"
        args.test_file = "../data/Dataset1_test.tsv"

    set_seed(args.seed)
    raw_datasets = process(args)

    if args.log_file is None:
        args.log_file = f"./logs/{time.localtime().tm_mon}-{time.localtime().tm_mday}.txt"
    if not args.log_file.endswith(".txt"):
        args.log_file = f"./logs/{args.log_file}.txt"

    # Training
    if args.do_train:
        checkpoint = None
        if args.kfold == None:
            train(args, raw_datasets)
        else:
            train_kfold(args, raw_datasets)
    
    if args.best_model_result:
        best_model_result(args, raw_datasets)

if __name__ == "__main__":
    main()
