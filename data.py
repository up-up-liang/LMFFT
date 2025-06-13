import torch
import math
import random
from transformers import AutoTokenizer


def read_file(file, tokenizer, do_reverse=False):
    input_ids, labels = [], []
    with open(file, 'r') as f:
        lines = f.readlines()
        line_num = len(lines)
        for i in range(1, line_num):
            line = lines[i]
            seq, label = line.split()

            seq = seq.upper()
            seq = " ".join(seq)

            label = list(label)
            label = [int(_) for _ in label]

            input_ids.append(tokenizer(seq).input_ids)
            labels.append(label)

        f.close()
    return input_ids, labels


def process(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=False)
    train_input_ids, train_labels = read_file(args.train_file, tokenizer, args.do_reverse)
    test_input_ids, test_labels = read_file(args.test_file, tokenizer)
    raw_datasets = {"train_input_ids": train_input_ids, "train_labels": train_labels,
                    "test_input_ids": test_input_ids, "test_labels": test_labels}
    return raw_datasets


class DataIterator(object):
    def __init__(self, args, input_ids, labels):
        self.input_ids = input_ids
        self.labels = labels
        self.device = torch.device(args.device)
        self.sample_num = len(input_ids)
        self.batch_size = args.batch_size
        self.batch_count = math.ceil(len(input_ids) / self.batch_size)

    def get_index(self, index):
        input_ids = torch.tensor(self.input_ids[index]).unsqueeze(0).to(self.device)
        # print(input_ids)
        labels = torch.tensor(self.labels[index]).unsqueeze(0).to(self.device)
        return input_ids, labels

    def shuffle(self):
        indices = [i for i in range(self.sample_num)]
        random.shuffle(indices)

        new_input_ids = [self.input_ids[_] for _ in indices]
        self.input_ids = new_input_ids
        new_labels = [self.labels[_] for _ in indices]
        self.labels = new_labels


    def get_batch(self, index):
        input_ids = []
        labels = []

        # input
        for i in range(index * self.batch_size,
                       min((index + 1) * self.batch_size, len(self.input_ids))):
            input_ids.append(self.input_ids[i])
            labels.append(self.labels[i])

        input_ids = torch.tensor(input_ids).to(self.device)
        labels = torch.tensor(labels).to(self.device)
        return input_ids, labels