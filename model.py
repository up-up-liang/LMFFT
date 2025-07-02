import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import EsmConfig
from transformers import AutoModel
from transformers.models.esm.modeling_esm import EsmEncoder


# 生成旋转矩阵
def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    # 计算词向量元素两两分组之后，每组元素对应的旋转角度\theta_i
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成 token 序列索引 t = [0, 1,..., seq_len-1]
    t = torch.arange(seq_len, device=freqs.device)
    # freqs.shape = [seq_len, dim // 2]
    freqs = torch.outer(t, freqs).float()  # 计算m * \theta

    # 计算结果是个复数向量
    # 假设 freqs = [x, y]
    # 则 freqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs).cuda()
    return freqs_cis

# 旋转位置编码计算
def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor):
    batch_size, seq_len, dim = xq.shape
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)

    # 转为复数域
    xq_ = torch.view_as_complex(xq_)
    xk_ = torch.view_as_complex(xk_)

    # 应用旋转操作，然后将结果转回实数域
    xq_out = torch.view_as_real(xq_ * freqs_cis[seq_len * 2]).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis[seq_len * 2]).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.wq = nn.Linear(in_dim, out_dim)
        self.wk = nn.Linear(in_dim, out_dim)
        self.wv = nn.Linear(in_dim, out_dim)
        self.freqs_cis = precompute_freqs_cis(out_dim, 1500 * 2)

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, dim = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(batch_size, seq_len, -1)
        xk = xk.view(batch_size, seq_len, -1)
        xv = xv.view(batch_size, seq_len, -1)

        # attention 操作之前，应用旋转位置编码
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=self.freqs_cis)

        scores = torch.matmul(xq, xk.transpose(1, 2)) / math.sqrt(dim)
        scores = F.softmax(scores.float(), dim=-1)
        output = torch.matmul(scores, xv)
        return output


class TextCNN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        kernel_size = [3, 5]
        self.convs = nn.ModuleList(
            nn.Conv1d(dim, dim, kernel_size=ks, padding=ks//2, stride=1)
            for ks in kernel_size
        )
        self.bns = nn.ModuleList(
            nn.BatchNorm1d(dim)
            for i in kernel_size
        )
    def forward(self, X):
        X = X.permute(0, 2, 1)
        conved = []
        for conv, bn in zip(self.convs, self.bns):
            conved.append(bn(conv(X)))
        Y = X
        for y in conved:
            Y = torch.add(Y, y)
        Y = Y.permute(0, 2, 1)
        return F.relu(Y)

class JointTextCNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear_conv = nn.Linear(in_dim, out_dim, bias=True)
        self.action_conv = nn.ReLU()
        self.textcnn_conv = TextCNN(out_dim)

        self.linear_maxp = nn.MaxPool1d(kernel_size=in_dim//out_dim)
        self.action_maxp = nn.ReLU()
        self.textcnn_maxp = TextCNN(out_dim)
        
    def forward(self, x):
        x_conv = self.action_conv(self.linear_conv(x))
        x_conv = self.textcnn_conv(x_conv)
        return x_conv

class LinearVariation(nn.Module):
    def __init__(self, in_size):
        super(LinearVariation, self).__init__()
        kernel_sizes = [3, 5]
        self.convs = nn.ModuleList(
            nn.Conv1d(in_size, 32, kernel_size=ks, padding=ks//2, stride=1)
            for ks in kernel_sizes
        )

    def pool(self, conved):
        pooled = [F.max_pool1d(x, x.size(2)).permute(0, 2, 1) for x in conved]
        return pooled

    def forward(self, X):
        X = X.permute(0, 2, 1)
        conved = [conv(X) for conv in self.convs]
        pooled = self.pool(conved)
        w = torch.cat(pooled, dim=1)
        embs = torch.cat([torch.einsum("bsd,bld->bsl", x.permute(0, 2, 1), w) for x in conved], dim=-1)
        return embs


class JointLinearVariation(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear_conv = nn.Linear(in_dim, out_dim, bias=True)
        self.action_conv = nn.ReLU()
        self.linearvariation_conv = LinearVariation(out_dim)

        self.linear_maxp = nn.MaxPool1d(kernel_size=in_dim//out_dim)
        self.action_maxp = nn.ReLU()
        self.linearvariation_maxp = LinearVariation(out_dim)

    def forward(self, x):
        x_conv = self.action_conv(self.linear_conv(x))
        x_conv = self.linearvariation_conv(x_conv)
        return x_conv

class JointPeptide(nn.Module):
    def  __init__(self, in_dim, out_dim, ws):
        super().__init__()
        pretrained_dict = torch.load("./pkls/esm2_t33_cls_new.pkl")["dipeptides"].view(484, ws - 1, -1)
        self.dipeptides = nn.Parameter(pretrained_dict, requires_grad=True)

        self.ws = ws
        self.indices_y = torch.tensor([_ for _ in range(self.ws - 1)], device="cuda")
        # RoPE
        self.attention = Attention(in_dim, out_dim)
        # EsmEncoder
        config = EsmConfig(
            num_hidden_layers=2,
            hidden_size=128,
            num_attention_heads=8,  # 之前是8
            intermediate_size=128
        )
        self.encoder = EsmEncoder(config)
        self.gru = nn.GRU(input_size=128, hidden_size=64, bias=True, num_layers=2,
                          batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)

    def get_window_index(self, ids):
        indice = []
        for x in range(self.ws - 1):
            indice.append(ids[x]*22 + ids[x+1])
        return torch.tensor(indice, device="cuda")

    def sliding_window(self, x):
        ids = x[0, 1:-1] - 4

        ids = torch.cat([torch.full([self.ws // 2], 21, device="cuda"), ids])
        ids = torch.cat([ids, torch.full([self.ws // 2], 21, device="cuda")])

        indices = []
        for i in range(0, len(ids) - self.ws + 1):
            win_ids = ids[i:i + self.ws]
            indice = self.get_window_index(win_ids)
            indices.append(indice)
        indices = torch.stack(indices)

        return indices.long()

    def forward(self, x):
        indices = self.sliding_window(x)

        x = self.dipeptides[indices, self.indices_y].view(-1, 1280).unsqueeze(0)

        x = self.attention(x)
        x, _ = self.gru(x)
        x = self.encoder(x, return_dict=False)[0]
        x = self.dropout(x)

        return x


class LMFFT(nn.Module):
    def __init__(self, args):
        super(LMFFT, self).__init__()
        self.model = AutoModel.from_pretrained(args.model_name_or_path)

        if "esm" in args.model_name_or_path:
            with torch.no_grad():
                old_position_embeddings = self.model.embeddings.position_embeddings.weight.clone().detach()
                new_position_embeddings = self.model.embeddings.position_embeddings_test.weight.clone().detach()
                alpha = torch.tensor(0.4)
                for j in range(1500):
                    x = j // 1024 + 1
                    y = j % 1024
                    new_position_embeddings[j] = alpha * old_position_embeddings[x] + (1 - alpha) * \
                                                 old_position_embeddings[
                                                     y]
            self.model.embeddings.position_embeddings_test.weight = nn.Parameter(
                new_position_embeddings.clone().detach().requires_grad_(True))
            if "t30" in args.model_name_or_path:
                args.model_dim = 640
            else:
                args.model_dim = 1280

        self.dim = 128
        self.joint_textcnn = JointTextCNN(args.model_dim + 128, self.dim)
        self.joint_linearvariation = JointLinearVariation(args.model_dim + 128, self.dim)

        self.joint_peptide_both = JointPeptide(1280, 128, args.ws)

        self.fnn = nn.Sequential(
            nn.Linear(self.dim + 4, 64),  # +64
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, args.logit, bias=True)
        )

    def forward(self, x, logits=True): 
        model_out = self.model(x).last_hidden_state[:, 1:-1]

        out_peptide = self.joint_peptide_both(x)
        model_out = torch.cat([model_out, out_peptide], dim=-1)

        out_textcnn = self.joint_textcnn(model_out)
        out_linear_var = self.joint_linearvariation(model_out)

        out = torch.cat([out_textcnn, out_linear_var], dim=-1)
        
        if logits:
            return self.fnn(out)[0]
        else:
            out_feature = self.fnn[:3](out)
            preds = self.fnn[3:](out_feature)
            return out[0], out_feature[0], preds[0]