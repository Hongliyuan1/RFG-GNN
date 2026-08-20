# 内置
import os

# 为保证 CPU/DGL 运行的可复现性，必须在导入 dgl/numpy/torch 之前设置
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import csv
import hashlib
import warnings

# dgl
import dgl
import hydra

# 机器学习
import numpy as np

# torch
import torch
import torch.nn.functional as F

# 工程化、自建和其他
import wandb
from dgl import function as fn
from dgl.data.utils import load_graphs
from dgl.utils import expand_as_pair, dgl_warning
from omegaconf import DictConfig, OmegaConf

# pl
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Timer
from pytorch_lightning.loggers.wandb import WandbLogger

from sklearn.metrics import roc_auc_score, average_precision_score
from torch import nn
from tqdm import tqdm

from myutils import (
    describe,
    mask_to_index,
    set_all_seed,
    cal_metrics,
    bin_encoding2,
    woe_encoding,
    iv_encoding,
)

warnings.filterwarnings("ignore")
print(os.getcwd())


class IntraConv_single(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        aggregator_type,
        feat_drop=0.0,
        add_self=True,
        bias=True,
        norm=None,
        activation=None,
    ):
        super(IntraConv_single, self).__init__()

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.add_self = add_self
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation

        self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)
        self.bias = nn.parameter.Parameter(torch.zeros(self._out_feats))
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def _compatibility_check(self):
        """Address the backward compatibility issue brought by #2747"""
        if not hasattr(self, "bias"):
            dgl_warning(
                "You are loading a GraphSAGE model trained from a old version of DGL, "
                "DGL automatically convert it to be compatible with latest version."
            )
            bias = self.fc_neigh.bias
            self.fc_neigh.bias = None
            if hasattr(self, "fc_self"):
                if bias is not None:
                    bias = bias + self.fc_self.bias
                    self.fc_self.bias = None
            self.bias = bias

    def forward(self, graph, feat, etype=None, edge_weight=None):
        self._compatibility_check()

        with graph.local_scope():
            if isinstance(feat, tuple):
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
            else:
                feat_src = feat_dst = self.feat_drop(feat)
                if graph.is_block:
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]

            msg_fn = fn.copy_u("h", "m")

            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.srcdata["degree"] = torch.ones(
                    (graph.num_src_nodes(), 1)
                ).to(feat.device)
                graph.edata["_edge_weight"] = edge_weight
                msg_fn1 = fn.u_mul_e("h", "_edge_weight", "m")
                msg_fn2 = fn.u_mul_e("degree", "_edge_weight", "degree")

            h_self = feat_dst

            if graph.number_of_edges() == 0:
                graph.dstdata["neigh"] = torch.zeros(
                    feat_dst.shape[0], self._in_src_feats
                ).to(feat_dst)

            lin_before_mp = self._in_src_feats > self._out_feats

            graph.srcdata["h"] = (
                self.fc_neigh(feat_src) if lin_before_mp else feat_src
            )

            if edge_weight is not None:
                graph.update_all(msg_fn1, fn.sum("m", "neigh"))
                graph.update_all(msg_fn2, fn.sum("degree", "degree"))
                h_neigh = graph.dstdata["neigh"] / (
                    graph.dstdata["degree"]
                    + torch.FloatTensor([1e-8]).to(feat.device)
                )
            else:
                graph.update_all(msg_fn, fn.mean("m", "neigh"))
                h_neigh = graph.dstdata["neigh"]

            if not lin_before_mp:
                h_neigh = self.fc_neigh(h_neigh)

            h_self = self.fc_self(h_self)

            if self.add_self:
                rst = h_self + h_neigh
            else:
                rst = h_neigh

            if self.bias is not None:
                rst = rst + self.bias

            if self.activation is not None:
                rst = self.activation(rst)

            if self.norm is not None:
                rst = self.norm(rst)

            return rst


class IntraConv_multi(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        aggregator_type,
        feat_drop=0.0,
        add_self=True,
        bias=True,
        norm=None,
        activation=None,
    ):
        super(IntraConv_multi, self).__init__()

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.add_self = add_self
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation

        self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)
        self.bias = nn.parameter.Parameter(torch.zeros(self._out_feats))
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def _compatibility_check(self):
        """Address the backward compatibility issue brought by #2747"""
        if not hasattr(self, "bias"):
            dgl_warning(
                "You are loading a GraphSAGE model trained from a old version of DGL, "
                "DGL automatically convert it to be compatible with latest version."
            )
            bias = self.fc_neigh.bias
            self.fc_neigh.bias = None
            if hasattr(self, "fc_self"):
                if bias is not None:
                    bias = bias + self.fc_self.bias
                    self.fc_self.bias = None
            self.bias = bias

    def forward(self, graph, feat, etype, edge_weight=None):
        self._compatibility_check()

        with graph.local_scope():
            if isinstance(feat, tuple):
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
            else:
                feat_src = feat_dst = self.feat_drop(feat)
                if graph.is_block:
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]

            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges(etype=etype)
                graph.srcdata["degree"] = torch.ones(
                    (graph.num_src_nodes(), 1)
                ).to(feat.device)
                graph.edata["_edge_weight"] = {etype: edge_weight}
                msg_fn1 = fn.u_mul_e("h", "_edge_weight", "m")
                msg_fn2 = fn.u_mul_e("degree", "_edge_weight", "degree")

            h_self = feat_dst

            if graph.number_of_edges() == 0:
                graph.dstdata["neigh"] = torch.zeros(
                    feat_dst.shape[0], self._in_src_feats
                ).to(feat_dst)

            lin_before_mp = self._in_src_feats > self._out_feats
            msg_fn = fn.copy_u("h", "m")

            graph.srcdata["h"] = (
                self.fc_neigh(feat_src) if lin_before_mp else feat_src
            )

            if edge_weight is not None:
                graph.multi_update_all(
                    {etype: (msg_fn1, fn.sum("m", "neigh"))},
                    "sum",
                )
                graph.multi_update_all(
                    {etype: (msg_fn2, fn.sum("degree", "degree"))},
                    "sum",
                )
                h_neigh = graph.dstdata["neigh"] / (
                    graph.dstdata["degree"]
                    + torch.FloatTensor([1e-8]).to(feat.device)
                )
            else:
                graph.multi_update_all(
                    {etype: (msg_fn, fn.mean("m", "neigh"))},
                    "sum",
                )
                h_neigh = graph.dstdata["neigh"]

            if not lin_before_mp:
                h_neigh = self.fc_neigh(h_neigh)

            h_self = self.fc_self(h_self)

            if self.add_self:
                rst = h_self + h_neigh
            else:
                rst = h_neigh

            if self.bias is not None:
                rst = rst + self.bias

            if self.activation is not None:
                rst = self.activation(rst)

            if self.norm is not None:
                rst = self.norm(rst)

            return rst


# 构建 dataloader
class DataModule(LightningDataModule):
    def __init__(self, graph, batch_size, n_classes, seed=42):
        super().__init__()

        self.g = graph
        self.trn_idx = mask_to_index(graph.ndata["trn_msk"])
        self.val_idx = mask_to_index(graph.ndata["val_msk"])
        self.tst_idx = mask_to_index(graph.ndata["tst_msk"])

        self.trn_sampler = dgl.dataloading.NeighborSampler([-1])
        self.val_sampler = dgl.dataloading.NeighborSampler([-1])
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.seed = seed

        # 固定训练节点 shuffle 的随机序列
        self.train_generator = torch.Generator()
        self.train_generator.manual_seed(seed)

    def train_dataloader(self):
        return dgl.dataloading.DataLoader(
            self.g,
            self.trn_idx,
            self.trn_sampler,
            device="cpu",
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            use_uva=False,
            num_workers=0,
            generator=self.train_generator,
        )

    def val_dataloader(self):
        # 保留原来的全图直推式动态分组协议：
        # val_loss 只使用验证节点，但全图预测用于更新动态分组状态。
        return dgl.dataloading.DataLoader(
            self.g,
            torch.arange(self.g.num_nodes()),
            self.val_sampler,
            device="cpu",
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            use_uva=False,
            num_workers=0,
        )


class DGA(nn.Module):
    def __init__(
        self,
        in_feats,
        n_hidden,
        num_nodes,
        n_classes,
        n_etypes,
        p=0.3,
        n_head=1,
        unclear_up=0.1,
        unclear_down=0.1,
    ):
        super().__init__()

        self.dropout = nn.Dropout(p)
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_etypes = n_etypes
        self.unclear_up = unclear_up
        self.unclear_down = unclear_down
        self.register_buffer(
            "super_mask",
            torch.ones((num_nodes, self.n_classes)),
        )
        self.n_head = n_head

        self.risk_layer = nn.Linear(n_hidden, n_hidden)

        # 特征编码
        hidden_units = [n_hidden, n_hidden]
        input_size = in_feats
        all_layers = []

        for hidden_unit in hidden_units:
            layer = nn.Linear(input_size, hidden_unit)
            all_layers.append(nn.Dropout(p))
            all_layers.append(layer)
            all_layers.append(nn.BatchNorm1d(hidden_unit))
            all_layers.append(nn.ReLU())
            input_size = hidden_unit
            self.last_dim = hidden_unit

        self.emb_layer = nn.Sequential(*all_layers)

        # 分组器损失
        self.emb_layer_fc = nn.Sequential(
            nn.Linear(n_hidden, self.n_classes)
        )

        # 输出层
        self.final_fc_layer = nn.Sequential(
            nn.Linear(self.n_head * n_hidden, n_hidden // 2),
            nn.ReLU(),
            nn.Linear(n_hidden // 2, self.n_classes),
        )

        self.attn_fn = nn.Tanh()
        self.W_f = nn.Sequential(
            nn.Linear(n_hidden, n_hidden * self.n_head),
            self.attn_fn,
        )
        self.W_x = nn.Sequential(
            nn.Linear(n_hidden, n_hidden * self.n_head),
            self.attn_fn,
        )

        self.reset_parameters()

        intra_conv = IntraConv_single if n_etypes == 1 else IntraConv_multi

        dgas = []
        for _ in range(self.n_etypes):
            m = nn.ModuleDict(
                {
                    "all": intra_conv(
                        self.last_dim,
                        n_hidden,
                        "mean",
                        norm=nn.BatchNorm1d(n_hidden),
                        activation=nn.ReLU(),
                        bias=False,
                    ),
                    "gp0": intra_conv(
                        self.last_dim,
                        n_hidden,
                        "mean",
                        norm=nn.BatchNorm1d(n_hidden),
                        activation=nn.ReLU(),
                        bias=False,
                        add_self=False,
                    ),
                    "gp1": intra_conv(
                        self.last_dim,
                        n_hidden,
                        "mean",
                        norm=nn.BatchNorm1d(n_hidden),
                        activation=nn.ReLU(),
                        bias=False,
                        add_self=False,
                    ),
                }
            )
            dgas.append(m)

        self.dgas = nn.ModuleList(dgas)

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=gain)
                nn.init.constant_(m.bias, 0)

    def dynamic_grouping(self, mask, block, unclear_down, unclear_up):
        mask0 = (
            (mask[:, 1] <= unclear_down)[
                block.srcdata[dgl.NID][block.edges()[0]]
            ].float()
        )
        mask1 = (
            (mask[:, 1] > unclear_up)[
                block.srcdata[dgl.NID][block.edges()[0]]
            ].float()
        )
        return mask0, mask1

    def forward(self, blocks, x):
        batch_size = blocks[-1].dstdata["feat"].shape[0]

        x = self.emb_layer(x)
        emb_out = self.emb_layer_fc(x)

        # 风险特征层
        risk_feature = self.risk_layer(x)
        x = x + 0.1 * F.relu(risk_feature)

        mask0_dict = {}
        mask1_dict = {}
        block = blocks[0]

        for etype in block.etypes:
            mask0_dict[etype], mask1_dict[etype] = self.dynamic_grouping(
                self.super_mask,
                block.edge_type_subgraph(etypes=[etype]),
                self.unclear_up,
                self.unclear_down,
            )

        h_list = []

        for idx, etype in enumerate(block.etypes):
            h_list.append(self.dgas[idx]["all"](block, x, etype))
            h_list.append(
                self.dgas[idx]["gp0"](
                    block,
                    x,
                    etype,
                    mask0_dict[etype],
                )
            )
            h_list.append(
                self.dgas[idx]["gp1"](
                    block,
                    x,
                    etype,
                    mask1_dict[etype],
                )
            )

        s_len = len(h_list)
        h_list = torch.stack(h_list, dim=1)

        h_list_proj = self.W_f(h_list).view(
            batch_size,
            s_len,
            self.n_head,
            self.n_hidden,
        )
        h_list_proj = (
            h_list_proj.permute(0, 2, 1, 3)
            .contiguous()
            .view(-1, s_len, self.n_hidden)
        )

        x_proj = self.W_x(x[:batch_size]).view(
            batch_size,
            self.n_head,
            self.n_hidden,
            1,
        )
        x_proj = x_proj.view(-1, self.n_hidden, 1)

        attention_logit = torch.bmm(h_list_proj, x_proj)
        soft_attention = F.softmax(
            attention_logit,
            dim=1,
        ).transpose(1, 2)

        # 机制可视化：当前 batch 的平均 attention
        # T-Finance n_etypes=1 时，三个通道依次为 all / gp0 / gp1
        attn_mean = soft_attention.reshape(
            batch_size,
            self.n_head,
            s_len,
        ).mean(dim=(0, 1)).detach()

        h_list_rep = h_list.repeat([self.n_head, 1, 1])
        weighted_features = torch.bmm(
            soft_attention,
            h_list_rep,
        ).squeeze(-2)

        h = weighted_features.view(batch_size, -1)
        o = self.final_fc_layer(h)

        return o, emb_out[:batch_size], attn_mean


# 构建网络结构
class pl_DGA(LightningModule):
    def __init__(
        self,
        in_feats,
        n_hidden,
        num_nodes,
        n_classes,
        n_etypes,
        lr=1e-3,
        weight_decay=5e-4,
        p=0.3,
        n_head=1,
        w=None,
        unclear_up=0.1,
        unclear_down=0.1,
        trn_idx=None,
        val_idx=None,
        tst_idx=None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.n_etypes = n_etypes
        self.n_classes = n_classes
        self.lr = lr
        self.weight_decay = weight_decay

        self.dga = DGA(
            in_feats[0],
            n_hidden,
            num_nodes,
            n_classes,
            n_etypes,
            p,
            n_head,
            unclear_up,
            unclear_down,
        )

        self.trn_idx = trn_idx
        self.val_idx = val_idx
        self.tst_idx = tst_idx
        self.unclear_up = unclear_up
        self.unclear_down = unclear_down

        self.register_buffer("w", torch.FloatTensor(w))

        # 保存每个 epoch 的全图概率
        self.ps = []

        # 上一 epoch 的实际动态分组状态
        self.prev_group_state = None

        # validation_epoch_end 只生成“下一训练 epoch 要使用”的新动态分组状态。
        # 不立即写入 self.dga.super_mask，避免 checkpoint 保存的状态
        # 与本轮 validation 指标对应的状态不一致。
        self.pending_super_mask = None

        # 论文机制图所需的 epoch 级记录
        self.visual_history = []

    def forward(self, blocks, x):
        o, emb_out, attn_mean = self.dga(blocks, x)
        return o, emb_out, attn_mean

    def training_step(self, batch, batch_idx):
        input_nodes, output_nodes, blocks = batch

        x = blocks[0].srcdata["feat"]
        y = blocks[-1].dstdata["aux_label"]

        logits, emb_logits, _ = self(blocks, x)

        loss = F.cross_entropy(
            logits,
            y,
            self.w,
        )

        self.log(
            "trn_loss0",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=len(output_nodes),
        )

        return loss

    def validation_step(self, batch, batch_idx):
        input_nodes, output_nodes, blocks = batch

        x = blocks[0].srcdata["feat"]
        y = blocks[-1].dstdata["label"]

        logits, emb_logits, attn_mean = self(blocks, x)

        # 严格协议：验证损失只使用验证集节点
        val_idx = self.val_idx.to(output_nodes.device)
        val_mask = torch.isin(output_nodes, val_idx)

        if val_mask.any():
            loss = F.cross_entropy(
                logits[val_mask],
                y[val_mask],
                self.w,
            )

            self.log(
                "val_loss",
                loss,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=int(val_mask.sum().item()),
            )

        # 返回全图预测以及 batch attention，供 epoch 结束时统计
        return logits, y, attn_mean, int(output_nodes.numel())

    def validation_epoch_end(self, outs):
        if self.trainer.sanity_checking:
            return

        y = torch.cat([x[1] for x in outs]).cpu().numpy()
        p = torch.cat([x[0] for x in outs]).softmax(-1).cpu().numpy()

        self.ps.append(p)
        prob = p[:, 1]

        # 最近 10 个 epoch 概率取平均，生成“下一训练 epoch”的动态分组状态。
        # 这里先不修改 self.dga.super_mask：
        # 当前 validation 的预测是基于“当前 super_mask”产生的，
        # checkpoint 也应保存这个与当前 validation 指标一致的状态。
        smooth_p = np.mean(self.ps[-10:], axis=0)

        self.pending_super_mask = torch.as_tensor(
            smooth_p,
            dtype=torch.float32,
        ).clone()

        group_prob = smooth_p[:, 1]

        # 0=low-risk, 1=unclear, 2=high-risk
        group_state = np.ones(len(group_prob), dtype=np.int8)
        group_state[group_prob <= self.unclear_down] = 0
        group_state[group_prob > self.unclear_up] = 2

        low_ratio = float(np.mean(group_state == 0))
        unclear_ratio = float(np.mean(group_state == 1))
        high_ratio = float(np.mean(group_state == 2))

        # 真正的动态分组迁移率
        if self.prev_group_state is None:
            group_flip_ratio = np.nan
        else:
            group_flip_ratio = float(
                np.mean(group_state != self.prev_group_state)
            )

        self.prev_group_state = group_state.copy()

        # 验证集高风险组中的真实欺诈比例，只使用验证标签
        val_idx_np = self.val_idx.detach().cpu().numpy()
        val_y = y[val_idx_np]
        val_state = group_state[val_idx_np]
        val_high_mask = val_state == 2

        if np.any(val_high_mask):
            val_high_fraud_ratio = float(
                np.mean(val_y[val_high_mask])
            )
        else:
            val_high_fraud_ratio = np.nan

        # 聚合全 validation epoch 的 attention
        attn_sum = None
        attn_n = 0

        for out in outs:
            attn_batch = out[2].detach().cpu().numpy()
            batch_n = int(out[3])

            if attn_sum is None:
                attn_sum = attn_batch * batch_n
            else:
                attn_sum += attn_batch * batch_n

            attn_n += batch_n

        if attn_sum is not None and attn_n > 0:
            attn_epoch = attn_sum / attn_n
        else:
            attn_epoch = None

        if attn_epoch is not None and len(attn_epoch) == 3:
            attn_all = float(attn_epoch[0])
            attn_low = float(attn_epoch[1])
            attn_high = float(attn_epoch[2])
        else:
            attn_all = np.nan
            attn_low = np.nan
            attn_high = np.nan

        trn_idx_np = self.trn_idx.detach().cpu().numpy()

        trn_auc = roc_auc_score(
            y[trn_idx_np],
            prob[trn_idx_np],
        )
        val_auc = roc_auc_score(
            y[val_idx_np],
            prob[val_idx_np],
        )

        trn_aps = average_precision_score(
            y[trn_idx_np],
            prob[trn_idx_np],
        )
        val_aps = average_precision_score(
            y[val_idx_np],
            prob[val_idx_np],
        )

        pmean = float(prob.mean())

        # 写入 Lightning/WandB
        self.log(
            "low_ratio",
            low_ratio,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "unclear_ratio",
            unclear_ratio,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "high_ratio",
            high_ratio,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        if not np.isnan(group_flip_ratio):
            self.log(
                "group_flip_ratio",
                group_flip_ratio,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )

        if not np.isnan(val_high_fraud_ratio):
            self.log(
                "val_high_fraud_ratio",
                val_high_fraud_ratio,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )

        if not np.isnan(attn_all):
            self.log(
                "attn_all",
                attn_all,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                "attn_low",
                attn_low,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                "attn_high",
                attn_high,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )

        self.log(
            "pmean",
            pmean,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "trn_auc",
            trn_auc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_auc",
            val_auc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "trn_aps",
            trn_aps,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_aps",
            val_aps,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        self.visual_history.append(
            {
                "epoch": int(self.trainer.current_epoch),
                "low_ratio": low_ratio,
                "unclear_ratio": unclear_ratio,
                "high_ratio": high_ratio,
                "group_flip_ratio": group_flip_ratio,
                "val_high_fraud_ratio": val_high_fraud_ratio,
                "pmean": pmean,
                "attn_all": attn_all,
                "attn_low": attn_low,
                "attn_high": attn_high,
                "trn_auc": float(trn_auc),
                "val_auc": float(val_auc),
                "trn_aps": float(trn_aps),
                "val_aps": float(val_aps),
            }
        )

    def on_train_epoch_start(self):
        """
        在新的训练 epoch 真正开始之前，应用上一轮 validation 得到的新动态分组状态。

        这样可以保证：
        1. Epoch t 的 validation 指标由状态 S_t 产生；
        2. Epoch t 的 checkpoint 也保存状态 S_t；
        3. validation 结束后得到的新状态 S_{t+1}，只在 Epoch t+1 训练开始时启用。
        """
        if self.pending_super_mask is not None:
            self.dga.super_mask.copy_(
                self.pending_super_mask.to(
                    self.dga.super_mask.device
                )
            )
            self.pending_super_mask = None


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.dga.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            verbose=True,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def on_train_epoch_end(self):
        if self.trainer.sanity_checking:
            return

        epoch = self.trainer.current_epoch

        metrics = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in self.trainer.logged_metrics.items()
        }

        formatted_metrics = {}

        for k, v in metrics.items():
            try:
                formatted_metrics[k] = f"{float(v):.5f}"
            except (TypeError, ValueError):
                formatted_metrics[k] = str(v)

        print(f"Epoch {epoch}: {formatted_metrics}")

    def inference(
        self,
        g,
        device,
        batch_size,
        num_workers,
        buffer_device=None,
    ):
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)

        dataloader = dgl.dataloading.DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size * 5,
            shuffle=False,
            drop_last=False,
            use_uva=False,
            num_workers=0,
        )

        if buffer_device is None:
            buffer_device = device

        y = torch.zeros(
            g.num_nodes(),
            self.n_classes,
            device=buffer_device,
        )

        for input_nodes, output_nodes, blocks in tqdm(dataloader):
            x = blocks[0].srcdata["feat"]
            logits, emb_logits, _ = self(blocks, x)
            y[output_nodes] = logits.to(buffer_device)

        return y


@hydra.main(
    config_path="configs",
    config_name="amazon",
    version_base=None,
)
def run(args: DictConfig):
    # 设置设备和随机种子
    device = torch.device(
        f"cuda:{args.gpuid}"
        if torch.cuda.is_available() and args.usegpu
        else "cpu"
    )

    accelerator = (
        "gpu"
        if torch.cuda.is_available() and args.usegpu
        else "cpu"
    )

    set_all_seed(args.seed)

    print(
        "CPU deterministic threads:",
        {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        }
    )
    print("Checkpoint dynamic-state fix: ENABLED")

    args.model = f"{args.model}"
    suffix = "_bin" if args.bin_encoding else ""
    args.model = args.model + suffix

    # logger
    mode = "offline" if args.nowandb else "online"

    wandb_run = wandb.init(
        project=args.project,
        config=OmegaConf.to_container(args),
        mode=mode,
    )

    # 读取数据
    DATA_PATH = "../../data/processed/"
    data_file = DATA_PATH + args.dname + ".dgldata"

    print("Loading data file:", os.path.abspath(data_file))

    graph, split_dict = load_graphs(data_file)
    graph = graph[0]

    y = graph.ndata["label"].cpu().numpy()
    graph.ndata["aux_label"] = graph.ndata["label"]

    w = [1, 1]
    n_classes = 2
    n_etypes = len(graph.etypes)

    trn_idx = mask_to_index(graph.ndata["trn_msk"])
    val_idx = mask_to_index(graph.ndata["val_msk"])
    tst_idx = mask_to_index(graph.ndata["tst_msk"])

    print("==" * 20)
    print("数据名称", args.dname)
    describe(graph)
    print("==" * 20)
    print("超参数设定：")
    print(OmegaConf.to_yaml(args))
    print("n_etypes", n_etypes)
    print("n_classes", n_classes)
    print("==" * 20)

    if args.bin_encoding:
        # 关键阶段 1：DTBE 编码前重新固定随机状态
        set_all_seed(args.seed)

        feature = bin_encoding2(
            graph,
            trn_idx,
            n_bins=args.k,
        )

        # 如需 WOE / IV 再手动切换：
        # feature = woe_encoding(graph, trn_idx, n_bins=args.k)
        # feature = iv_encoding(graph, trn_idx, n_bins=args.k)

        graph.ndata["feat"] = torch.FloatTensor(
            feature.values
        ).contiguous()

        # 检查每次运行得到的编码特征是否完全相同
        feature_hash = hashlib.md5(
            graph.ndata["feat"]
            .cpu()
            .numpy()
            .tobytes()
        ).hexdigest()

        print("FEATURE_MD5:", feature_hash)
        print("after bin_encoding：", feature.shape)
        print("==" * 20)

    else:
        feature = graph.ndata["feat"]
        graph.ndata["feat"] = feature.contiguous()

        feature_hash = hashlib.md5(
            graph.ndata["feat"]
            .cpu()
            .numpy()
            .tobytes()
        ).hexdigest()

        print("FEATURE_MD5:", feature_hash)
        print("after bin_encoding：", feature.shape)

    in_feats = [graph.ndata["feat"].shape[1]]

    unclear_up = args.z
    unclear_down = args.z

    wandb.log(
        {
            "unclear_up": unclear_up,
            "unclear_down": unclear_down,
        }
    )

    print(
        {
            "unclear_up": unclear_up,
            "unclear_down": unclear_down,
        }
    )

    datamodule = DataModule(
        graph,
        args.bs,
        n_classes,
        seed=args.seed,
    )

    # 关键阶段 2：模型初始化前重新固定随机状态
    set_all_seed(args.seed)

    model = pl_DGA(
        in_feats,
        args.n_hidden,
        graph.num_nodes(),
        n_classes=n_classes,
        n_etypes=n_etypes,
        lr=args.lr,
        weight_decay=args.weight_decay,
        p=args.p,
        n_head=args.n_head,
        w=w,
        unclear_up=unclear_up,
        unclear_down=unclear_down,
        trn_idx=trn_idx,
        val_idx=val_idx,
        tst_idx=tst_idx,
    )

    # 检查模型初始参数是否完全一致
    first_param = next(
        model.parameters()
    ).detach().cpu().numpy()

    model_hash = hashlib.md5(
        first_param.tobytes()
    ).hexdigest()

    print("MODEL_INIT_MD5:", model_hash)

    timer = Timer()

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        verbose=False,
    )

    early_stopping = EarlyStopping(
        "val_loss",
        verbose=False,
        mode="min",
        patience=args.patience,
    )

    logger = WandbLogger(wandb=wandb_run)

    trainer = Trainer(
        accelerator=accelerator,
        devices=(
            1
            if accelerator == "cpu"
            else [args.gpuid]
        ),
        max_epochs=args.max_epochs,
        logger=logger,
        enable_progress_bar=False,
        callbacks=[
            checkpoint_callback,
            early_stopping,
            timer,
        ],
        deterministic=True,
    )

    # 关键阶段 3：构建 DataLoader / 正式训练前再次固定随机状态
    set_all_seed(args.seed)

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    set_all_seed(args.seed)

    trainer.fit(
        model,
        train_loader,
        val_loader,
    )

    print(
        f'training time elapsed '
        f'{timer.time_elapsed("train"):.2f}s'
    )

    # 保存论文机制图数据
    visual_dir = os.path.join(
        os.getcwd(),
        "visual_logs",
    )
    os.makedirs(
        visual_dir,
        exist_ok=True,
    )

    visual_csv = os.path.join(
        visual_dir,
        f"{args.dname}_seed{args.seed}_visual.csv",
    )

    fieldnames = [
        "epoch",
        "low_ratio",
        "unclear_ratio",
        "high_ratio",
        "group_flip_ratio",
        "val_high_fraud_ratio",
        "pmean",
        "attn_all",
        "attn_low",
        "attn_high",
        "trn_auc",
        "val_auc",
        "trn_aps",
        "val_aps",
    ]

    with open(
        visual_csv,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(
            model.visual_history
        )

    print(
        "Visualization metrics saved to:",
        visual_csv,
    )

    # 读取最优 checkpoint 并推理
    print(
        "Evaluating model in",
        trainer.checkpoint_callback.best_model_path,
    )

    model.load_state_dict(
        torch.load(
            trainer.checkpoint_callback.best_model_path
        )["state_dict"]
    )

    model = model.to(device)

    with torch.no_grad():
        model.eval()
        y_hat = model.inference(
            graph,
            device,
            5120,
            0,
            "cpu",
        )

    prob = (
        y_hat.softmax(-1)
        .cpu()
        .numpy()[:, 1]
    )

    dic = cal_metrics(
        prob,
        y,
        trn_idx,
        val_idx,
        tst_idx,
        verbose=True,
    )

    wandb.log(dic)
    wandb.finish()

    print("===" * 10)
    print("===" * 10)


if __name__ == "__main__":
    run()
