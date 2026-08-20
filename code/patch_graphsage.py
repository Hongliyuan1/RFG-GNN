from pathlib import Path

target = Path("train_graphsage_strict.py")
text = target.read_text(encoding="utf-8")

# 1. 用标准 GraphSAGE 主体替换 FraNAD-GNN/DGA 主体
model_start = text.index("class DGA(nn.Module):")
wrapper_start = text.index("class pl_DGA(LightningModule):")

graphsage_model = '''class DGA(nn.Module):
    """One-layer mean-aggregation GraphSAGE baseline."""

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

        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_etypes = n_etypes

        # Keep the same feature encoder as the main model for a fair comparison.
        self.emb_layer = nn.Sequential(
            nn.Dropout(p),
            nn.Linear(in_feats, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
        )

        if n_etypes == 1:
            intra_conv = IntraConv_single
        else:
            intra_conv = IntraConv_multi

        self.convs = nn.ModuleList([
            intra_conv(
                n_hidden,
                n_hidden,
                "mean",
                norm=nn.BatchNorm1d(n_hidden),
                activation=nn.ReLU(),
                bias=False,
            )
            for _ in range(n_etypes)
        ])

        self.final_fc_layer = nn.Sequential(
            nn.Linear(n_hidden, n_hidden // 2),
            nn.ReLU(),
            nn.Linear(n_hidden // 2, n_classes),
        )

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=gain)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, blocks, x):
        block = blocks[0]
        x = self.emb_layer(x)

        relation_outputs = []
        for index, etype in enumerate(block.etypes):
            relation_outputs.append(
                self.convs[index](block, x, etype)
            )

        # Elliptic contains one edge type. Mean fusion also supports
        # datasets containing more than one relation.
        h = torch.stack(relation_outputs, dim=1).mean(dim=1)
        logits = self.final_fc_layer(h)

        # The second return value is retained for compatibility with
        # the existing Lightning training wrapper.
        return logits, logits


'''

text = text[:model_start] + graphsage_model + text[wrapper_start:]

# 2. 删除动态分组反馈更新，只保留训练集和验证集指标
validation_start = text.index("    def validation_epoch_end(self, outs):")
optimizer_start = text.index("    def configure_optimizers(self):")

validation_method = '''    def validation_epoch_end(self, outs):
        """Report training and validation metrics only."""
        if self.trainer.sanity_checking:
            return

        y = torch.cat([item[1] for item in outs]).cpu().numpy()
        prob = (
            torch.cat([item[0] for item in outs])
            .softmax(-1)
            .cpu()
            .numpy()[:, 1]
        )

        trn_auc = roc_auc_score(y[self.trn_idx], prob[self.trn_idx])
        val_auc = roc_auc_score(y[self.val_idx], prob[self.val_idx])
        trn_aps = average_precision_score(
            y[self.trn_idx], prob[self.trn_idx]
        )
        val_aps = average_precision_score(
            y[self.val_idx], prob[self.val_idx]
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


'''

text = (
    text[:validation_start]
    + validation_method
    + text[optimizer_start:]
)

target.write_text(text, encoding="utf-8")
print("GraphSAGE strict script patched successfully:", target)
