# 内置
import os
import warnings

# dgl / hydra
import dgl
import hydra

# torch
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# 工程化、自建和其他
import wandb
from dgl.data.utils import load_graphs
from omegaconf import DictConfig, OmegaConf

# pytorch lightning
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Timer
from pytorch_lightning.loggers.wandb import WandbLogger

# metrics
from sklearn.metrics import roc_auc_score, average_precision_score

# BWGNN
from bwgnn_strict_model import BWGNNStrict

# utils
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


def make_one_batch_loader():
    """
    BWGNN is a full-graph model.  We only need one dummy batch per epoch
    so that PyTorch Lightning can drive the training/validation loops.
    The real graph, features and masks are stored in the LightningModule.
    """
    dummy = torch.zeros(1, dtype=torch.long)
    return DataLoader(
        TensorDataset(dummy),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )


class pl_BWGNN(LightningModule):
    """
    Strict-protocol BWGNN wrapper.

    Important:
    1) BWGNN performs full-graph propagation.
    2) Training loss uses training nodes only.
    3) val_loss uses validation nodes only and selects the checkpoint.
    4) Test labels are not used during training/model selection.
    """

    def __init__(
        self,
        graph,
        in_feats,
        n_hidden,
        n_classes,
        order=2,
        lr=1e-3,
        weight_decay=5e-4,
        w=None,
        trn_idx=None,
        val_idx=None,
        tst_idx=None,
    ):
        super().__init__()

        # Do not put the DGL graph or index tensors into hparams/checkpoints.
        self.save_hyperparameters(
            ignore=["graph", "trn_idx", "val_idx", "tst_idx", "w"]
        )

        self.g = graph
        self.n_classes = n_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.order = order

        self.bwgnn = BWGNNStrict(
            in_feats=in_feats,
            hidden=n_hidden,
            num_classes=n_classes,
            order=order,
            graph=None,
        )

        self.trn_idx = trn_idx.long().cpu()
        self.val_idx = val_idx.long().cpu()
        self.tst_idx = tst_idx.long().cpu()

        if w is None:
            w = [1.0, 1.0]
        self.register_buffer("w", torch.tensor(w, dtype=torch.float32))

    def _graph_on_current_device(self):
        """
        Lightning moves model parameters automatically, but not an arbitrary
        DGLGraph stored as self.g. Move the graph only when needed.
        """
        if self.g.device != self.device:
            self.g = self.g.to(self.device)
        return self.g

    def forward(self):
        g = self._graph_on_current_device()
        x = g.ndata["feat"]
        logits = self.bwgnn(g, x)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self()
        g = self._graph_on_current_device()

        y = g.ndata["aux_label"].long()
        trn_idx = self.trn_idx.to(self.device)

        loss = F.cross_entropy(
            logits[trn_idx],
            y[trn_idx],
            weight=self.w,
        )

        self.log(
            "trn_loss0",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=int(trn_idx.numel()),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self()
        g = self._graph_on_current_device()

        y = g.ndata["label"].long()
        trn_idx = self.trn_idx.to(self.device)
        val_idx = self.val_idx.to(self.device)

        # 严格协议：验证损失只使用验证集节点
        val_loss = F.cross_entropy(
            logits[val_idx],
            y[val_idx],
            weight=self.w,
        )

        self.log(
            "val_loss",
            val_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=int(val_idx.numel()),
        )

        if self.trainer.sanity_checking:
            return

        prob = logits.softmax(-1)[:, 1].detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()

        trn_np = self.trn_idx.numpy()
        val_np = self.val_idx.numpy()

        trn_auc = roc_auc_score(y_np[trn_np], prob[trn_np])
        val_auc = roc_auc_score(y_np[val_np], prob[val_np])
        trn_aps = average_precision_score(y_np[trn_np], prob[trn_np])
        val_aps = average_precision_score(y_np[val_np], prob[val_np])

        self.log(
            "trn_auc",
            trn_auc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=int(self.trn_idx.numel()),
        )
        self.log(
            "val_auc",
            val_auc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=int(self.val_idx.numel()),
        )
        self.log(
            "trn_aps",
            trn_aps,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=int(self.trn_idx.numel()),
        )
        self.log(
            "val_aps",
            val_aps,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=int(self.val_idx.numel()),
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.bwgnn.parameters(),
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

    @torch.no_grad()
    def inference(self, graph, device, buffer_device="cpu"):
        """
        Full-graph inference from the selected checkpoint.
        """
        if graph.device != device:
            graph = graph.to(device)

        self.g = graph
        self.eval()

        logits = self()
        return logits.to(buffer_device)


@hydra.main(config_path="configs", config_name="amazon", version_base=None)
def run(args: DictConfig):
    # ============================================================
    # 1. Device / random seed
    # ============================================================
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

    args.model = f"{args.model}"
    suffix = "_bin" if args.bin_encoding else ""
    args.model = args.model + suffix

    # BWGNN order: use bw_order from yaml/CLI if it exists; otherwise 2.
    order = int(args.get("bw_order", 2))

    # ============================================================
    # 2. Logger
    # ============================================================
    mode = "offline" if args.nowandb else "online"
    wandb_run = wandb.init(
        project=args.project,
        config=OmegaConf.to_container(args),
        mode=mode,
    )

    # ============================================================
    # 3. Load the EXACT strict graph
    # ============================================================
    DATA_PATH = "data/processed/"
    graphs, split_dict = load_graphs(
        DATA_PATH + args.dname + ".dgldata"
    )
    graph = graphs[0]

    # The official T-Finance BWGNN setting is homogeneous.
    # Do not silently merge edge types here because that would change
    # the already-validated strict graph protocol.
    if not graph.is_homogeneous:
        raise ValueError(
            "This strict BWGNN script expects a homogeneous DGL graph. "
            f"Current graph has edge types: {graph.etypes}. "
            "Stop here rather than silently changing graph structure."
        )

    y = graph.ndata["label"].cpu().numpy()
    graph.ndata["aux_label"] = graph.ndata["label"]

    # Keep the SAME class weighting as the current strict GCN script.
    w = [1.0, 1.0]
    n_classes = 2

    trn_idx = mask_to_index(graph.ndata["trn_msk"])
    val_idx = mask_to_index(graph.ndata["val_msk"])
    tst_idx = mask_to_index(graph.ndata["tst_msk"])

    # ============================================================
    # 4. Print protocol information
    # ============================================================
    print("==" * 20)
    print("数据名称", args.dname)
    describe(graph)
    print("==" * 20)
    print("超参数设定：")
    print(OmegaConf.to_yaml(args))
    print("n_classes", n_classes)
    print("BWGNN order", order)
    print("BWGNN hidden", args.n_hidden)
    print(
        "split sizes:",
        {
            "train": int(len(trn_idx)),
            "val": int(len(val_idx)),
            "test": int(len(tst_idx)),
        },
    )
    print("==" * 20)

    # ============================================================
    # 5. DTBE — KEEP THE SAME TRAIN-ONLY FITTING FUNCTION
    # ============================================================
    if args.bin_encoding:
        feature = bin_encoding2(
            graph,
            trn_idx,
            n_bins=args.k,
        )

        # Alternative encodings are intentionally left disabled.
        # feature = woe_encoding(graph, trn_idx, n_bins=args.k)
        # feature = iv_encoding(graph, trn_idx, n_bins=args.k)

        graph.ndata["feat"] = torch.FloatTensor(
            feature.values
        ).contiguous()

        print("after bin_encoding：", feature.shape)
        print("==" * 20)

    else:
        feature = graph.ndata["feat"]
        graph.ndata["feat"] = feature.contiguous()
        print("after bin_encoding：", feature.shape)

    in_feats = graph.ndata["feat"].shape[1]

    print(
        "[BWGNN]",
        {
            "in_feats": int(in_feats),
            "hidden": int(args.n_hidden),
            "order": order,
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
        },
    )

    # ============================================================
    # 6. Model
    # ============================================================
    model = pl_BWGNN(
        graph=graph,
        in_feats=in_feats,
        n_hidden=args.n_hidden,
        n_classes=n_classes,
        order=order,
        lr=args.lr,
        weight_decay=args.weight_decay,
        w=w,
        trn_idx=trn_idx,
        val_idx=val_idx,
        tst_idx=tst_idx,
    )

    # BWGNN is full-graph: one dummy batch = one full-graph pass.
    train_loader = make_one_batch_loader()
    val_loader = make_one_batch_loader()

    # ============================================================
    # 7. Same val_loss checkpoint / early-stopping protocol
    # ============================================================
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
        devices=1 if accelerator == "cpu" else [args.gpuid],
        max_epochs=args.max_epochs,
        logger=logger,
        enable_progress_bar=False,
        callbacks=[
            checkpoint_callback,
            early_stopping,
            timer,
        ],
    )

    # ============================================================
    # 8. Train
    # ============================================================
    trainer.fit(model, train_loader, val_loader)

    print(
        f'training time elapsed '
        f'{timer.time_elapsed("train"):.2f}s'
    )

    # ============================================================
    # 9. Load BEST validation-loss checkpoint, then test ONCE
    # ============================================================
    best_path = trainer.checkpoint_callback.best_model_path
    print("Evaluating model in", best_path)

    checkpoint = torch.load(
        best_path,
        map_location="cpu",
    )
    model.load_state_dict(checkpoint["state_dict"])

    model = model.to(device)

    with torch.no_grad():
        model.eval()
        y_hat = model.inference(
            graph,
            device=device,
            buffer_device="cpu",
        )

    prob = (
        y_hat.softmax(-1)
        .cpu()
        .numpy()[:, 1]
    )

    # EXACT same final metric function as the current strict GCN baseline.
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
