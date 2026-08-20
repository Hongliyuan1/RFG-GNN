from pathlib import Path
import random

import dgl
import numpy as np
import torch
from dgl.data.utils import load_graphs, save_graphs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


SEED = 20230415
TRAIN_RATIO = 0.40
SECOND_SPLIT_TEST_RATIO = 0.67


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_mask(indices: torch.Tensor, num_nodes: int) -> torch.Tensor:
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[indices] = True
    return mask


def find_raw_tfinance(base_dir: Path) -> Path:
    """
    兼容两种可能的项目目录结构：
    1. FraNAD-GNN/code/data/raw/tfinance
    2. FraNAD-GNN/data/raw/tfinance
    """
    candidates = [
        base_dir.parent / "data" / "raw" / "tfinance",
        base_dir.parent.parent / "data" / "raw" / "tfinance",
    ]

    for path in candidates:
        if path.exists():
            return path

    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "没有找到 T-Finance 原始图文件。已搜索：\n"
        f"{searched}"
    )


def class_rate(labels: np.ndarray, indices: np.ndarray) -> float:
    return float(labels[indices].mean())


def main() -> None:
    set_seed(SEED)

    base_dir = Path(__file__).resolve().parent
    raw_path = find_raw_tfinance(base_dir)

    # 与当前训练脚本的数据读取路径保持一致：
    # Repository root -> data/processed
    output_dir = base_dir.parent / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_graph_path = output_dir / "tfinance_strict.dgldata"
    output_split_path = output_dir / "tfinance_strict_split.npz"
    output_scaler_path = output_dir / "tfinance_strict_scaler.npz"

    print("=" * 70)
    print("构建 T-Finance strict 数据")
    print(f"原始文件：{raw_path}")
    print(f"输出文件：{output_graph_path}")
    print("=" * 70)

    graphs, _ = load_graphs(str(raw_path))

    if not graphs:
        raise RuntimeError("T-Finance 原始文件中没有图对象。")

    raw_graph = graphs[0]
    num_nodes = raw_graph.num_nodes()

    if "feature" not in raw_graph.ndata:
        raise KeyError(
            "原始 T-Finance 图中不存在节点特征字段 'feature'。"
        )

    if "label" not in raw_graph.ndata:
        raise KeyError(
            "原始 T-Finance 图中不存在节点标签字段 'label'。"
        )

    X = (
        raw_graph.ndata["feature"]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    raw_labels = (
        raw_graph.ndata["label"]
        .detach()
        .cpu()
        .numpy()
    )

    # 兼容 one-hot 标签和一维标签
    if raw_labels.ndim == 2:
        if raw_labels.shape[1] < 2:
            raise ValueError(
                "标签为二维格式，但列数小于2，无法提取正类标签。"
            )
        y = raw_labels[:, 1]
    else:
        y = raw_labels.reshape(-1)

    y = y.astype(np.int64)

    if len(X) != num_nodes or len(y) != num_nodes:
        raise ValueError(
            "节点数、特征行数和标签数量不一致："
            f"nodes={num_nodes}, X={len(X)}, y={len(y)}"
        )

    all_indices = np.arange(num_nodes)

    # 第一次划分：训练集40%，剩余60%
    trn_idx, rest_idx, y_trn, y_rest = train_test_split(
        all_indices,
        y,
        stratify=y,
        train_size=TRAIN_RATIO,
        random_state=SEED,
        shuffle=True,
    )

    # 第二次划分：剩余60%中，67%作为测试集
    # 最终约为：
    # train 40.0%，validation 19.8%，test 40.2%
    val_idx, tst_idx, _, _ = train_test_split(
        rest_idx,
        y_rest,
        stratify=y_rest,
        test_size=SECOND_SPLIT_TEST_RATIO,
        random_state=SEED,
        shuffle=True,
    )

    trn_idx = np.sort(np.asarray(trn_idx, dtype=np.int64))
    val_idx = np.sort(np.asarray(val_idx, dtype=np.int64))
    tst_idx = np.sort(np.asarray(tst_idx, dtype=np.int64))

    # 严格标准化：
    # 只用训练集拟合均值、方差，再转换全部节点
    scaler = StandardScaler()
    scaler.fit(X[trn_idx])
    X_std = scaler.transform(X).astype(np.float32)

    trn_idx_tensor = torch.from_numpy(trn_idx).long()
    val_idx_tensor = torch.from_numpy(val_idx).long()
    tst_idx_tensor = torch.from_numpy(tst_idx).long()

    trn_msk = make_mask(trn_idx_tensor, num_nodes)
    val_msk = make_mask(val_idx_tensor, num_nodes)
    tst_msk = make_mask(tst_idx_tensor, num_nodes)

    # 检查三个集合是否互斥并覆盖全部节点
    if torch.any(trn_msk & val_msk):
        raise RuntimeError("训练集和验证集存在节点重叠。")

    if torch.any(trn_msk & tst_msk):
        raise RuntimeError("训练集和测试集存在节点重叠。")

    if torch.any(val_msk & tst_msk):
        raise RuntimeError("验证集和测试集存在节点重叠。")

    if int((trn_msk | val_msk | tst_msk).sum()) != num_nodes:
        raise RuntimeError("训练、验证、测试集合未覆盖全部节点。")

    src_nodes, dst_nodes = raw_graph.edges()

    graph = dgl.graph(
        (src_nodes, dst_nodes),
        num_nodes=num_nodes,
    )

    # 保持原 data_handle.py 的处理方式
    graph = dgl.to_bidirected(graph)
    graph.create_formats_()

    graph.ndata["feat"] = torch.from_numpy(X_std).float()
    graph.ndata["label"] = torch.from_numpy(y).long()
    graph.ndata["trn_msk"] = trn_msk
    graph.ndata["val_msk"] = val_msk
    graph.ndata["tst_msk"] = tst_msk

    split_dict = {
        "trn_msk": trn_msk,
        "val_msk": val_msk,
        "tst_msk": tst_msk,
        "trn_idx": trn_idx_tensor,
        "val_idx": val_idx_tensor,
        "tst_idx": tst_idx_tensor,
    }

    save_graphs(
        str(output_graph_path),
        graph,
        split_dict,
    )

    # 单独保存划分，方便后续审计
    np.savez(
        output_split_path,
        trn_idx=trn_idx,
        val_idx=val_idx,
        tst_idx=tst_idx,
        seed=np.asarray([SEED]),
    )

    # 保存训练集拟合得到的标准化参数
    np.savez(
        output_scaler_path,
        mean=scaler.mean_,
        scale=scaler.scale_,
        var=scaler.var_,
        n_features=np.asarray([X.shape[1]]),
    )

    print("\n数据规模：")
    print(f"节点数：{num_nodes}")
    print(f"原始边数：{raw_graph.num_edges()}")
    print(f"双向化后边数：{graph.num_edges()}")
    print(f"特征维度：{X.shape[1]}")
    print(f"正类节点数：{int(y.sum())}")
    print(f"正类比例：{y.mean():.6f}")

    print("\n数据划分：")
    print(
        f"训练集：{len(trn_idx)} "
        f"({len(trn_idx) / num_nodes:.4%})，"
        f"正类比例={class_rate(y, trn_idx):.6f}"
    )
    print(
        f"验证集：{len(val_idx)} "
        f"({len(val_idx) / num_nodes:.4%})，"
        f"正类比例={class_rate(y, val_idx):.6f}"
    )
    print(
        f"测试集：{len(tst_idx)} "
        f"({len(tst_idx) / num_nodes:.4%})，"
        f"正类比例={class_rate(y, tst_idx):.6f}"
    )

    print("\n标准化审计：")
    print("StandardScaler.fit：仅训练节点")
    print("StandardScaler.transform：全部节点")
    print(
        "标准化后训练集平均绝对均值："
        f"{np.abs(X_std[trn_idx].mean(axis=0)).mean():.8f}"
    )

    print("\n已生成：")
    print(output_graph_path)
    print(output_split_path)
    print(output_scaler_path)
    print("\nT-Finance strict 数据构建成功。")


if __name__ == "__main__":
    main()
