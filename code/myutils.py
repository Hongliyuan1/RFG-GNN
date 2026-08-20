import os
import multiprocessing
import random

import dgl
import numpy as np
import pandas as pd
import toad
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics import confusion_matrix
from sklearn.metrics._ranking import _binary_clf_curve


def index_to_mask(index_list, length):
    """
    将给定的索引列表转换为一个长度为length的掩码张量。

    参数:
        index_list (list): 包含要转换为掩码的索引的列表。
        length (int): 掩码张量的长度。

    返回:
        mask (torch.Tensor): 一个长度为length的布尔掩码张量，其中给定索引的位置为True，其他位置为False。
    """
    mask = torch.zeros(length, dtype=torch.bool)
    mask[index_list] = True
    return mask


def mask_to_index(mask):
    """
    将给定的掩码张量转换为一个索引列表。

    参数:
        mask (torch.Tensor): 一个布尔掩码张量。

    返回:
        index_list (list): 包含掩码中True值对应的索引的列表。
    """
    index_list = torch.nonzero(mask).squeeze()
    return index_list


# 设置随机种子
def set_all_seed(seed):
    """
    尽可能固定 Python、NumPy、PyTorch、CUDA 和 DGL 的随机状态，
    用于提高相同 seed 下实验的可复现性。
    """

    # Python
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)

    # CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # DGL
    dgl.seed(seed)
    try:
        dgl.random.seed(seed)
    except Exception:
        pass

    # cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # CUDA 确定性配置
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # 尽量使用确定性算法。
    # warn_only=True：若某个旧版本/DGL算子无法完全确定性执行，
    # 给出 warning，而不是直接终止训练。
    try:
        torch.use_deterministic_algorithms(
            True,
            warn_only=True
        )
    except (AttributeError, TypeError):
        # 兼容较旧的 PyTorch
        pass


def describe(graph):
    # 统计输出
    num_nodes = graph.ndata['feat'].shape[0]
    num_features = graph.ndata['feat'].shape[1]

    # 计算正类（欺诈样本）和负类（非欺诈样本）的数量
    num_positive = torch.sum(graph.ndata['label'] == 1)
    num_negative = torch.sum(graph.ndata['label'] == 0)

    # 计算欺诈样本占比
    fraud_ratio = num_positive / (num_positive + num_negative)

    print(f"总样本数: {num_nodes}")
    print(f"欺诈样本数: {num_positive}")
    print(f"非欺诈样本数: {num_negative}")
    print(f"欺诈样本占比: {fraud_ratio:.2%}")
    print(f"特征个数: {num_features}")
    print()

    for etype in graph.etypes:
        subgraph = graph.edge_type_subgraph([etype])
        num_edges = subgraph.number_of_edges()
        avg_out_degree = np.mean(
            subgraph.out_degrees().numpy()
        )

        # 输出统计信息
        print(f"{etype}边关系下的统计信息:")
        print(f"边的个数: {num_edges}")
        print(f"平均出度: {avg_out_degree:.2f}")

    print("\n数据划分情况:")
    print(
        f"训练集: {graph.ndata['trn_msk'].sum()} "
        f"验证集: {graph.ndata['val_msk'].sum()} "
        f"测试集: {graph.ndata['tst_msk'].sum()}"
    )
    print(
        f"训练集: {graph.ndata['trn_msk'].sum() / num_nodes:.2%} "
        f"验证集: {graph.ndata['val_msk'].sum() / num_nodes:.2%} "
        f"测试集: {graph.ndata['tst_msk'].sum() / num_nodes:.2%}"
    )


# 计算获得最优的macrof1,gmean和对应的阈值
def get_max_macrof1_gmean(true, prob):
    fps, tps, thresholds = _binary_clf_curve(
        true,
        prob
    )

    n_pos = np.sum(true)
    n_neg = len(true) - n_pos

    fns = n_pos - tps
    tns = n_neg - fps

    f11 = 2 * tps / (
        2 * tps + fns + fps
    )

    f10 = 2 * tns / (
        2 * tns + fns + fps
    )

    marco_f1 = (
        f11 + f10
    ) / 2

    idx = np.argmax(marco_f1)

    best_marco_f1 = marco_f1[idx]
    best_marco_f1_thr = thresholds[idx]

    gmean = np.sqrt(
        tps / n_pos
        * tns / n_neg
    )

    idx = np.argmax(gmean)

    best_gmean = gmean[idx]
    best_gmean_thr = thresholds[idx]

    return (
        best_marco_f1,
        best_marco_f1_thr,
        best_gmean,
        best_gmean_thr
    )


# 计算所有metrics指标
def cal_metrics(
    prob,
    y,
    trn_idx,
    val_idx,
    tst_idx,
    verbose=False
):
    out_dic = {}

    val_th1 = 0
    val_th2 = 0

    for prefix, idx in zip(
        [
            'final_trn/',
            'final_val/',
            'final_tst/'
        ],
        [
            trn_idx,
            val_idx,
            tst_idx
        ]
    ):
        prob_ = prob[idx]
        y_ = y[idx]

        if prefix in [
            'final_trn/',
            'final_val/'
        ]:
            (
                mf1,
                th1,
                gme,
                th2
            ) = get_max_macrof1_gmean(
                y_,
                prob_
            )

            val_th1 = th1
            val_th2 = th2

            pred = np.where(
                prob_ > th1,
                1,
                0
            )

        elif 'tst' in prefix:
            th1 = val_th1
            th2 = val_th2

            pred = np.where(
                prob_ > th1,
                1,
                0
            )

            mf1 = f1_score(
                y_true=y_,
                y_pred=pred,
                average='macro'
            )

            tn, fp, fn, tp = confusion_matrix(
                y_,
                pred
            ).ravel()

            gme = np.sqrt(
                (tp / (tp + fn))
                * (tn / (tn + fp))
            )

        rec = recall_score(
            y_,
            pred
        )

        pre = precision_score(
            y_,
            pred
        )

        auc = roc_auc_score(
            y_,
            prob_
        )

        aps = average_precision_score(
            y_,
            prob_
        )

        dic = {
            f'{prefix}auc': np.round(
                auc,
                5
            ),
            f'{prefix}aps': np.round(
                aps,
                5
            ),
            f'{prefix}mf1': np.round(
                mf1,
                5
            ),
            f'{prefix}th1': np.round(
                th1,
                5
            ),
            f'{prefix}gme': np.round(
                gme,
                5
            ),
            f'{prefix}th2': np.round(
                th2,
                5
            ),
            f'{prefix}rec': np.round(
                rec,
                5
            ),
            f'{prefix}pre': np.round(
                pre,
                5
            ),
        }

        formatted_dic = {
            k: f"{v:.5f}"
            for k, v in dic.items()
        }

        if verbose == True:
            print(formatted_dic)

        out_dic.update(dic)

    return out_dic


# 决策树分箱编码
def bin_encoding2(
    graph,
    trn_idx,
    n_bins,
    BCD=False,
    col_index=None
):
    X = graph.ndata[
        'feat'
    ].numpy()

    y = graph.ndata[
        'label'
    ].numpy()

    X = pd.DataFrame(X)

    trn_X = X.iloc[
        trn_idx
    ]

    trn_y = pd.DataFrame(
        y[trn_idx]
    )

    combiner = (
        toad.transform.Combiner()
    )

    combiner.fit(
        trn_X,
        trn_y,
        method='dt',
        min_samples=0.01,
        n_bins=n_bins,
    )

    bins = combiner.export()

    if (
        col_index is None
        or col_index == 'None'
    ):
        col_index = X.columns

    bin_encoded_X = (
        combiner.transform(
            X[col_index]
        )
    )

    bin_encoded_X_dummies = (
        pd.get_dummies(
            bin_encoded_X,
            columns=col_index
        )
    )

    feature = pd.concat(
        [
            X,
            bin_encoded_X_dummies
        ],
        axis=1
    )

    feature = feature.astype(
        float
    )

    return feature


# WOE编码
def woe_encoding(
    graph,
    trn_idx,
    n_bins=10,
    min_samples=0.05,
    col_index=None
):
    """
    WOE编码：仅使用训练集确定分箱边界和WOE值，
    再使用同一套规则转换全部数据。

    参数:
    graph: DGL图对象
    trn_idx: 训练集索引
    n_bins: 分箱数量
    min_samples: 每个分箱期望的最小样本比例
    col_index: 需要编码的列索引

    返回:
    feature: 原始特征与WOE加权独热特征
    """

    # 转换节点特征和标签
    feat_tensor = graph.ndata[
        'feat'
    ]

    label_tensor = graph.ndata[
        'label'
    ]

    X = pd.DataFrame(
        feat_tensor
        .detach()
        .cpu()
        .numpy()
    )

    y = (
        label_tensor
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    # 将训练索引转换为NumPy数组
    if hasattr(
        trn_idx,
        'detach'
    ):
        trn_idx = (
            trn_idx
            .detach()
            .cpu()
            .numpy()
        )

    trn_idx = np.asarray(
        trn_idx,
        dtype=int
    )

    # 只提取训练集
    trn_X = X.iloc[
        trn_idx
    ].copy()

    trn_y = y[
        trn_idx
    ]

    if (
        col_index is None
        or col_index == 'None'
    ):
        col_index = list(
            X.columns
        )
    else:
        col_index = list(
            col_index
        )

    # 根据最小样本比例限制分箱数量
    if (
        min_samples is not None
        and 0 < min_samples < 1
    ):
        max_bins = max(
            2,
            int(
                np.floor(
                    1.0 / min_samples
                )
            )
        )

        actual_n_bins = min(
            n_bins,
            max_bins
        )

    else:
        actual_n_bins = n_bins

    alpha = 0.5
    encoded_parts = []

    for col in col_index:
        train_col = trn_X[
            col
        ]

        # 只使用训练集确定等频分箱边界
        try:
            _, bin_edges = pd.qcut(
                train_col,
                q=actual_n_bins,
                duplicates='drop',
                retbins=True
            )

        except ValueError:
            # 常数列或无法分箱的列不进行WOE编码
            continue

        bin_edges = np.unique(
            np.asarray(
                bin_edges,
                dtype=float
            )
        )

        if len(bin_edges) < 2:
            continue

        # 允许验证集和测试集出现训练范围之外的数值
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        # 训练集和全数据使用完全相同的分箱边界
        train_bins = pd.cut(
            train_col,
            bins=bin_edges,
            labels=False,
            include_lowest=True
        )

        all_bins = pd.cut(
            X[col],
            bins=bin_edges,
            labels=False,
            include_lowest=True
        )

        k = len(
            bin_edges
        ) - 1

        total_good = np.sum(
            trn_y == 0
        )

        total_bad = np.sum(
            trn_y == 1
        )

        woe_dict = {}

        for bin_val in range(k):
            mask = (
                train_bins
                .to_numpy()
                == bin_val
            )

            good_count = np.sum(
                trn_y[mask] == 0
            )

            bad_count = np.sum(
                trn_y[mask] == 1
            )

            # 标准WOE公式，并采用alpha=0.5拉普拉斯平滑
            good_dist = (
                (good_count + alpha)
                /
                (
                    total_good
                    + alpha * k
                )
            )

            bad_dist = (
                (bad_count + alpha)
                /
                (
                    total_bad
                    + alpha * k
                )
            )

            woe_dict[
                bin_val
            ] = np.log(
                good_dist
                / bad_dist
            )

        # 生成WOE加权独热表示
        encoded_col = (
            pd.DataFrame(
                index=X.index
            )
        )

        for bin_val in range(k):
            encoded_col[
                f'{col}_bin_{bin_val}'
            ] = (
                (
                    all_bins
                    == bin_val
                ).astype(float)
                * woe_dict[
                    bin_val
                ]
            )

        encoded_parts.append(
            encoded_col
        )

    # 保留原始特征，并附加WOE编码特征
    if encoded_parts:
        woe_encoded_X = (
            pd.concat(
                encoded_parts,
                axis=1
            )
        )

        feature = pd.concat(
            [
                X,
                woe_encoded_X
            ],
            axis=1
        )

    else:
        feature = X.copy()

    return feature.astype(
        float
    )


# Quantile + IV(Information Value)编码
def iv_encoding(
    graph,
    trn_idx,
    n_bins=10,
    col_index=None
):
    """
    IV编码：
    仅使用训练集确定分箱边界和每个分箱的IV贡献，
    再使用同一套分箱规则转换全部节点。

    参数:
    graph: DGL图对象
    trn_idx: 训练集索引
    n_bins: 分箱数量
    col_index: 需要编码的特征列，None表示全部列

    返回:
    feature: 原始特征与IV加权独热特征
    """

    # 转换节点特征和标签
    X = pd.DataFrame(
        graph.ndata[
            'feat'
        ]
        .detach()
        .cpu()
        .numpy()
    )

    y = (
        graph.ndata[
            'label'
        ]
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    # 将训练索引转换为NumPy数组
    if hasattr(
        trn_idx,
        'detach'
    ):
        trn_idx = (
            trn_idx
            .detach()
            .cpu()
            .numpy()
        )

    trn_idx = np.asarray(
        trn_idx,
        dtype=int
    )

    # 只提取训练集
    trn_X = X.iloc[
        trn_idx
    ].copy()

    trn_y = y[
        trn_idx
    ]

    if (
        col_index is None
        or col_index == 'None'
    ):
        col_index = list(
            X.columns
        )
    else:
        col_index = list(
            col_index
        )

    # 拉普拉斯平滑参数
    alpha = 0.5
    encoded_parts = []

    for col in col_index:
        train_col = trn_X[
            col
        ]

        # 只使用训练集确定等频分箱边界
        try:
            _, bin_edges = pd.qcut(
                train_col,
                q=n_bins,
                duplicates='drop',
                retbins=True
            )

        except ValueError:
            # 常数列或无法分箱的列直接跳过
            continue

        bin_edges = np.unique(
            np.asarray(
                bin_edges,
                dtype=float
            )
        )

        if len(bin_edges) < 2:
            continue

        # 接纳验证集和测试集中超出训练范围的取值
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        # 训练集和全数据使用完全相同的分箱边界
        train_bins = pd.cut(
            train_col,
            bins=bin_edges,
            labels=False,
            include_lowest=True
        )

        all_bins = pd.cut(
            X[col],
            bins=bin_edges,
            labels=False,
            include_lowest=True
        )

        k = len(
            bin_edges
        ) - 1

        total_good = np.sum(
            trn_y == 0
        )

        total_bad = np.sum(
            trn_y == 1
        )

        iv_dict = {}

        for bin_val in range(k):
            mask = (
                train_bins
                .to_numpy()
                == bin_val
            )

            good_count = np.sum(
                trn_y[mask] == 0
            )

            bad_count = np.sum(
                trn_y[mask] == 1
            )

            # 使用训练集分布并加入拉普拉斯平滑
            good_dist = (
                (good_count + alpha)
                /
                (
                    total_good
                    + alpha * k
                )
            )

            bad_dist = (
                (bad_count + alpha)
                /
                (
                    total_bad
                    + alpha * k
                )
            )

            # 每个分箱的IV贡献
            iv_value = (
                (
                    good_dist
                    - bad_dist
                )
                * np.log(
                    good_dist
                    / bad_dist
                )
            )

            iv_dict[
                bin_val
            ] = iv_value

        # 生成IV加权独热表示
        encoded_col = (
            pd.DataFrame(
                index=X.index
            )
        )

        for bin_val in range(k):
            encoded_col[
                f'{col}_bin_{bin_val}'
            ] = (
                (
                    all_bins
                    == bin_val
                ).astype(float)
                * iv_dict[
                    bin_val
                ]
            )

        encoded_parts.append(
            encoded_col
        )

    # 保留原始特征，并附加IV编码特征
    if encoded_parts:
        iv_encoded_X = (
            pd.concat(
                encoded_parts,
                axis=1
            )
        )

        feature = pd.concat(
            [
                X,
                iv_encoded_X
            ],
            axis=1
        )

    else:
        feature = X.copy()

    return feature.astype(
        float
    )
