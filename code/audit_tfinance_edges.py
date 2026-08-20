from pathlib import Path

import torch
from dgl.data.utils import load_graphs


DATA_FILE = Path("data/processed/tfinance_strict.dgldata")
BATCH_SIZE = 1_000_000


def get_index(split_dict, candidates):
    for key in candidates:
        if key in split_dict:
            value = split_dict[key]
            if isinstance(value, torch.Tensor):
                return value.long().cpu()
            return torch.as_tensor(value, dtype=torch.long)
    raise KeyError(
        f"找不到划分字段，候选字段为 {candidates}；"
        f"实际字段为 {list(split_dict.keys())}"
    )


def main():
    graphs, split_dict = load_graphs(str(DATA_FILE))
    graph = graphs[0]

    train_idx = get_index(
        split_dict, ["trn_idx", "train_idx", "train", "trn"]
    )
    val_idx = get_index(
        split_dict, ["val_idx", "valid_idx", "validation_idx", "val"]
    )
    test_idx = get_index(
        split_dict, ["tst_idx", "test_idx", "test", "tst"]
    )

    num_nodes = graph.num_nodes()
    num_edges = graph.num_edges()

    split_code = torch.full((num_nodes,), -1, dtype=torch.int8)
    split_code[train_idx] = 0
    split_code[val_idx] = 1
    split_code[test_idx] = 2

    print("=" * 70)
    print("T-Finance graph split audit")
    print(f"Data file: {DATA_FILE.resolve()}")
    print(f"Split keys: {list(split_dict.keys())}")
    print(f"Nodes: {num_nodes}")
    print(f"Edges: {num_edges}")
    print(f"Train nodes: {len(train_idx)}")
    print(f"Validation nodes: {len(val_idx)}")
    print(f"Test nodes: {len(test_idx)}")
    print(f"Unassigned nodes: {(split_code < 0).sum().item()}")
    print("=" * 70)

    # 行表示源节点集合，列表示目标节点集合：
    # 0=train, 1=validation, 2=test
    matrix = torch.zeros((3, 3), dtype=torch.long)

    for start in range(0, num_edges, BATCH_SIZE):
        end = min(start + BATCH_SIZE, num_edges)
        edge_ids = torch.arange(start, end, dtype=torch.long)
        src, dst = graph.find_edges(edge_ids)

        src_group = split_code[src]
        dst_group = split_code[dst]

        valid = (src_group >= 0) & (dst_group >= 0)
        pair_code = src_group[valid].long() * 3 + dst_group[valid].long()
        counts = torch.bincount(pair_code, minlength=9).reshape(3, 3)
        matrix += counts

        print(
            f"Processed edges: {end:,}/{num_edges:,}",
            end="\r",
            flush=True,
        )

    print("\n")
    names = ["Train", "Validation", "Test"]

    print("Directed edge count matrix (source -> destination)")
    print(f"{'':14s}{'Train':>14s}{'Validation':>14s}{'Test':>14s}")
    for row, name in enumerate(names):
        print(
            f"{name:14s}"
            f"{matrix[row, 0].item():14,d}"
            f"{matrix[row, 1].item():14,d}"
            f"{matrix[row, 2].item():14,d}"
        )

    within_edges = int(torch.diag(matrix).sum().item())
    audited_edges = int(matrix.sum().item())
    cross_edges = audited_edges - within_edges

    train_test = int(matrix[0, 2] + matrix[2, 0])
    val_test = int(matrix[1, 2] + matrix[2, 1])
    train_val = int(matrix[0, 1] + matrix[1, 0])

    print("\nSummary")
    print(f"Audited edges: {audited_edges:,}")
    print(f"Within-split edges: {within_edges:,}")
    print(f"Cross-split edges: {cross_edges:,}")
    print(f"Cross-split ratio: {cross_edges / audited_edges:.6%}")
    print(f"Train <-> Validation: {train_val:,}")
    print(f"Train <-> Test: {train_test:,}")
    print(f"Validation <-> Test: {val_test:,}")

    output_file = Path("tfinance_edge_audit.txt")
    with output_file.open("w", encoding="utf-8") as file:
        file.write("T-Finance graph split audit\n")
        file.write(f"Nodes: {num_nodes}\n")
        file.write(f"Edges: {num_edges}\n")
        file.write(f"Train nodes: {len(train_idx)}\n")
        file.write(f"Validation nodes: {len(val_idx)}\n")
        file.write(f"Test nodes: {len(test_idx)}\n")
        file.write("\nDirected matrix:\n")
        file.write(str(matrix.tolist()) + "\n")
        file.write(f"Within-split edges: {within_edges}\n")
        file.write(f"Cross-split edges: {cross_edges}\n")
        file.write(
            f"Cross-split ratio: {cross_edges / audited_edges:.10f}\n"
        )
        file.write(f"Train-Test edges: {train_test}\n")
        file.write(f"Validation-Test edges: {val_test}\n")
        file.write(f"Train-Validation edges: {train_val}\n")

    print(f"\nSaved audit result: {output_file.resolve()}")


if __name__ == "__main__":
    main()
