from pathlib import Path

import dgl
import torch
from dgl.data.utils import load_graphs, save_graphs


INPUT_FILE = Path("data/processed/tfinance_strict.dgldata")
OUTPUT_FILE = Path(
    "data/processed/tfinance_strict_isolated.dgldata"
)
BATCH_SIZE = 1_000_000


def get_index(split_dict, keys):
    for key in keys:
        if key in split_dict:
            return torch.as_tensor(
                split_dict[key], dtype=torch.long
            ).cpu()

    raise KeyError(
        f"Cannot find split index. "
        f"Candidates={keys}, actual={list(split_dict.keys())}"
    )


def main():
    graphs, split_dict = load_graphs(str(INPUT_FILE))
    graph = graphs[0]

    train_idx = get_index(
        split_dict, ["trn_idx", "train_idx", "train"]
    )
    val_idx = get_index(
        split_dict, ["val_idx", "valid_idx", "val"]
    )
    test_idx = get_index(
        split_dict, ["tst_idx", "test_idx", "test"]
    )

    split_code = torch.full(
        (graph.num_nodes(),), -1, dtype=torch.int8
    )
    split_code[train_idx] = 0
    split_code[val_idx] = 1
    split_code[test_idx] = 2

    if (split_code < 0).any():
        raise RuntimeError("Some nodes are not assigned to a split.")

    num_edges = graph.num_edges()
    kept_eid_chunks = []

    print("=" * 70)
    print("Building edge-isolated T-Finance graph")
    print(f"Input: {INPUT_FILE.resolve()}")
    print(f"Nodes: {graph.num_nodes():,}")
    print(f"Original edges: {num_edges:,}")
    print("=" * 70)

    for start in range(0, num_edges, BATCH_SIZE):
        end = min(start + BATCH_SIZE, num_edges)

        edge_ids = torch.arange(
            start, end, dtype=torch.long
        )
        src, dst = graph.find_edges(edge_ids)

        # 只保留源节点和目标节点属于同一划分的边
        keep_mask = split_code[src] == split_code[dst]
        kept_eid_chunks.append(edge_ids[keep_mask])

        print(
            f"Checked edges: {end:,}/{num_edges:,}",
            end="\r",
            flush=True,
        )

    print()

    kept_eids = torch.cat(kept_eid_chunks)
    del kept_eid_chunks

    print(f"Retained edges: {len(kept_eids):,}")
    print(f"Removed edges: {num_edges - len(kept_eids):,}")

    isolated_graph = dgl.edge_subgraph(
        graph,
        kept_eids,
        relabel_nodes=False,
        store_ids=False,
    )

    if isolated_graph.num_nodes() != graph.num_nodes():
        raise RuntimeError("Node count changed unexpectedly.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    save_graphs(
        str(OUTPUT_FILE),
        [isolated_graph],
        split_dict,
    )

    print("=" * 70)
    print("Edge-isolated graph generated successfully.")
    print(f"Output: {OUTPUT_FILE.resolve()}")
    print(f"Nodes: {isolated_graph.num_nodes():,}")
    print(f"Edges: {isolated_graph.num_edges():,}")
    print("=" * 70)


if __name__ == "__main__":
    main()
