import pickle
import torch
import dgl

from sklearn.preprocessing import StandardScaler
from dgl.data.utils import save_graphs
from myutils import describe

RAW_PATH = 'data/raw/'
DATA_PATH = 'data/processed/'

print('=' * 60)
print('Building Elliptic strict dataset...')

# 读取原始 Elliptic 数据
data = pickle.load(open(RAW_PATH + 'elliptic.dat', 'rb'))

# ------------------------------------------------------------
# 1. 使用 Elliptic 原始 train / val / test mask
# ------------------------------------------------------------
trn_msk = data.train_mask.bool().reshape(-1)
val_msk = data.val_mask.bool().reshape(-1)
tst_msk = data.test_mask.bool().reshape(-1)

trn_idx = torch.where(trn_msk)[0].long()
val_idx = torch.where(val_msk)[0].long()
tst_idx = torch.where(tst_msk)[0].long()

# ------------------------------------------------------------
# 2. StandardScaler 仅在训练集上拟合
# ------------------------------------------------------------
X = data.x.detach().cpu().numpy()

scaler = StandardScaler()
scaler.fit(X[trn_idx.cpu().numpy()])

X_std = scaler.transform(X)

# ------------------------------------------------------------
# 3. 构建图
# ------------------------------------------------------------
num_nodes = data.num_nodes
src_nodes = data.edge_index[0]
dst_nodes = data.edge_index[1]

graph = dgl.graph(
    (src_nodes, dst_nodes),
    num_nodes=num_nodes
)

graph.ndata['feat'] = torch.FloatTensor(X_std)
graph.ndata['label'] = data.y.long()

graph.ndata['trn_msk'] = trn_msk
graph.ndata['val_msk'] = val_msk
graph.ndata['tst_msk'] = tst_msk

# ------------------------------------------------------------
# 4. 保存正确的划分信息
# ------------------------------------------------------------
split_dict = {
    'trn_msk': trn_msk,
    'val_msk': val_msk,
    'tst_msk': tst_msk,
    'trn_idx': trn_idx,
    'val_idx': val_idx,
    'tst_idx': tst_idx,
}

# ------------------------------------------------------------
# 5. 另存，不覆盖旧数据
# ------------------------------------------------------------
save_graphs(
    DATA_PATH + 'elliptic_of_amnet_strict.dgldata',
    graph,
    split_dict
)

describe(graph)

print('=' * 60)
print('Strict split:')
print('train:', len(trn_idx))
print('val  :', len(val_idx))
print('test :', len(tst_idx))
print('total:', num_nodes)

print('overlap train-val:', int((trn_msk & val_msk).sum()))
print('overlap train-test:', int((trn_msk & tst_msk).sum()))
print('overlap val-test:', int((val_msk & tst_msk).sum()))

print('Saved to:')
print(DATA_PATH + 'elliptic_of_amnet_strict.dgldata')
print('=' * 60)
