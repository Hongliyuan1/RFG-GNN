from pathlib import Path

target = Path("train1221_dtbe_no_group_strict.py")
source = target.read_text(encoding="utf-8")

# 1. 删除未使用的 super_mask 缓冲区
super_mask_line = (
    "        self.register_buffer('super_mask', "
    "torch.ones((num_nodes, self.n_classes)))\n"
)
if super_mask_line in source:
    source = source.replace(super_mask_line, "", 1)

# 2. 单一全局聚合输出维度为 n_hidden，不再是 n_head * n_hidden
old_classifier = (
    "        all_layers.append("
    "nn.Linear(self.n_head * n_hidden, n_hidden // 2))"
)
new_classifier = (
    "        all_layers.append("
    "nn.Linear(n_hidden, n_hidden // 2))"
)

if old_classifier not in source:
    raise RuntimeError("未找到 final_fc_layer 输入维度设置")

source = source.replace(old_classifier, new_classifier, 1)

# 3. 删除注意力融合层 W_f、W_x
attn_start = source.find("        self.attn_fn = nn.Tanh()")
attn_end = source.find("        self.reset_parameters()", attn_start)

if attn_start == -1 or attn_end == -1:
    raise RuntimeError("未找到注意力层定义")

source = source[:attn_start] + source[attn_end:]

# 4. ModuleDict 只保留 all 全局聚合分支
dgas_start = source.find(
    "            m = nn.ModuleDict({",
    source.find("        dgas = []")
)
dgas_end = source.find("            dgas.append(m)", dgas_start)

if dgas_start == -1 or dgas_end == -1:
    raise RuntimeError("未找到 dgas ModuleDict")

new_dgas = """            m = nn.ModuleDict({
                'all': intra_conv(
                    self.last_dim,
                    n_hidden,
                    "mean",
                    norm=nn.BatchNorm1d(n_hidden),
                    activation=nn.ReLU(),
                    bias=False
                )
            })
"""

source = source[:dgas_start] + new_dgas + source[dgas_end:]

# 5. 删除 dynamic_grouping 函数
group_start = source.find("    def dynamic_grouping")
forward_start = source.find("    def forward", group_start)

if group_start == -1 or forward_start == -1:
    raise RuntimeError("未找到 dynamic_grouping 或 forward")

source = source[:group_start] + source[forward_start:]

# 6. forward 中删除 gp0、gp1、mask 和注意力融合
forward_body_start = source.find(
    "        mask0_dict = {}",
    source.find("    def forward")
)
classifier_start = source.find(
    "        o = self.final_fc_layer(h)",
    forward_body_start
)

if forward_body_start == -1 or classifier_start == -1:
    raise RuntimeError("未找到 forward 中的动态分组代码")

new_forward_body = """        # No-group ablation:
        # 仅保留每种边类型的全局邻居聚合，不使用动态分组和注意力融合
        h_all = []
        block = blocks[0]

        for idx, etype in enumerate(block.etypes):
            h_all.append(
                self.dgas[idx]['all'](block, x, etype)
            )

        if len(h_all) == 1:
            h = h_all[0]
        else:
            # 多关系图采用简单平均，不进行注意力融合
            h = torch.stack(h_all, dim=0).mean(dim=0)

"""

source = (
    source[:forward_body_start]
    + new_forward_body
    + source[classifier_start:]
)

# 7. 禁用验证阶段的动态反馈分组更新
feedback_line = (
    "            self.dga.super_mask.copy_("
    "torch.FloatTensor(np.mean(self.ps[-10:], axis=0)))"
)

if feedback_line in source:
    source = source.replace(
        feedback_line,
        "            # No-group ablation: "
        "dynamic grouping feedback is disabled",
        1
    )

target.write_text(source, encoding="utf-8")
print("No-group patch completed successfully.")
