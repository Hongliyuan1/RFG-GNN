import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn


def calculate_theta(order: int):
    """
    Calculate the polynomial coefficients of the Beta-wavelet filters.

    This reproduces the coefficient construction used by BWGNN for
    (x/2)^i * (1 - x/2)^(order-i) / Beta(i+1, order+1-i),
    but uses only Python's math module, so sympy/scipy are not required.
    """
    thetas = []

    for i in range(order + 1):
        beta_value = (
            math.gamma(i + 1)
            * math.gamma(order + 1 - i)
            / math.gamma(order + 2)
        )

        coeff = [0.0 for _ in range(order + 1)]

        for j in range(order - i + 1):
            degree = i + j
            coeff[degree] = (
                math.comb(order - i, j)
                * ((-1.0) ** j)
                / (2.0 ** degree)
                / beta_value
            )

        thetas.append(coeff)

    return thetas


class PolyConv(nn.Module):
    """
    Polynomial graph convolution used by BWGNN.
    """

    def __init__(self, theta, activation=F.leaky_relu):
        super().__init__()
        self.theta = list(theta)
        self.k = len(self.theta)
        self.activation = activation

    def forward(self, graph, feat):
        def unnormalized_laplacian(x, d_inv_sqrt):
            # Lx = x - D^{-1/2} A D^{-1/2} x
            graph.ndata["_bw_h"] = x * d_inv_sqrt

            graph.update_all(
                fn.copy_u("_bw_h", "m"),
                fn.sum("m", "_bw_h"),
            )

            aggregated = graph.ndata.pop("_bw_h")
            return x - aggregated * d_inv_sqrt

        with graph.local_scope():
            d_inv_sqrt = (
                torch.pow(
                    graph.in_degrees().float().clamp(min=1),
                    -0.5,
                )
                .unsqueeze(-1)
                .to(feat.device)
            )

            x = feat
            h = self.theta[0] * x

            for k in range(1, self.k):
                x = unnormalized_laplacian(
                    x,
                    d_inv_sqrt,
                )
                h = h + self.theta[k] * x

            if self.activation is not None:
                h = self.activation(h)

            return h


class BWGNNStrict(nn.Module):
    """
    BWGNN adapted for the strict FraNAD/RFG experimental pipeline.

    The class supports both:
        logits = model(features)
    when graph was provided during __init__, and:
        logits = model(graph, features)
    when graph is provided during forward.
    """

    def __init__(
        self,
        in_feats,
        hidden=128,
        num_classes=2,
        order=2,
        graph=None,
    ):
        super().__init__()

        self.graph = graph
        self.hidden = hidden
        self.order = order

        thetas = calculate_theta(order)

        self.linear1 = nn.Linear(in_feats, hidden)
        self.linear2 = nn.Linear(hidden, hidden)

        self.convs = nn.ModuleList(
            [
                PolyConv(theta)
                for theta in thetas
            ]
        )

        self.linear3 = nn.Linear(
            hidden * len(self.convs),
            hidden,
        )
        self.linear4 = nn.Linear(
            hidden,
            num_classes,
        )

        self.act = nn.ReLU()

    def forward(self, graph_or_features, features=None):
        # Case 1: model(features)
        if features is None:
            features = graph_or_features
            graph = self.graph

            if graph is None:
                raise ValueError(
                    "BWGNNStrict needs a graph. "
                    "Pass graph=... in __init__ or call model(graph, features)."
                )

        # Case 2: model(graph, features)
        else:
            graph = graph_or_features

        h = self.linear1(features)
        h = self.act(h)

        h = self.linear2(h)
        h = self.act(h)

        wavelet_outputs = [
            conv(graph, h)
            for conv in self.convs
        ]

        h = torch.cat(
            wavelet_outputs,
            dim=-1,
        )

        h = self.linear3(h)
        h = self.act(h)
        logits = self.linear4(h)

        return logits
