import math
import numpy as np
from typing import List, Union, Callable

import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.attention import STConv


class MySTConv(STConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._graph_conv.node_dim = 2

    def forward(self, X: torch.FloatTensor, edge_index: torch.LongTensor, edge_weight: torch.FloatTensor = None,) -> torch.FloatTensor:
        r"""Forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph.

        Arg types:
            * **X** (PyTorch FloatTensor) - Sequence of node features of shape (Batch size X Input time steps X Num nodes X In channels).
            * **edge_index** (PyTorch LongTensor) - Graph edge indices.
            * **edge_weight** (PyTorch LongTensor, optional)- Edge weight vector.

        Return types:
            * **T** (PyTorch FloatTensor) - Sequence of node features.
        """
        T_0 = self._temporal_conv1(X)
        T = self._graph_conv(T_0, edge_index, edge_weight)
        # T = torch.zeros_like(T_0).to(T_0.device)
        # for b in range(T_0.size(0)):
        #     for t in range(T_0.size(1)):
        #         T[b][t] = self._graph_conv(T_0[b][t], edge_index, edge_weight)
        T = F.relu(T)
        T = self._temporal_conv2(T)
        T = T.permute(0, 2, 1, 3)
        T = self._batch_norm(T)
        T = T.permute(0, 2, 1, 3)
        return T