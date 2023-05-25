from typing import Union, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils as pyg_utils
from torch import Tensor
from torch.nn import Parameter, Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import (OptPairTensor, Adj, Size, OptTensor)
from torch_sparse import SparseTensor, set_diag


class MGCATConv(MessagePassing):
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = False,
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 add_self_loops: bool = False, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(MGCATConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        assert isinstance(in_channels, int) is True
        self.lin_l = Linear(in_channels, heads * out_channels, bias=False)
        self.lin_r = Linear(in_channels, heads * out_channels, bias=False)

        self.att_l = Linear(in_channels, heads, bias=False)
        self.att_r = Linear(in_channels, heads, bias=False)

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        zeros(self.bias)
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.lin_l.weight, gain=gain)
        nn.init.xavier_normal_(self.lin_r.weight, gain=gain)
        nn.init.xavier_normal_(self.att_l.weight, gain=gain)
        nn.init.xavier_normal_(self.att_r.weight, gain=gain)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, return_attention_weights=None):
        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        assert isinstance(x, Tensor) is True
        assert x.dim() == 2, ''

        x_l = self.lin_l(x).view(-1, H, C)
        x_r = self.lin_r(x).view(-1, H, C)
        alpha_l = self.att_l(x).view(-1, H)

        assert x_l is not None
        assert alpha_l is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, _ = pyg_utils.remove_self_loops(edge_index)
                edge_index, _ = pyg_utils.add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=(x_l, None), alpha=(alpha_l, None), size=size)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j, alpha_j, alpha_i, index, ptr: OptTensor, size_i: Optional[int]) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = pyg_utils.softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def update(self, inputs: Tensor) -> Tensor:
        return inputs

    def message_and_aggregate(self, adj_t: SparseTensor) -> Tensor:
        pass

    def edge_update(self) -> Tensor:
        pass

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.heads)
