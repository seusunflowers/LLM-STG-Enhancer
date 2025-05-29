import math
import numpy as np
from typing import List, Union, Callable

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from pytorch_tcn import TCN
import torch_geometric
from torch_geometric_temporal.nn.attention import TemporalConv
from torch_geometric_temporal.nn.recurrent import A3TGCN2, AGCRN

from submodules import MySTConv

MAX_VAR = 25
EPS = 1e-15

class BaseTCN(torch.nn.Module):
    def __init__(self, in_features, in_channels, hidden_channels:list, out_features, kernel_size:int=3, causal=True, dropout_p:float=0.5):
        super(BaseTCN, self).__init__()
        self.dropout_p = dropout_p
        self.tcn = torch.nn.Sequential(
            TCN(num_inputs=in_channels, num_channels=hidden_channels, kernel_size=kernel_size, dropout=dropout_p, causal=True),
            torch.nn.BatchNorm1d(hidden_channels[-1], eps=1e-12),
            torch.nn.GELU(),
            torch.nn.Dropout(p=self.dropout_p),
            torch.nn.Linear(in_features, 1),
            torch.nn.GELU(),
            torch.nn.Dropout(p=self.dropout_p),
        )
        self.lin1 = torch.nn.Linear(hidden_channels[-1], 128)
        self.lin2 = torch.nn.Linear(128, out_features)

    def _st_learning(self, x):
        h = self.tcn(x)  # x (_, in_channels, 12) -> (_, hidden_channels[-1], 1)
        return h.squeeze()
    
    def forward(self, x):
        batch_size, num_nodes, in_channels, in_features = x.shape
        h = self._st_learning(x.view(-1, in_channels, in_features))
        h = F.dropout(F.gelu(self.lin1(h)), p=self.dropout_p, training=self.training)
        h = self.lin2(h).view(batch_size, num_nodes, -1)
        return h


class TextTCN(BaseTCN):
    def __init__(self, num_nodes, in_features, in_channels, hidden_channels:list, out_features, in_text_dim, text_attn_layer_num:int=3, kernel_size:int=3, causal=True, dropout_p:float=0.5):
        super(TextTCN, self).__init__(in_features, in_channels, hidden_channels, out_features, kernel_size, causal, dropout_p)
        self.decode_text = torch.nn.Sequential(
            torch.nn.Dropout(p=self.dropout_p),
            torch.nn.Linear(in_text_dim, 256),
            torch.nn.GELU(),
            torch.nn.Dropout(p=self.dropout_p),
            torch.nn.Linear(256, 64),
            # torch.nn.LayerNorm([num_nodes, out_features//12, 64], eps=1e-12),
            torch.nn.GELU(),
            torch.nn.Dropout(p=self.dropout_p),
            torch.nn.Linear(64, 12),
            torch.nn.GELU(),
            torch.nn.Dropout(p=self.dropout_p),
        )
        self.text_ln1 = torch.nn.Linear(out_features, 128)
        self.text_ln2 = torch.nn.Linear(128, 128)
        self.layer_norm2 = torch.nn.LayerNorm([num_nodes, 128], eps=1e-12)
        # self.text_q_ln_list = torch.nn.ModuleList([torch.nn.Linear(out_features, out_features) for i in range (text_attn_layer_num)])
        self.text_k_ln_list = torch.nn.ModuleList([torch.nn.Linear(128, 128) for i in range (text_attn_layer_num)])
        self.text_v_ln_list = torch.nn.ModuleList([torch.nn.Linear(128, 128) for i in range (text_attn_layer_num)])
        self.text_ln_list = torch.nn.ModuleList([torch.nn.Linear(128, 128) for i in range (text_attn_layer_num)])
        self.text_layer_norm_list = torch.nn.ModuleList([torch.nn.LayerNorm([num_nodes, 128], eps=1e-12) for i in range(text_attn_layer_num)])

    def _text_learning(self, text, h):
        '''text: Tensor(Batch Size, Node Num, Hour Num, Embedding Dim)'''
        batch_size, node_num, _, _ = text.shape
        text = F.gelu(self.layer_norm2(self.text_ln1(self.decode_text(text).view(batch_size, node_num, -1))))
        for text_k_ln, text_v_ln, text_ln, text_layer_norm in zip(self.text_k_ln_list, self.text_v_ln_list, self.text_ln_list, self.text_layer_norm_list):
            text_k, text_v = text_k_ln(h), text_v_ln(h)
            h = F.dropout(F.gelu(text_layer_norm(text_v_ln(h + F.scaled_dot_product_attention(query=text, key=text_k, value=text_v)))), p=self.dropout_p, training=self.training)
        h = F.gelu(self.text_ln2(h))
        return h

    def forward(self, x, text):
        batch_size, num_nodes, in_channels, in_features = x.shape
        h = self._st_learning(x.view(-1, in_channels, in_features)).view(batch_size, num_nodes, -1)
        h = F.dropout(self._text_learning(text, h), p=self.dropout_p, training=self.training)
        h = F.dropout(F.gelu(self.lin1(h)), p=self.dropout_p, training=self.training)
        h = self.lin2(h)
        return h


class BaseSTGNN(torch.nn.Module):
    def __init__(self, num_nodes, in_channels, out_features, in_features=12, dropout_p:float=0.5, t_conv_k=3, s_hidden_channels=16, t_hidden_channels=64, final_act='linear'):
        super(BaseSTGNN, self).__init__()
        self.dropout_p = dropout_p
        self.final_act = final_act
        self.stgcn1 = MySTConv(num_nodes=num_nodes, in_channels=in_channels, hidden_channels=s_hidden_channels, out_channels=t_hidden_channels, kernel_size=t_conv_k, K=1)
        self.stgcn2 = MySTConv(num_nodes=num_nodes, in_channels=t_hidden_channels, hidden_channels=s_hidden_channels, out_channels=t_hidden_channels, kernel_size=t_conv_k, K=1)
        self.tcn = TemporalConv(in_channels=t_hidden_channels, out_channels=128, kernel_size=in_features-4*(t_conv_k-1))
        self.lin1 = torch.nn.Linear(128, 128)
        self.lin2 = torch.nn.Linear(128, out_features)
        self.layer_norm1 = torch.nn.LayerNorm([num_nodes, 128], eps=1e-12)

    def _st_learning(self, x, edge_index, edge_weight):
        h = F.dropout(F.gelu(self.stgcn1(x, edge_index, edge_weight)), p=self.dropout_p, training=self.training) # x [Batch size, 12, 207, 2] -> h [Batch size, 8, 207, 64]
        h = F.gelu(self.stgcn2(h, edge_index, edge_weight))                                                      # h [Batch size, 8, 207, 64] -> h [Batch size, 4, 207, 64]
        h = self.layer_norm1(self.tcn(h).squeeze())                                                              # h [Batch size, 4, 207, 64] -> h [Batch size, 207, 128]
        h = F.dropout(F.gelu(h), p=self.dropout_p, training=self.training)
        return h
    
    def forward(self, x, edge_index, edge_weight):
        h = self._st_learning(x, edge_index, edge_weight)
        h = F.dropout(F.gelu(self.lin1(h)), p=self.dropout_p, training=self.training)
        h = self.lin2(h)
        match self.final_act:
            case 'linear': pass
            case 'relu': h = F.relu(h)
            case _: raise ValueError('Unknown activation function type')
        return h


class TextSTGNN3(BaseSTGNN):
    def __init__(self, num_nodes, in_channels, out_features, in_text_dim, text_attn_layer_num:int=3, in_features=12, dropout_p:float=0.5, t_conv_k=3, final_act='linear'):
        super(TextSTGNN3, self).__init__(num_nodes, in_channels, out_features, in_features, dropout_p, t_conv_k, final_act=final_act)
        self.decode_text = torch.nn.Sequential(
            torch.nn.Dropout(p=self.dropout_p),
            torch.nn.Linear(in_text_dim, 256),
            torch.nn.GELU(),
            torch.nn.Dropout(p=self.dropout_p),
            torch.nn.Linear(256, 64),
            # torch.nn.LayerNorm([num_nodes, out_features//12, 64], eps=1e-12),
            torch.nn.GELU(),
            torch.nn.Dropout(p=self.dropout_p),
            torch.nn.Linear(64, 12),
            torch.nn.GELU(),
            torch.nn.Dropout(p=self.dropout_p),
        )
        self.text_ln1 = torch.nn.Linear(out_features, 128)
        self.text_ln2 = torch.nn.Linear(128, 128)
        self.layer_norm2 = torch.nn.LayerNorm([num_nodes, 128], eps=1e-12)
        # self.text_q_ln_list = torch.nn.ModuleList([torch.nn.Linear(out_features, out_features) for i in range (text_attn_layer_num)])
        self.text_k_ln_list = torch.nn.ModuleList([torch.nn.Linear(128, 128) for i in range (text_attn_layer_num)])
        self.text_v_ln_list = torch.nn.ModuleList([torch.nn.Linear(128, 128) for i in range (text_attn_layer_num)])
        self.text_ln_list = torch.nn.ModuleList([torch.nn.Linear(128, 128) for i in range (text_attn_layer_num)])
        self.text_layer_norm_list = torch.nn.ModuleList([torch.nn.LayerNorm([num_nodes, 128], eps=1e-12) for i in range(text_attn_layer_num)])

    def _text_learning(self, text, h):
        '''text: Tensor(Batch Size, Node Num, Hour Num, Embedding Dim)'''
        batch_size, node_num, _, _ = text.shape
        text = F.gelu(self.layer_norm2(self.text_ln1(self.decode_text(text).view(batch_size, node_num, -1))))
        for text_k_ln, text_v_ln, text_ln, text_layer_norm in zip(self.text_k_ln_list, self.text_v_ln_list, self.text_ln_list, self.text_layer_norm_list):
            text_k, text_v = text_k_ln(h), text_v_ln(h)
            h = F.dropout(F.gelu(text_layer_norm(text_v_ln(h + F.scaled_dot_product_attention(query=text, key=text_k, value=text_v)))), p=self.dropout_p, training=self.training)
        h = F.gelu(self.text_ln2(h))
        return h

    def forward(self, x, text, edge_index, edge_weight):
        h = self._st_learning(x, edge_index, edge_weight)
        h = F.dropout(self._text_learning(text, h), p=self.dropout_p, training=self.training)
        h = F.dropout(F.gelu(self.lin1(h)), p=self.dropout_p, training=self.training)
        h = self.lin2(h)
        # h = self.lin3(torch.cat([h, self._text_learning(text, h)], dim=-1))
        match self.final_act:
            case 'linear': pass
            case 'relu': h = F.relu(h)
            case _: raise ValueError('Unknown activation function type')
        return h


class BaseA3TGCN2(torch.nn.Module):
    def __init__(self, num_nodes:int, in_channels:int, in_periods:int, out_periods:int, batch_size:int, add_self_loops:bool=True, dropout_p:float=0.5, final_act='linear'):
        super(BaseA3TGCN2, self).__init__()
        self.dropout_p = dropout_p
        self.final_act = final_act
        # Attention Temporal Graph Convolutional Cell
        self.atgcn = A3TGCN2(in_channels=in_channels, out_channels=128, periods=in_periods, batch_size=batch_size, add_self_loops=add_self_loops)
        # Equals single-shot prediction
        self.lin1 = torch.nn.Linear(128, 128)
        self.lin2 = torch.nn.Linear(128, out_periods)
        self.layer_norm1 = torch.nn.LayerNorm([num_nodes, 128], eps=1e-12)

    def _st_learning(self, x, edge_index, edge_weight):
        h = self.layer_norm1(self.atgcn(x, edge_index, edge_weight)) # x [b, num_nodes, in_channels, 12] -> h [b, num_nodes, 128]
        h = F.dropout(F.gelu(h), p=self.dropout_p, training=self.training)
        return h
    
    def forward(self, x, edge_index, edge_weight):
        h = self._st_learning(x, edge_index, edge_weight)
        h = F.dropout(F.gelu(self.lin1(h)), p=self.dropout_p, training=self.training)
        h = self.lin2(h)
        match self.final_act:
            case 'linear': pass
            case 'relu': h = F.relu(h)
            case _: raise ValueError('Unknown activation function type')
        return h


class TextA3TGCN(BaseA3TGCN2):
    def __init__(self, num_nodes:int, in_channels:int, in_periods:int, out_periods:int, batch_size:int, in_text_dim:int, text_attn_layer_num:int=3, add_self_loops:bool=True, dropout_p:float=0.5):
        super(TextA3TGCN, self).__init__(num_nodes, in_channels, in_periods, out_periods, batch_size, add_self_loops, dropout_p)
        self.decode_text = torch.nn.Sequential(
            torch.nn.Dropout(p=self.dropout_p),
            torch.nn.Linear(in_text_dim, 256),
            torch.nn.GELU(),
            torch.nn.Dropout(p=self.dropout_p),
            torch.nn.Linear(256, 64),
            # torch.nn.LayerNorm([num_nodes, out_features//12, 64], eps=1e-12),
            torch.nn.GELU(),
            torch.nn.Dropout(p=self.dropout_p),
            torch.nn.Linear(64, 12),
            torch.nn.GELU(),
            torch.nn.Dropout(p=self.dropout_p),
        )
        self.text_ln1 = torch.nn.Linear(out_periods, 128)
        self.text_ln2 = torch.nn.Linear(128, 128)
        self.layer_norm2 = torch.nn.LayerNorm([num_nodes, 128], eps=1e-12)
        # self.text_q_ln_list = torch.nn.ModuleList([torch.nn.Linear(out_features, out_features) for i in range (text_attn_layer_num)])
        self.text_k_ln_list = torch.nn.ModuleList([torch.nn.Linear(128, 128) for i in range (text_attn_layer_num)])
        self.text_v_ln_list = torch.nn.ModuleList([torch.nn.Linear(128, 128) for i in range (text_attn_layer_num)])
        self.text_ln_list = torch.nn.ModuleList([torch.nn.Linear(128, 128) for i in range (text_attn_layer_num)])
        self.text_layer_norm_list = torch.nn.ModuleList([torch.nn.LayerNorm([num_nodes, 128], eps=1e-12) for i in range(text_attn_layer_num)])

    def _text_learning(self, text, h):
        '''text: Tensor(Batch Size, Node Num, Hour Num, Embedding Dim)'''
        batch_size, node_num, _, _ = text.shape
        text = F.gelu(self.layer_norm2(self.text_ln1(self.decode_text(text).view(batch_size, node_num, -1))))
        for text_k_ln, text_v_ln, text_ln, text_layer_norm in zip(self.text_k_ln_list, self.text_v_ln_list, self.text_ln_list, self.text_layer_norm_list):
            text_k, text_v = text_k_ln(h), text_v_ln(h)
            h = F.dropout(F.gelu(text_layer_norm(text_v_ln(h + F.scaled_dot_product_attention(query=text, key=text_k, value=text_v)))), p=self.dropout_p, training=self.training)
        h = F.gelu(self.text_ln2(h))
        return h

    def forward(self, x, text, edge_index, edge_weight):
        h = self._st_learning(x, edge_index, edge_weight)
        h = F.dropout(self._text_learning(text, h), p=self.dropout_p, training=self.training)
        h = F.dropout(F.gelu(self.lin1(h)), p=self.dropout_p, training=self.training)
        h = self.lin2(h)
        # h = self.lin3(torch.cat([h, self._text_learning(text, h)], dim=-1))
        return h


class BaseAGCRN(torch.nn.Module):
    def __init__(self, num_nodes:int, in_channels:int, in_periods:int, gru_channels:int, out_periods:int, K:int, embedding_dimensions:int, dropout_p:float=0.5, final_act='linear'):
        super(BaseAGCRN, self).__init__()
        self.dropout_p = dropout_p
        self.final_act = final_act
        self.in_periods = in_periods # 12
        self.e = torch.nn.Parameter(torch.empty(num_nodes, embedding_dimensions))
        self.agcrn = AGCRN(number_of_nodes=num_nodes, in_channels=in_channels, out_channels=gru_channels, K=K, embedding_dimensions=embedding_dimensions)
        # Equals single-shot prediction
        # self.lin0 = torch.nn.Linear(gru_channels, 128)
        self.lin1 = torch.nn.Linear(gru_channels, gru_channels)
        self.lin2 = torch.nn.Linear(gru_channels, out_periods)
        self.layer_norm1 = torch.nn.LayerNorm([num_nodes, gru_channels], eps=1e-12)
        # 参数初始化
        torch.nn.init.xavier_uniform_(self.e)

    def _st_learning(self, x:torch.FloatTensor):
        '''x [b, num_nodes, in_channels, in_periods]'''
        h = None
        for period in range(self.in_periods):
            # e = F.dropout(self.e, p=self.dropout_p, training=self.training)
            h = self.agcrn(x[:, :, :, period], self.e, h)
        h = F.dropout(F.gelu(self.layer_norm1(h)), p=self.dropout_p, training=self.training)
        return h
    
    def forward(self, x):
        h = self._st_learning(x)
        h = F.dropout(F.gelu(self.lin1(h)), p=self.dropout_p, training=self.training)
        h = self.lin2(h)
        match self.final_act:
            case 'linear': pass
            case 'relu': h = F.relu(h)
            case _: raise ValueError('Unknown activation function type')
        return h


class TextAGCRN(BaseAGCRN):
    def __init__(self, num_nodes:int, in_channels:int, in_periods:int, gru_channels:int, out_periods:int, K:int, embedding_dimensions:int, in_text_dim:int, text_attn_layer_num:int=3, dropout_p:float=0.5):
        super(TextAGCRN, self).__init__(num_nodes, in_channels, in_periods, gru_channels, out_periods, K, embedding_dimensions, dropout_p)
        self.decode_text = torch.nn.Sequential(
            torch.nn.Dropout(p=self.dropout_p),
            torch.nn.Linear(in_text_dim, 256),
            torch.nn.GELU(),
            torch.nn.Dropout(p=self.dropout_p),
            torch.nn.Linear(256, 64),
            # torch.nn.LayerNorm([num_nodes, out_features//12, 64], eps=1e-12),
            torch.nn.GELU(),
            torch.nn.Dropout(p=self.dropout_p),
            torch.nn.Linear(64, 12),
            torch.nn.GELU(),
            torch.nn.Dropout(p=self.dropout_p),
        )
        self.text_ln1 = torch.nn.Linear(out_periods, gru_channels)
        self.text_ln2 = torch.nn.Linear(gru_channels, gru_channels)
        self.layer_norm2 = torch.nn.LayerNorm([num_nodes, gru_channels], eps=1e-12)
        # self.text_q_ln_list = torch.nn.ModuleList([torch.nn.Linear(out_features, out_features) for i in range (text_attn_layer_num)])
        self.text_k_ln_list = torch.nn.ModuleList([torch.nn.Linear(gru_channels, gru_channels) for i in range (text_attn_layer_num)])
        self.text_v_ln_list = torch.nn.ModuleList([torch.nn.Linear(gru_channels, gru_channels) for i in range (text_attn_layer_num)])
        self.text_ln_list = torch.nn.ModuleList([torch.nn.Linear(gru_channels, gru_channels) for i in range (text_attn_layer_num)])
        self.text_layer_norm_list = torch.nn.ModuleList([torch.nn.LayerNorm([num_nodes, gru_channels], eps=1e-12) for i in range(text_attn_layer_num)])

    def _text_learning(self, text, h):
        '''text: Tensor(Batch Size, Node Num, Hour Num, Embedding Dim)'''
        batch_size, node_num, _, _ = text.shape
        text = F.gelu(self.layer_norm2(self.text_ln1(self.decode_text(text).view(batch_size, node_num, -1))))
        for text_k_ln, text_v_ln, text_ln, text_layer_norm in zip(self.text_k_ln_list, self.text_v_ln_list, self.text_ln_list, self.text_layer_norm_list):
            text_k, text_v = text_k_ln(h), text_v_ln(h)
            h = F.dropout(F.gelu(text_layer_norm(text_v_ln(h + F.scaled_dot_product_attention(query=text, key=text_k, value=text_v)))), p=self.dropout_p, training=self.training)
            # h = F.dropout(F.gelu(text_layer_norm(h + F.scaled_dot_product_attention(query=text, key=text_k, value=text_v))), p=self.dropout_p, training=self.training)
        h = F.gelu(self.text_ln2(h))
        return h

    def forward(self, x, text):
        h = self._st_learning(x)
        h = F.dropout(self._text_learning(text, h), p=self.dropout_p, training=self.training)
        h = F.dropout(F.gelu(self.lin1(h)), p=self.dropout_p, training=self.training)
        h = self.lin2(h)
        # h = self.lin3(torch.cat([h, self._text_learning(text, h)], dim=-1))
        return h