import sys
sys.path.insert(0, '')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.ms_tcn import MultiScale_TemporalConv as MS_TCN
from model.mlp import MLP
from model.spatial_transformer import spatial_attention
from model.activation import activation_factory
from graph.tools import k_adjacency, normalize_adjacency_matrix


class UnfoldTemporalWindows(nn.Module):
    def __init__(self, window_size, window_stride, window_dilation=1):
        super().__init__()
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_dilation = window_dilation

        self.padding = (window_size + (window_size-1) * (window_dilation-1) - 1) // 2
        self.unfold = nn.Unfold(kernel_size=(self.window_size, 1),
                                dilation=(self.window_dilation, 1),
                                stride=(self.window_stride, 1),
                                padding=(self.padding, 0))

    def forward(self, x):
        # Input shape: (N,C,T,V), out: (N,C,T,V*window_size)
        N, C, T, V = x.shape
        x = self.unfold(x)
        # Permute extra channels from window size to the graph dimension; -1 for number of windows
        x = x.view(N, C, self.window_size, -1, V).permute(0,1,3,2,4).contiguous()
        x = x.view(N, C, -1, self.window_size * V)
        return x


class SpatialTemporal_MS_GCN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A_binary,
                 num_scales,
                 window_size,
                 disentangled_agg=True,
                 use_Ares=True,
                 residual=False,
                 dropout=0,
                 activation='relu'):

        super().__init__()
        self.num_scales = num_scales
        self.window_size = window_size
        self.use_Ares = use_Ares
        A = self.build_spatial_temporal_graph(A_binary, window_size)

        if disentangled_agg:
            A_scales = [k_adjacency(A, k, with_self=True) for k in range(num_scales)]
            A_scales = np.concatenate([normalize_adjacency_matrix(g) for g in A_scales])
        else:
            # Self-loops have already been included in A
            A_scales = [normalize_adjacency_matrix(A) for k in range(num_scales)]
            A_scales = [np.linalg.matrix_power(g, k) for k, g in enumerate(A_scales)]
            A_scales = np.concatenate(A_scales)

        self.A_scales = torch.Tensor(A_scales)
        self.V = len(A_binary)

        if use_Ares:
            self.A_res = nn.init.uniform_(nn.Parameter(torch.randn(self.A_scales.shape)), -1e-6, 1e-6)
        else:
            self.A_res = torch.tensor(0)

        self.mlp = MLP(in_channels * num_scales, [out_channels], dropout=dropout, activation='linear')

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels):
            self.residual = lambda x: x
        else:
            self.residual = MLP(in_channels, [out_channels], activation='linear')

        self.act = activation_factory(activation)

        self.num_point= 25
        self.data_normalization= True
        self.skip_conn= True
        self.drop_connect= True
        self.concat_original= True
        self.all_layers= False
        self.adjacency= False
        self.agcn= False
        self.dv= 0.25
        self.dk= 0.25
        self.Nh= 8
        self.n= 4
        self.relative= False
        self.visualization= False
        self.more_channels = False
        self.last_graph=False
        self.complete=True

        self.attention_conv = spatial_attention(in_channels=in_channels, kernel_size=1,
                                                dk=int(out_channels * self.dk),
                                                dv=int(out_channels), Nh=self.Nh, complete=self.complete,
                                                relative=self.relative,
                                                stride=1, last_graph=self.last_graph,
                                                A=A, num=self.n, more_channels=self.more_channels,
                                                drop_connect=self.drop_connect,
                                                data_normalization=self.data_normalization, skip_conn=self.skip_conn,
                                                adjacency=self.adjacency, visualization=self.visualization, num_point=self.num_point)


    def build_spatial_temporal_graph(self, A_binary, window_size):
        assert isinstance(A_binary, np.ndarray), 'A_binary should be of type `np.ndarray`'
        V = len(A_binary)
        V_large = V * window_size
        A_binary_with_I = A_binary + np.eye(len(A_binary), dtype=A_binary.dtype)
        # Build spatial-temporal graph
        A_large = np.tile(A_binary_with_I, (window_size, window_size)).copy()
        return A_large

    def forward(self, x):
        N, C, T, V = x.shape    # T = number of windows
        # print(N, C, T, V)
        # Build graphs
        A = self.A_scales.to(x.dtype).to(x.device) + self.A_res.to(x.dtype).to(x.device)

        # Perform Graph Convolution
        res = self.residual(x)
        # print('x before attention', x.size())
       
        # N, T, C, V > NT, C, 1, V
        xa = x.permute(0, 2, 1, 3).reshape(-1, C, 1, V)

        x = self.attention_conv(xa)
        # print('A after attention', A.size())
        # print('x after attention', x.size())
        
        agg = torch.einsum('vu,nctu->nctv', A, x)
        # print("agg ",agg.size())
        # print()
        # C = 96
        # T = 1 
        agg = agg.view(N*T, C, 1, self.num_scales, V)
        agg = agg.permute(0,3,1,2,4).contiguous().view(N, self.num_scales*C, T, V)
        # print("agg ",agg.size())

        out = self.mlp(agg) # TODO see whether we need CNN here (doing adding all the scale insider the msgcn block)
        # out = agg
        # print("res ",res)

        out += res
        # print("out ",out.size())
        return self.act(out)

