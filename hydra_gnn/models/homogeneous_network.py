import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.norm.batch_norm import BatchNorm
from hydra_gnn.models.utils import build_conv_layer, build_GAT_conv_layers


class HomogeneousNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, conv_block='GCN', hidden_dim=None, num_layers=None,
                 GAT_hidden_dims=None, GAT_heads=None, GAT_concats=None, dropout=0.25, **kwargs):
        """
        This HomogeneousNetwork class implements message passing on homogeneous room-object graphs.
        :param input_dim: int, input feature dimension
        :param output_dim: int, output dimension, i.e. the number of room labels
        :param hidden_dim: int, the output dimension of the graph convolution (ignored if conv_block='GAT')
        :param num_layers: int, the number of graph convolution iterations (ignored if conv_block='GAT')
        :param GAT_hidden_dims: list of int, the output dimensions of GAT convolution (ignored if conv_block!='GAT')
        :param GAT_heads: list of int, the number of attention heads of GAT convolution (ignored if conv_block!='GAT')
        :param GAT_concats GAT_heads: list of bool, concatenate output of multi-head GAT convolution (ignored if conv_block!='GAT')
        :param dropout: float, dropout ratio during training
        """
        super(HomogeneousNetwork, self).__init__()
        self.conv_block = conv_block
        self.num_layers = num_layers if conv_block[:3] != 'GAT' else len(GAT_heads)
        self.dropout = dropout

        # message passing
        self.convs = nn.ModuleList()
        if self.conv_block == 'GAT':
            self.convs = build_GAT_conv_layers(input_dim, GAT_hidden_dims + [output_dim], GAT_heads,
                GAT_concats, dropout=dropout)
        elif self.conv_block == 'GAT_edge':
            self.convs = build_GAT_conv_layers(input_dim, GAT_hidden_dims + [output_dim], GAT_heads,
                GAT_concats, dropout=dropout, edge_dim=3, add_self_loop=True, fill_value=torch.zeros(3, dtype=torch.float64))
        else:  # GAT dimensions are different than others
            self.convs.append(build_conv_layer(self.conv_block, input_dim, hidden_dim))
            for _ in range(1, self.num_layers - 1):
                self.convs.append(build_conv_layer(self.conv_block, hidden_dim, hidden_dim))
            self.convs.append(build_conv_layer(self.conv_block, hidden_dim, output_dim))

        # batch normalization
        if self.conv_block == 'GIN':
            self.batch_norms = nn.ModuleList()
            for _ in range(self.num_layers):
                self.batch_norms.append(BatchNorm(hidden_dim))


    def forward(self, data):
        x, edge_index, room_mask = data.x, data.edge_index, data.room_mask

        # x = F.dropout(x, p=self.dropout, training=self.training)
        for i in range(self.num_layers):
            if self.conv_block == 'GAT_edge':
                x = self.convs[i](x, edge_index=edge_index, edge_attr=data.edge_attr)
            else:
                x = self.convs[i](x, edge_index=edge_index)
            if i != self.num_layers - 1:    # activation and dropout, except for the last iteration
                if self.conv_block == 'GIN':
                    x = self.batch_norms[i](x)
                x = F.relu(x) if self.conv_block[:3] != 'GAT' else F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x[room_mask, :]

    def loss(self, pred, label, mask=None):
        if mask is None:
            return F.cross_entropy(pred, label)
        else:
            return F.cross_entropy(pred[mask, :], label[mask])
