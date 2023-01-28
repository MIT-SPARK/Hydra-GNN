from hydra_gnn.models.utils import build_hetero_conv, build_GAT_hetero_conv
from hydra_gnn.mp3d_dataset import EDGE_TYPES
import torch
import torch.nn as nn
import torch.nn.functional as F


class HeterogeneousNetwork(nn.Module):
    def __init__(self, input_dim_dict, output_dim, conv_block='GraphSAGE', hidden_dim=None, num_layers=None,
                 GAT_hidden_dims=None, GAT_heads=None, GAT_concats=None, dropout=0.25, **kwargs):
        """
        This HeterogeneousNetwork class implements message passing on heterogenoues room-object graphs.
        :param input_dim_dict: dictionary of node type to input feature dimension mapping
        :param output_dim: int, output dimension, i.e. the number of room labels
        :param hidden_dim: int, the output dimension of the graph convolution (ignored if conv_block='GAT')
        :param num_layers: int, the number of graph convolution iterations (ignored if conv_block='GAT')
        :param GAT_hidden_dims: list of int, the output dimensions of GAT convolution (ignored if conv_block!='GAT')
        :param GAT_heads: list of int, the number of attention heads of GAT convolution (ignored if conv_block!='GAT')
        :param GAT_concats GAT_heads: list of bool, concatenate output of multi-head GAT convolution (ignored if conv_block!='GAT')
        :param dropout: float, dropout ratio during training
        """
        super(HeterogeneousNetwork, self).__init__()
        assert conv_block in ['GraphSAGE', 'GAT', 'GAT_edge', 'PointTransformer']
        self.conv_block = conv_block
        self.num_layers = num_layers if conv_block[:3] != 'GAT' else len(GAT_heads)
        self.dropout = dropout

        # message passing
        hidden_dim_dict = {'rooms': hidden_dim, 'objects':hidden_dim}
        output_dim_dict = {'rooms': output_dim, 'objects':output_dim}
        if self.conv_block == 'GAT':
            self.convs = build_GAT_hetero_conv(EDGE_TYPES, input_dim_dict, output_dim_dict, 
                GAT_hidden_dims, GAT_heads, GAT_concats, dropout)
        elif self.conv_block == 'GAT_edge':
            self.convs = build_GAT_hetero_conv(EDGE_TYPES, input_dim_dict, output_dim_dict, 
                GAT_hidden_dims, GAT_heads, GAT_concats, dropout, edge_dim=3, 
                fill_value=torch.zeros(3, dtype=torch.float64))
        else:
            self.convs = nn.ModuleList()
            self.convs.append(build_hetero_conv(conv_block, EDGE_TYPES, input_dim_dict, hidden_dim_dict))
            for _ in range(1, self.num_layers - 1):
                self.convs.append(build_hetero_conv(conv_block, EDGE_TYPES, hidden_dim_dict, hidden_dim_dict))
            self.convs.append(build_hetero_conv(conv_block, EDGE_TYPES, hidden_dim_dict, output_dim_dict))            

    def forward(self, data):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict

        # x = F.dropout(x, p=self.dropout, training=self.training)
        for i in range(self.num_layers):
            if self.conv_block == 'GAT_edge':
                x_dict = self.convs[i](x_dict, edge_index_dict, data.edge_attr_dict)
            elif self.conv_block == 'PointTransformer':
                # todo: siyi - this does not work yet
                x_dict = self.convs[i](x_dict, edge_index_dict, data.pos_dict)
            else:
                x_dict = self.convs[i](x_dict, edge_index_dict)
            
            if i != self.num_layers - 1:    # activation and dropout, except for the last iteration
                x_dict = {key: F.relu(x) for key, x in x_dict.items()} if self.conv_block[:3] != 'GAT' \
                    else {key: F.elu(x) for key, x in x_dict.items()}
                x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) \
                     for key, x in x_dict.items()}

        return x_dict['rooms']

    def loss(self, pred, label, mask=None):
        if mask is None:
            return F.cross_entropy(pred, label)
        else:
            return F.cross_entropy(pred[mask, :], label[mask])
