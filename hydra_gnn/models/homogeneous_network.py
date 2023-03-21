from hydra_gnn.models.utils import build_conv_layer, build_GAT_conv_layers, cross_entropy_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.norm.batch_norm import BatchNorm


class HomogeneousNetwork(nn.Module):
    """
    This HomogeneousNetwork class implements message passing on homogeneous room-object graphs.
    """
    def __init__(self, input_dim, output_dim=None, output_dim_dict=None, conv_block='GCN', 
                 hidden_dim=None, num_layers=None, GAT_hidden_dims=None, GAT_heads=None, GAT_concats=None, dropout=0.25, **kwargs):
        """
        :param input_dim: int, input feature dimension
        :param output_dim: int, output dimension, i.e. the number of room labels
        :param output_dim_dict: dict, output dimensions, i.e. the number of room labels and object labels
        :param conv_block: str, message passing convolution block
        :param hidden_dim: int, the output dimension of the graph convolution (ignored if conv_block='GAT')
        :param num_layers: int, the number of graph convolution iterations (ignored if conv_block='GAT')
        :param GAT_hidden_dims: list of int, the output dimensions of GAT convolution (ignored if conv_block!='GAT')
        :param GAT_heads: list of int, the number of attention heads of GAT convolution (ignored if conv_block!='GAT')
        :param GAT_concats GAT_heads: list of bool, concatenate output of multi-head GAT convolution (ignored if conv_block!='GAT')
        :param dropout: float, dropout ratio during training
        """
        super(HomogeneousNetwork, self).__init__()
        self.conv_block = conv_block
        if output_dim is not None:
            assert output_dim_dict is None
            self.classification_task = 'room'
            if self.conv_block[:3] != 'GAT':
                mp_output_dim = output_dim
            else:
                GAT_hidden_dims_with_last_layer = GAT_hidden_dims + [output_dim]
        else:
            assert output_dim_dict is not None
            self.classification_task = 'all'
            if self.conv_block[:3] != 'GAT':
                mp_output_dim = hidden_dim
            else:
                GAT_hidden_dims_with_last_layer = GAT_hidden_dims
        self.num_layers = num_layers if conv_block[:3] != 'GAT' else len(GAT_heads)
        self.dropout = dropout

        # message passing
        self.convs = nn.ModuleList()
        if self.conv_block == 'GAT':
            self.convs = build_GAT_conv_layers(input_dim, GAT_hidden_dims_with_last_layer, GAT_heads,
                GAT_concats, dropout=dropout)
        elif self.conv_block == 'GAT_edge':
            self.convs = build_GAT_conv_layers(input_dim, GAT_hidden_dims_with_last_layer, GAT_heads,
                GAT_concats, dropout=dropout, edge_dim=3, add_self_loop=True, fill_value=torch.zeros(3, dtype=torch.float64))
        else:  # GAT dimensions are different than others
            self.convs.append(build_conv_layer(self.conv_block, input_dim, hidden_dim))
            for _ in range(1, self.num_layers - 1):
                self.convs.append(build_conv_layer(self.conv_block, hidden_dim, hidden_dim))
            self.convs.append(build_conv_layer(self.conv_block, hidden_dim, mp_output_dim))

        # batch normalization
        if self.conv_block == 'GIN':
            self.batch_norms = nn.ModuleList()
            for _ in range(self.num_layers):
                self.batch_norms.append(BatchNorm(hidden_dim))
        
        if self.classification_task == 'all':
            if self.conv_block[:3] != 'GAT':
                final_hidden_dim = hidden_dim
            else:
                final_hidden_dim = GAT_hidden_dims_with_last_layer[-1] * GAT_heads[-1] if GAT_concats[-1] \
                    else GAT_hidden_dims_with_last_layer[-1]
            # remove 's' for HomogeneousNeuralTreeNetwork child class
            num_room_labels = output_dim_dict['rooms'] if 'rooms' in output_dim_dict else output_dim_dict['room']
            num_object_labels = output_dim_dict['objects'] if 'objects' in output_dim_dict else output_dim_dict['object']
            self.post_mp_room = nn.Linear(final_hidden_dim, num_room_labels)
            self.post_mp_object = nn.Linear(final_hidden_dim, num_object_labels)

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

        if self.classification_task == 'room':
            return x[room_mask, :]
        else:
            x = F.relu(x) if self.conv_block[:3] != 'GAT' else F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            return self.post_mp_room(x[room_mask, :]), self.post_mp_object(x[~room_mask, :])

    def loss(self, pred, label, mask=None):
        return cross_entropy_loss(pred, label, mask)
