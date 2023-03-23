from hydra_gnn.models.utils import build_hetero_conv, build_GAT_hetero_conv, cross_entropy_loss
from neural_tree.construct import HTREE_NODE_TYPES, HTREE_EDGE_TYPES, HTREE_INIT_EDGE_TYPES
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import MessagePassing, HeteroConv


class LeafPool(MessagePassing):
    """
    Helper class to pool final hidden states from the leaf nodes for classification.
    """
    def __init__(self, aggr='mean'):
        super().__init__(aggr=aggr)

    def forward(self, x, edge_index):
        out = self.propagate(edge_index, x=x)
        return out

    def message(self, x_j):
        return x_j


class HeterogeneousNeuralTreeNetwork(nn.Module):
    """
    This HeterogeneousNeuralTreeNetwork class implements message passing on augmented htrees, which includes the htree and virtual 
    nodes for initialization and most message-passing pooling.
    """
    def __init__(self, input_dim_dict, output_dim=None, output_dim_dict=None, conv_block='GraphSAGE',
                 hidden_dim=None, num_layers=None, GAT_hidden_dims=None, GAT_heads=None, GAT_concats=None, dropout=0.25, **kwargs):
        """
        :param input_dim_dict: dictionary of node type to input feature dimension mapping
        :param output_dim: int or a tuple of int, output dimension(s), i.e. the number of labels
        :param output_dim_dict: dict, output dimensions, i.e. the number of room labels and object labels
        :param conv_block: str, message passing convolution block
        :param hidden_dim: int, the output dimension of the graph convolution operations (ignored if conv_block='GAT')
        :param num_layers: int, the number of graph convolution iterations (ignored if conv_block='GAT')
        :param GAT_hidden_dims: list of int, the output dimensions of GAT convolution operations (ignored if
        conv_block!='GAT')
        :param GAT_heads: list of int, the number of attention heads of GAT convolution operation (ignored if
        conv_block!='GAT')
        :param GAT_concats GAT_heads: list of bool, concatenate output of multi-head GAT convolution operation (ignored
        if conv_block!='GAT')
        :param dropout: float, dropout ratio during training
        """
        super(HeterogeneousNeuralTreeNetwork, self).__init__()
        assert conv_block in ['GraphSAGE', 'GAT', 'GAT_edge']
        self.conv_block = conv_block
        if output_dim is not None:
            assert output_dim_dict is None
            self.classification_task = 'room'
            output_dim_dict = {node_type: output_dim for node_type in HTREE_NODE_TYPES}
        else:
            assert output_dim_dict is not None
            self.classification_task = 'all'
        self.num_layers = num_layers if conv_block[:3] != 'GAT' else len(GAT_heads)
        self.dropout = dropout

        # dimension dictionaries
        assert input_dim_dict['object'] == input_dim_dict['object_virtual']
        assert input_dim_dict['room'] == input_dim_dict['room_virtual']
        assert input_dim_dict['object-room'] == input_dim_dict['room-room']

        # initialize clique features - match output dimension with leaf dimension
        conv_dict = dict()
        for source, edge_name, target in HTREE_INIT_EDGE_TYPES:
            conv_dict[source, edge_name, target] = \
                pyg_nn.GATConv((input_dim_dict[source], input_dim_dict[target]), input_dim_dict[target], 
                               heads=1, concat=False, dropout=0.0, add_self_loops=False)
        self.pre_mp = HeteroConv(conv_dict, aggr='mean')

        # message passing
        mp_input_dim_dict = {node_type: input_dim_dict[node_type] for node_type in HTREE_NODE_TYPES}
        hidden_dim_dict = {node_type: hidden_dim for node_type in HTREE_NODE_TYPES}
        if self.conv_block == 'GAT':
            self.convs = build_GAT_hetero_conv(HTREE_EDGE_TYPES, mp_input_dim_dict, output_dim_dict, 
                GAT_hidden_dims, GAT_heads, GAT_concats, dropout)
        elif self.conv_block == 'GAT_edge':
            self.convs = build_GAT_hetero_conv(HTREE_EDGE_TYPES, mp_input_dim_dict, output_dim_dict, 
                GAT_hidden_dims, GAT_heads, GAT_concats, dropout, edge_dim=3, 
                fill_value=torch.zeros(3, dtype=torch.float64))
        else:
            self.convs = nn.ModuleList()
            self.convs.append(build_hetero_conv(conv_block, HTREE_EDGE_TYPES, mp_input_dim_dict, hidden_dim_dict))
            for _ in range(1, self.num_layers - 1):
                self.convs.append(build_hetero_conv(conv_block, HTREE_EDGE_TYPES, hidden_dim_dict, hidden_dim_dict))
            self.convs.append(build_hetero_conv(conv_block, HTREE_EDGE_TYPES, hidden_dim_dict, output_dim_dict))
        
        # post message passing pooling
        # conv_dict = dict()
        # for source, edge_name, target in HTREE_POOL_EDGE_TYPES:
        #     conv_dict[source, edge_name, target] = LeafPool(aggr='mean')
        # self.post_mp = HeteroConv(conv_dict)
        self.post_mp = LeafPool(aggr='mean')

    def forward(self, data):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict

        # initialize clique nodes
        x_dict.update(self.pre_mp(x_dict, edge_index_dict))

        # x = F.dropout(x, p=self.dropout, training=self.training)
        for i in range(self.num_layers):
            if self.conv_block == 'GAT_edge':
                x_dict = self.convs[i](x_dict, edge_index_dict, data.edge_attr_dict)
            elif self.conv_block == 'PointTransformer':
                pos_attr_dict = {edge_type: (data[edge_type[0]].pos, data[edge_type[-1]].pos) for edge_type in HTREE_EDGE_TYPES}
                x_dict = self.convs[i](x_dict=x_dict, edge_index_dict=edge_index_dict, pos_dict=pos_attr_dict)
            else:
                x_dict = self.convs[i](x_dict, edge_index_dict)
            
            if i != self.num_layers - 1:    # activation and dropout, except for the last iteration
                x_dict = {key: F.relu(x) for key, x in x_dict.items()} if self.conv_block[:3] != 'GAT' \
                    else {key: F.elu(x) for key, x in x_dict.items()}
                x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) \
                     for key, x in x_dict.items()}
        
        # pool room nodes' final hidden state to room_virtual nodes
        if self.classification_task == 'room':
            x_room = self.post_mp(x_dict['room'], edge_index_dict['room', 'r_to_rv', 'room_virtual']) \
                [0: data['room_virtual'].num_nodes]
            return x_room
        else:
            x_dict = {key: F.relu(x) for key, x in x_dict.items()} if self.conv_block[:3] != 'GAT' \
                else {key: F.elu(x) for key, x in x_dict.items()}
            x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) \
                    for key, x in x_dict.items()}
            x_room = self.post_mp(x_dict['room'], edge_index_dict['room', 'r_to_rv', 'room_virtual']) \
                [0: data['room_virtual'].num_nodes]
            x_object = self.post_mp(x_dict['object'], edge_index_dict['object', 'o_to_ov', 'object_virtual'])\
                [0: data['object_virtual'].num_nodes]
            return x_room, x_object

    def loss(self, pred, label, mask=None):
        return cross_entropy_loss(pred, label, mask)
