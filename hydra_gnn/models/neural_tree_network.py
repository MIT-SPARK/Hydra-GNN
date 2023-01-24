from hydra_gnn.models.utils import build_hetero_conv, build_GAT_hetero_conv
from neural_tree.construct import HTREE_NODE_TYPES, HTREE_VIRTUAL_NODE_TYPES, HTREE_EDGE_TYPES, HTREE_INIT_EDGE_TYPES, HTREE_POOL_EDGE_TYPES
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, HeteroConv


class LeafPool(MessagePassing):
    """
    Helper class to pool final hidden states from the leaf nodes for classification.
    """
    def __init__(self, aggr= "mean"):
        super().__init__(aggr=aggr)

    def forward(self, x, edge_index):
        out = self.propagate(edge_index, x=x)
        return out

    def message(self, x_j):
        return x_j


class NeuralTreeNetwork(nn.Module):
    """
    This NeuralTreeNetwork class implements message passing on augmented htrees, which includes the htree and virtual 
    nodes for initialization and most message-passing pooling.
    """
    def __init__(self, input_dim, output_dim, conv_block='GraphSAGE', hidden_dim=None, num_layers=None,
                 GAT_hidden_dims=None, GAT_heads=None, GAT_concats=None, dropout=0.25, **kwargs):
        """
        :param input_dim: int, input feature dimension
        :param output_dim: int or a tuple of int, output dimension(s), i.e. the number of labels
        :param task: string, only 'node' classification is implemented now
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
        super(NeuralTreeNetwork, self).__init__()
        assert conv_block in ['GraphSAGE', 'GAT']
        self.conv_block = conv_block
        self.num_layers = num_layers if conv_block != 'GAT' else len(GAT_heads)
        self.dropout = dropout

        # initialize clique features - siyi: todo

        # message passing
        input_dim_dict = {node_type: input_dim for node_type in HTREE_NODE_TYPES}
        hidden_dim_dict = {node_type: hidden_dim for node_type in HTREE_NODE_TYPES}
        output_dim_dict = {node_type: output_dim for node_type in HTREE_NODE_TYPES}
        if self.conv_block != 'GAT':  # GAT dimensions are different than others
            self.convs = nn.ModuleList()
            self.convs.append(build_hetero_conv(conv_block, HTREE_EDGE_TYPES, input_dim_dict, hidden_dim_dict))
            for _ in range(1, self.num_layers - 1):
                self.convs.append(build_hetero_conv(conv_block, HTREE_EDGE_TYPES, hidden_dim_dict, hidden_dim_dict))
            self.convs.append(build_hetero_conv(conv_block, HTREE_EDGE_TYPES, hidden_dim_dict, output_dim_dict))
        else:
            self.convs = build_GAT_hetero_conv(HTREE_EDGE_TYPES, input_dim_dict, output_dim_dict, 
                GAT_hidden_dims, GAT_heads, GAT_concats, dropout)

        # post message passing pooling
        # conv_dict = dict()
        # for source, edge_name, target in HTREE_POOL_EDGE_TYPES:
        #     conv_dict[source, edge_name, target] = LeafPool(aggr='mean')
        # self.post_mp = HeteroConv(conv_dict)
        self.post_mp = LeafPool(aggr='mean')

    def forward(self, data):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict

        # x = F.dropout(x, p=self.dropout, training=self.training)
        for i in range(self.num_layers):
            x_dict = self.convs[i](x_dict, edge_index_dict)
            if i != self.num_layers - 1:    # activation and dropout, except for the last iteration
                x_dict = {key: F.relu(x) for key, x in x_dict.items()} if self.conv_block != 'GAT' \
                    else {key: F.elu(x) for key, x in x_dict.items()}
                x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) \
                     for key, x in x_dict.items()}
        
        x_room = self.post_mp(x_dict['room'], edge_index_dict['room', 'r_to_rv', 'room_virtual'])\
            [0: data['room_virtual'].num_nodes]
        return x_room

    def loss(self, pred, label, mask=None):
        if mask is None:
            return F.cross_entropy(pred, label)
        else:
            return F.cross_entropy(pred[mask, :], label[mask])
