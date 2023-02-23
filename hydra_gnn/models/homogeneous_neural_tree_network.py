from hydra_gnn.models.homogeneous_network import HomogeneousNetwork
from hydra_gnn.models.neural_tree_network import LeafPool
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn


class HomogeneousNeuralTreeNetwork(HomogeneousNetwork):
    """
    This HomogeneousNeuralTreeNetwork class implements pre-message passing initialization (of clique nodes) using virtual nodes,
    homogeneous message passing on htrees, and post message passing pooling from leaf nodes to virtual nodes.
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
        super(HomogeneousNeuralTreeNetwork, self).__init__(input_dim, output_dim, conv_block, hidden_dim, num_layers, 
                                                           GAT_hidden_dims, GAT_heads, GAT_concats, dropout, **kwargs)
        assert conv_block in ['GraphSAGE', 'GAT', 'GAT_edge']

        # initialize clique features -- keep dimension the same
        self.pre_mp = pyg_nn.GATConv(input_dim, input_dim, heads=1, concat=False, dropout=0.0, add_self_loops=False)

        # post message passing pooling
        self.post_mp = LeafPool(aggr='mean')

    def forward(self, data):
        x, edge_index, init_edge_index, pool_edge_index, room_mask = \
            data.x, data.edge_index, data.init_edge_index, data.pool_edge_index, data.room_mask

        # initialize clique nodes
        x = self.pre_mp(x, edge_index=init_edge_index)

        # x = F.dropout(x, p=self.dropout, training=self.training)
        for i in range(self.num_layers):
            if self.conv_block == 'GAT_edge':
                x = self.convs[i](x, edge_index=edge_index, edge_attr=data.edge_attr)
            else:
                x = self.convs[i](x, edge_index=edge_index)
            if i != self.num_layers - 1:    # activation and dropout, except for the last iteration
                x = F.relu(x) if self.conv_block[:3] != 'GAT' else F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # pool room nodes' final hidden state to room_virtual nodes
        x_room = self.post_mp(x, pool_edge_index)[room_mask, :]

        return x_room
