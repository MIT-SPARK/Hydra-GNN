import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import HeteroConv


def build_conv_layer(conv_block, input_dim, output_dim, **kwargs):
    """
    Build a PyTorch Geometric convolution layer given specified input and output dimension.
    """
    if conv_block == 'GraphSAGE':
        return pyg_nn.SAGEConv(input_dim, output_dim, normalize=False, bias=True)
    elif conv_block == 'GCN':
        return pyg_nn.GCNConv(input_dim, output_dim, add_self_loops=True)
    elif conv_block == 'GIN':
        return pyg_nn.GINConv(nn.Sequential(nn.Linear(input_dim, output_dim), nn.ReLU(),
                                            nn.Linear(output_dim, output_dim)), eps=0., train_eps=True)
    elif conv_block == 'PointTransformer':
        return pyg_nn.PointTransformerConv(input_dim, output_dim, **kwargs)
    else:
        return NotImplemented


def build_GAT_conv_layers(input_dim, hidden_dims, heads, concats, dropout=0., add_self_loop=True, \
    edge_dim=None, fill_value='mean'):
    """
    Build a list of PyTorch Geometric GAT convolution layers given input dimension and dropout ratio. This function also
     requires hidden dimensions, number of attention heads, concatenation flags for all layers.
    """
    assert len(hidden_dims) == len(heads)
    assert len(hidden_dims) == len(concats)
    convs = nn.ModuleList()
    convs.append(pyg_nn.GATConv(input_dim, hidden_dims[0], heads=heads[0], concat=concats[0], 
                                dropout=dropout, add_self_loops=add_self_loop, edge_dim=edge_dim, 
                                fill_value=fill_value))
    for i in range(1, len(hidden_dims)):
        if concats[i - 1]:
            convs.append(pyg_nn.GATConv(hidden_dims[i - 1] * heads[i - 1], hidden_dims[i], heads=heads[i], 
                                        concat=concats[i], dropout=dropout, add_self_loops=add_self_loop,
                                        edge_dim=edge_dim, fill_value=fill_value))
        else:
            convs.append(pyg_nn.GATConv(hidden_dims[i - 1], hidden_dims[i], heads=heads[i], 
                                        concat=concats[i], dropout=dropout, add_self_loops=add_self_loop,
                                        edge_dim=edge_dim, fill_value=fill_value))
    return convs


def build_hetero_conv(conv_block, edge_types, input_dim_dict, output_dim_dict, aggr='sum'):
    conv_dict = dict()
    for source, edge_name, target in edge_types:
        if conv_block == 'PointTransformer':
            conv_dict[source, edge_name, target] = \
                build_conv_layer(conv_block, (input_dim_dict[source], input_dim_dict[target]),
                                 output_dim_dict[target], add_self_loops=(source==target))
        else:
            conv_dict[source, edge_name, target] = \
                build_conv_layer(conv_block, (input_dim_dict[source], input_dim_dict[target]), 
                                 output_dim_dict[target])
    return HeteroConv(conv_dict, aggr=aggr)


def build_GAT_hetero_conv(edge_types, input_dim_dict, output_dim_dict, 
                          GAT_hidden_dims, GAT_heads, GAT_concats, dropout, aggr='sum',
                          edge_dim=None, fill_value='mean'):
    # build GAT conv layers for each edge type
    conv_module_list_dict = dict()
    for source, edge_name, target in edge_types:
        conv_module_list_dict[source, edge_name, target] = \
            build_GAT_conv_layers((input_dim_dict[source], input_dim_dict[target]), 
                                  GAT_hidden_dims + [output_dim_dict[target]], 
                                  GAT_heads, GAT_concats, dropout, add_self_loop=(source==target),
                                  edge_dim=edge_dim, fill_value=fill_value)
    # assemble each layer using HeteroConv
    convs = nn.ModuleList()
    for i in range(len(GAT_heads)):
        convs.append(HeteroConv(
            {edge_type: conv_module_list_dict[edge_type][i] for edge_type in conv_module_list_dict.keys()}, 
            aggr=aggr))
    return convs


def cross_entropy_loss(pred, label, mask=None):
    if isinstance(pred, torch.Tensor):
        if mask is None:
            return F.cross_entropy(pred, label)
        else:
            return F.cross_entropy(pred[mask, :], label[mask])
    else:
        if mask is None:
            loss_sum = sum(F.cross_entropy(p, l, reduction='sum') for p, l in zip(pred, label))
            num_labels = sum(torch.numel(l) for l in label)
        else:
            loss_sum = sum(F.cross_entropy(p[m, :], l[m], reduction='sum') for p, l, m in zip(pred, label, mask))
            num_labels = sum(torch.numel(l[m]) for l, m in zip(label, mask))
        return loss_sum / num_labels
        # if mask is None:
        #     loss_sum = sum(F.cross_entropy(p, l) for p, l in zip(pred, label))
        # else:
        #     loss_sum = sum(F.cross_entropy(p[m, :], l[m]) for p, l, m in zip(pred, label, mask))
        # return loss_sum
