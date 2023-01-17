import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

import sys
sys.path.append('../')
from datasets.mp3d import MP3D
from neural_tree.h_tree.generate_junction_tree_hierarchies import sample_and_generate_jth


# h-tree data structure
HTREE_NODE_TYPES = ['object', 'room', 'object-room', 'room-room']
HTREE_EDGE_TYPES = [('object', 'o-to-or', 'object-room'),
                    ('object-room', 'or-to-o', 'object'),
                    ('room', 'r-to-or', 'object-room'),
                    ('object-room', 'or-to-r', 'room'),
                    ('room', 'r-to-rr', 'room-room'),
                    ('room-room', 'rr-to-r', 'room'),
                    ('object-room', 'or-ro-rr', 'room-room'),
                    ('room-room', 'rr-ro-or', 'object-room'),
                    ('object-room', 'or-to-or', 'object-room'),
                    ('room-room', 'rr-to-rr', 'room-room')]

# auxilary virtual nodes and edges for pre and post learning feature extraction
HTREE_VIRTUAL_NODE_TYPES = ['object_virtual', 'room_virtual']
HTREE_INIT_EDGE_TYPES = [('object_virtual', 'ov_to_or', 'object-room'),
                         ('room_virtual', 'rv_to_or', 'object-room'),
                         ('room_virtual', 'rv_to_rr', 'room-room')]
HTREE_POOL_EDGE_TYPES = [('object', 'o_to_ov', 'object_virtual'), 
                         ('room', 'r_to_rv', 'room_virtual')]


def nx_htree_to_torch(jth_nx):
    """
    takes an nx_dsg, and outputs
    """

    for _, data_dict in jth_nx.nodes.items():
        if 'label' not in data_dict.keys():
            data_dict['label'] = -1     # adding fake label to all clique nodes

        if 'pos' not in data_dict.keys():
            data_dict['pos'] = [0.0, 0.0, 0.0]    # adding fake pos to all clique nodes

    # nodes
    x_ = []
    pos_ = []
    label_ = []
    node_type_ = []

    for _, data_dict in jth_nx.nodes.items():
        x_.append(torch.tensor(data_dict['x'], dtype=torch.float64))
        pos_.append(torch.tensor(data_dict['pos'], dtype=torch.float64))
        label_.append(data_dict['label'])

        if data_dict['node_type'] == 'object':
            node_type_idx = 0
        elif data_dict['node_type'] == 'room':
            node_type_idx = 1
        elif data_dict['node_type'] == 'object-room':
            node_type_idx = 2
        elif data_dict['node_type'] == 'room-room':
            node_type_idx = 3
        else:
            raise ValueError(data_dict['node_type'])

        node_type_.append(node_type_idx)

    x_ = torch.vstack(x_)
    pos_ = torch.vstack(pos_)
    label_ = torch.tensor(label_, dtype=torch.int64)
    node_type_ = torch.tensor(node_type_)

    # edges
    edges = list(jth_nx.edges)
    edge_index = torch.tensor(edges, dtype=torch.int64).t().contiguous()

    edge_type_list = []
    for (i, j) in edges:

        ci = jth_nx.nodes[i]['node_type']
        cj = jth_nx.nodes[j]['node_type']

        if (ci == 'object' and cj == 'object-room') or (ci == 'object-room' and cj == 'object'):
            edge_type_idx = 0
        elif (ci == 'room' and cj == 'object-room') or (ci == 'object-room' and cj == 'room'):
            edge_type_idx = 1
        elif (ci == 'room' and cj == 'room-room') or (ci == 'room-room' and cj == 'room'):
            edge_type_idx = 2
        elif (ci == 'object-room' and cj == 'room-room') or (ci == 'room-room' and cj == 'object-room'):
            edge_type_idx = 3
        elif (ci == 'object-room' and cj == 'object-room'):
            edge_type_idx = 4
        elif (ci == 'room-room' and cj == 'room-room'):
            edge_type_idx = 5
        else:
            raise ValueError(i, j)
        edge_type_list.append(edge_type_idx)

    edge_type_ = torch.tensor(edge_type_list)

    # generate homogeneous graph
    jth_homogeneous = Data(x=x_, edge_index=edge_index, y=label_, pos=pos_, node_type=node_type_, edge_type=edge_type_)

    # extracting the heterogeneous graph
    jth_torch = jth_homogeneous.to_heterogeneous(
        node_type_names=HTREE_NODE_TYPES, edge_type_names=HTREE_EDGE_TYPES)

    return jth_torch


def draw_dsg_jth(jth_nx, attribute_type='clique_has'):
    label_dict = {idx: data_dict[attribute_type] for idx, data_dict in jth_nx.nodes.items()}
    nx.draw(jth_nx, labels=label_dict, with_labels=True)


def torch_dsg_to_nx(dsg_torch):
    """
    converts heterogeneous dsg_nx to a networkx graph, with object and room nodes labeled with node_type

    """

    # converting to networkx
    _g = dsg_torch.to_homogeneous()
    node_type = dsg_torch.node_types

    dsg_nx = to_networkx(_g, node_attrs=['x', 'pos', 'label', 'node_type']).to_undirected()

    # changing node_type from 0/1 to object/room
    for (_, data) in dsg_nx.nodes.items():
        # -1: for removing 's' from 'objects' and 'rooms'
        data['node_type'] = node_type[data['node_type']][:-1]

    return dsg_nx


def get_room_graph(dsg_nx):
    """
    This function extracts room graph from a networkx dsg.
    """
    room_idx_list = [idx for idx in dsg_nx.nodes \
        if dsg_nx.nodes[idx]['node_type'] == 'room']

    return dsg_nx.subgraph(room_idx_list), room_idx_list



def get_object_graph(dsg_nx, room_idx):
    """
    This function extracts object graph from a networkx dsg for a given room index.
    """
    object_idx_list = [idx for idx in dsg_nx.neighbors(room_idx) \
        if dsg_nx.nodes[idx]['node_type'] == 'object']

    return dsg_nx.subgraph(object_idx_list), object_idx_list


def generate_component_jth(dsg_nx_component, component_type, num_zero_padding, room_node_data=None, verbose=False):
    """
    This function generates jth and root_nodes, given either room graph or objects graph in dsg_nx format

    """
    # parameters
    treewidth_bound = 10
    zero_feature = [0.0] * num_zero_padding
    pos_zeros = [0.0] * 3

    assert component_type in ["rooms", "objects"]

    if component_type == "rooms":
        # extracting H-tree of the room graph
        _, jth, root_nodes = sample_and_generate_jth(dsg_nx_component,
                                                       k=treewidth_bound,
                                                       zero_feature=zero_feature,
                                                       copy_node_attributes=['x', 'pos', 'label', 'node_type'],
                                                       need_root_tree=True,
                                                       remove_edges_every_layer=True,
                                                       verbose=verbose)

        # if room graph has only one node, "sample_and_generate_jth" returns nothing for room_root_nodes
        if len(dsg_nx_component) == 1:
            root_nodes = [0]

        # filling node_type with 'room-room' for clique nodes
        for (_, data_dict) in jth.nodes.items():
            if data_dict['type'] == 'clique':
                data_dict['node_type'] = 'room-room'

        return jth, root_nodes

    else:   # component_type == "objects"

        if room_node_data is None:
            raise ValueError("room_node_data not specified.")
        else:
            r, room_data_dict = room_node_data

        # extracting H-tree of the single room room-object graph
        _, jth, root_nodes = sample_and_generate_jth(dsg_nx_component,
                                                       k=treewidth_bound,
                                                       zero_feature=zero_feature,
                                                       copy_node_attributes=['x', 'pos', 'label', 'node_type'],
                                                       need_root_tree=True,
                                                       remove_edges_every_layer=True,
                                                       verbose=verbose)

        # if object graph has only one node, "sample_and_generate_jth" returns nothing for room_root_nodes
        if len(dsg_nx_component) == 1:
            jth.add_node(1, x=zero_feature, pos=pos_zeros, type='clique', clique_has=[jth.nodes[0]['clique_has']])
            jth.add_edge(0, 1)
            root_nodes = [1]

        # the clique nodes should all contain the room node, fill node_type with "object-room" and add room node r
        for (_, data) in jth.nodes.items():
            if data['type'] == 'clique':
                data['node_type'] = 'object-room'  # object cliques, identified as node_type=object-room
                data['clique_has'].append(r)       # add room node to clique

        # adding room node r to all lowest level cliques
        # finding the lowest level cliques
        leaf_nodes = [idx for idx, data_dict in jth.nodes.items() if data_dict['type'] == 'node']
        lowest_level_clique_nodes = set(sum([[n for n in jth.neighbors(leaf_idx)] for leaf_idx in leaf_nodes], []))

        # adding a new compy of the room node, and an edge
        idx_count = jth.number_of_nodes()
        for clique_idx in lowest_level_clique_nodes:
            # add node indexed _idx_count:
            jth.add_node(idx_count,
                          x=room_data_dict['x'],
                          pos=room_data_dict['pos'],
                          label=room_data_dict['label'],
                          node_type=room_data_dict['node_type'],
                          type='node',
                          clique_has=[r])

            jth.add_edge(clique_idx, idx_count)

            # increase _idx_count by 1
            idx_count += 1

        return jth, root_nodes


class HTree:
    def __init__(self, room_jth, room_root_nodes):

        self.jth = room_jth
        self.root_nodes = room_root_nodes
        self.num_nodes = self.jth.number_of_nodes()

    def _reindex_and_compose(self, g, _root_nodes):

        # relabeling object_jth
        new_labels = dict()
        _idx = self.num_nodes
        for i in range(g.number_of_nodes()):
            new_labels[i] = _idx
            _idx += 1

        g = nx.relabel_nodes(g, new_labels)
        _roots = [new_labels[i] for i in _root_nodes]

        # adding object_jth to jth
        self.jth = nx.compose(self.jth, g)
        self.num_nodes = self.jth.number_of_nodes()

        return g, _roots

    def add_object_jth(self, object_jth, object_root_nodes, room_idx):

        # disconnecting all edges between object_root_nodes in object_jth
        H = object_jth.subgraph(object_root_nodes)
        for e_, data in H.edges.items():
            object_jth.remove_edge(e_[0], e_[1])

        # for root node that contains room_idx
        for _root_node in self.root_nodes:

            if not isinstance(self.jth.nodes[_root_node]['clique_has'], list):
                self.jth.nodes[_root_node]['clique_has'] = [self.jth.nodes[_root_node]['clique_has']]

            if room_idx in self.jth.nodes[_root_node]['clique_has']:

                g, _roots = self._reindex_and_compose(object_jth.copy(), object_root_nodes)

                for _object_root_idx in _roots:
                    self.jth.add_edge(_root_node, _object_root_idx)


def generate_htree(dsg_torch, verbose=False):
    """
    takes in heterogeneous dsg_torch and generates h-tree using the sequential procedure.

    """

    # converting to networkx
    dsg_nx = torch_dsg_to_nx(dsg_torch)

    # iterating over each connected component in dsg_nx
    htree_list = []
    for i, c in enumerate(nx.connected_components(dsg_nx)):

        # extracting connected component, from node index list c
        dsg_component = dsg_nx.subgraph(c).copy()

        # extracting room graph
        dsg_component_room, room_idx = get_room_graph(dsg_component)

        # extracting H-tree of the room graph
        dsg_component_room_jth, _room_root_nodes = \
            generate_component_jth(dsg_nx_component=dsg_component_room,
                                                                          component_type="rooms",
                                                                          num_zero_padding=dsg_torch['rooms'].num_features,
                                                                          verbose=verbose)

        # create a Htree
        _htree = HTree(room_jth=dsg_component_room_jth, room_root_nodes=_room_root_nodes)

        # extracting jth of the object graphs in a room
        for r in room_idx:

            # extracting all objects in room r
            dsg_component_object_r, object_idx = get_object_graph(dsg_component, r)

            for oc in nx.connected_components(dsg_component_object_r):

                # extracting object sub-graph
                object_graph_component = dsg_nx.subgraph(oc).copy()

                # extracting H-tree of the object graph component
                object_component_jth, _object_root_nodes = \
                      generate_component_jth(dsg_nx_component=object_graph_component,
                                             component_type="objects",
                                             room_node_data=(r, dsg_component.nodes[r]),
                                             num_zero_padding=dsg_torch['objects'].num_features,
                                             verbose=verbose)

                # adding to _htree
                _htree.add_object_jth(object_jth=object_component_jth,
                                      object_root_nodes=_object_root_nodes,
                                      room_idx=r)

        if verbose:
            print(f"Component {i}: H-tree contains {_htree.jth.number_of_nodes()} nodes "
                    f"and {_htree.jth.number_of_edges()} edges.")
        htree_list.append(_htree)

    return htree_list


if __name__ == "__main__":

    dset = MP3D(complete=True)

    for idx, data in enumerate(dset):

        dsg_torch = data['dsg_torch']

        if dsg_torch.num_edges < 1:

            print("---" * 40)
            print(f"DSG contains only {dsg_torch.num_nodes} node.")
            print(f"We skip H-tree construction for MP3D scene: {idx}")
            print("---" * 40)

        else:

            print("---" * 40)
            print(f"H-tree constructed for MP3D Scene: {idx}.")
            print(f"DSG contains {dsg_torch.num_nodes} nodes.")

            # compute h-trees
            dsg_jth_list = generate_htree(dsg_torch, verbose=False)

            print(f"DSG is divided into {len(dsg_jth_list)} disconnected components.")
            for c in range(len(dsg_jth_list)):
                print(f"Component {c}: H-tree contains {dsg_jth_list[c].jth.number_of_nodes()} nodes "
                      f"and {dsg_jth_list[c].jth.number_of_edges()} edges.")

                if dsg_jth_list[c].jth.number_of_edges() < 1:
                    print("We skip H-tree conversion to torch_geometric. H-tree has no edges.")
                else:
                    # convert to torch_geometric.data.HeteroData
                    dsg_jth_torch = nx_dsg_jth_to_torch(dsg_jth_list[c].jth)
                    print(f"Component {c}: H-tree converted to torch_geometric.data.HeteroData")

            print("---" * 40)


