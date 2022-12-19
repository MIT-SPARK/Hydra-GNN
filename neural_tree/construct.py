import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, from_networkx

import sys
sys.path.append('../')
from datasets.mp3d import MP3D, extract_object_graph, extract_room_graph
from neural_tree.h_tree.generate_junction_tree_hierarchies import sample_and_generate_jth, generate_node_labels


def nx_dsg_jth_to_torch(dsg_jth_nx):
    """
    takes an nx_dsg, and outputs
    """

    for idx, data_ in dsg_jth_nx.nodes.items():
        if 'label' not in set(data_.keys()):
            data_['label'] = -10    # adding fake label to all clique nodes

        if 'pos' not in set(data_.keys()):
            data_['pos'] = [0.0] * 3    # adding fake pos to all clique nodes

    # nodes
    x_ = []
    pos_ = []
    label_ = []
    node_type_ = []

    for idx, data in dsg_jth_nx.nodes.items():
        x_.append(torch.tensor(data['x']))
        pos_.append(torch.tensor(data['pos']))
        label_.append(data['label'])

        if data['node_type'] == 'object':
            _type = 0
        elif data['node_type'] == 'room':
            _type = 1
        elif data['node_type'] == 'object-room':
            _type = 2
        elif data['node_type'] == 'room-room':
            _type = 3
        else:
            raise ValueError

        node_type_.append(_type)

    x_ = torch.vstack(x_)
    pos_ = torch.vstack(pos_)
    label_ = torch.tensor(label_)
    node_type_ = torch.tensor(node_type_)

    # edges
    edges = list(dsg_jth_nx.edges)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    edge_type_ = []
    for (i, j) in edges:

        ci = dsg_jth_nx.nodes[i]['node_type']
        cj = dsg_jth_nx.nodes[j]['node_type']

        if (ci == 'object' and cj == 'object-room') or (ci == 'object-room' and cj == 'object'):
            _type = 0
        elif (ci == 'room' and cj == 'object-room') or (ci == 'object-room' and cj == 'room'):
            _type = 1
        elif (ci == 'room' and cj == 'room-room') or (ci == 'room-room' and cj == 'room'):
            _type = 2
        elif (ci == 'object-room' and cj == 'room-room') or (ci == 'room-room' and cj == 'object-room'):
            _type = 3
        elif (ci == 'object-room' and cj == 'object-room'):
            _type = 4
        elif (ci == 'room-room' and cj == 'room-room'):
            _type = 5
        else:
            print(i, j)
            breakpoint()
            raise ValueError

        edge_type_.append(_type)

    edge_type_ = torch.tensor(edge_type_)

    # labeling
    names_n = ['object', 'room', 'object-room', 'room-room']

    names_e = [('object', 'o-to-or', 'object-room'),
               ('room', 'r-to-or', 'object-room'),
               ('room', 'r-to-rr', 'room-room'),
               ('object-room', 'or-ro-rr', 'room-room'),
               ('object-room', 'or-to-or', 'object-room'),
               ('room-room', 'rr-to-rr', 'room-room')]

    # generate homogeneous graph
    _g = Data(x=x_, edge_index=edge_index, y=label_, pos=pos_, node_type=node_type_, edge_type=edge_type_)

    # extracting the heterogeneous graph
    jth_torch = _g.to_heterogeneous(node_type_names=names_n,
                                    edge_type_names=names_e)

    return jth_torch


def draw_dsg_jth(nx_dsg_jth):

    l_ = dict()
    for idx, data in nx_dsg_jth.nodes.items():
        l_[idx] = data["clique_has"]

    nx.draw(nx_dsg_jth, labels=l_, with_labels=True)

    return None


def dsg_torch_to_nx(dsg_torch):
    """
    converts heterogeneous dsg_nx to a networkx graph, with object and room nodes labeled with node_type

    """

    # converting to networkx
    _g = dsg_torch.to_homogeneous()
    node_type = dsg_torch.node_types

    dsg_nx = to_networkx(_g, node_attrs=['x', 'pos', 'label', 'node_type']).to_undirected()

    # changing node_type from 0/1 to object/room
    for (idx, data) in dsg_nx.nodes.items():
        d_ = node_type[data['node_type']][:-1]      # -1: for removing 's' from 'objects' and 'rooms'
        data['node_type'] = d_
        # if data['node_type'] == 1:
        #     data['node_type'] = 'room'
        # elif data['node_type'] == 0:
        #     data['node_type'] = 'object'
        # else:
        #     print("Error: the node is not room/object")

    return dsg_nx


def get_room_graph(dsg_nx):
    """
    extracts room graph from dsg_nx

    """

    _room_idx = []
    for (idx, data) in dsg_nx.nodes.items():
        if data['node_type'] == 'room':
            _room_idx.append(idx)

    return dsg_nx.subgraph(_room_idx), _room_idx


def generate_component_jth(dsg_nx_component, component_type, room_node_data=None, verbose=False):
    """
    generates jth and root_nodes, given either room graph or objects graph in dsg_nx format

    """
    # parameters
    treewidth_bound = 10
    zero_feature = [0.0] * 6
    pos_zeros = [0.0] * 3

    assert component_type in ["rooms", "objects"]

    if component_type == "rooms":
        # extracting H-tree of the room graph
        _, _jth, _root_nodes = sample_and_generate_jth(dsg_nx_component,
                                                       k=treewidth_bound,
                                                       zero_feature=zero_feature,
                                                       copy_node_attributes=['x', 'pos',
                                                                             'label',
                                                                             'node_type'],
                                                       need_root_tree=True,
                                                       remove_edges_every_layer=True,
                                                       verbose=verbose)

        # if room graph has only one node, "sample_and_generate_jth" returns nothing for room_root_nodes
        if len(dsg_nx_component) == 1:
            _root_nodes = [0]

        # filling node_type for clique nodes in the G_room_jth
        for (idx, data) in _jth.nodes.items():
            if data['type'] == 'clique':
                data['node_type'] = 'room-room'  # room cliques, identified as node_type=room-room

        return _jth, _root_nodes

    elif component_type == "objects":

        #
        if room_node_data is None:
            raise ValueError("room_node_data not specified.")
        else:
            r, _room_data = room_node_data

        #
        _, _jth, _root_nodes = sample_and_generate_jth(dsg_nx_component,
                                                       k=treewidth_bound,
                                                       zero_feature=zero_feature,
                                                       copy_node_attributes=['x',
                                                                             'pos',
                                                                             'label',
                                                                             'node_type'],
                                                       need_root_tree=True,
                                                       remove_edges_every_layer=True,
                                                       verbose=verbose)

        # if object graph has only one node, "sample_and_generate_jth" returns nothing for room_root_nodes
        if len(dsg_nx_component) == 1:
            _jth.add_node(1, x=zero_feature, pos=pos_zeros, type='clique', clique_has=[_jth.nodes[0]['clique_has']])
            _jth.add_edge(0, 1)
            _root_nodes = [1]

        #
        _num_nodes = _jth.number_of_nodes()
        _idx_count = _num_nodes

        # fill node type: i.e. set all type==clique with node_type = "object-room"
        for (idx, data) in _jth.nodes.items():
            if data['type'] == 'clique':
                data['node_type'] = 'object-room'  # object cliques, identified as node_type=object-room

        # add room node r to all cliques, and also add it as a leaf node to all cliques
        for (idx, data) in _jth.nodes.items():
            if data['type'] == 'clique':
                data['clique_has'].append(r)  # adding room node to clique

        # adding room node r to all lowest level cliques
        # finding the lowest level cliques
        leaf_nodes = []
        for (idx, data) in _jth.nodes.items():
            if data['type'] == 'node':
                leaf_nodes.append(idx)
        n_ = set()
        for l_ in leaf_nodes:
            for idx in _jth.neighbors(l_):
                n_.add(idx)
        n_ = list(n_)

        # adding a new compy of the room node, and an edge
        for nl_ in n_:
            # add node indexed _idx_count:
            _jth.add_node(_idx_count,
                          x=_room_data['x'],
                          pos=_room_data['pos'],
                          label=_room_data['label'],
                          node_type=_room_data['node_type'],
                          type='node',
                          clique_has=[r])

            # add edge (nl_, _idx_count):
            _jth.add_edge(nl_, _idx_count)

            # increase _idx_count by 1
            _idx_count += 1

        return _jth, _root_nodes

    else:
        raise ValueError("Incorrect component_type.")


def get_objects_in_room(dsg_nx, room_idx):

    object_idx = []
    for idx in dsg_nx.neighbors(room_idx):
        if dsg_nx.nodes[idx]['node_type'] == 'object':
            object_idx.append(idx)

    _objects_nx = dsg_nx.subgraph(object_idx)

    return _objects_nx, object_idx


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
        # print("_root_nodes: ", _root_nodes)
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
                    # print(f"Adding edge: ({_root_node}, {_object_root_idx})")
                    self.jth.add_edge(_root_node, _object_root_idx)


def generate_htree(dsg_torch, verbose=False):
    """
    takes in heterogeneous dsg_torch and generates h-tree using the sequential procedure.

    """

    # breakpoint()
    # converting to networkx
    dsg_nx = dsg_torch_to_nx(dsg_torch)
    htree_list = []

    # iterating over each connected component in dsg_nx
    for c in nx.connected_components(dsg_nx):

        # extracting connected component, from node index list c
        dsg_component = dsg_nx.subgraph(c).copy()

        # extracting room graph
        dsg_component_room, room_idx = get_room_graph(dsg_component)

        # extracting H-tree of the room graph
        dsg_component_room_jth, _room_root_nodes = generate_component_jth(dsg_nx_component=dsg_component_room,
                                                                          component_type="rooms",
                                                                          verbose=verbose)

        # create a Htree
        _htree = HTree(room_jth=dsg_component_room_jth, room_root_nodes=_room_root_nodes)

        # extracting jth of the object graphs in a room
        for r in room_idx:

            # extracting all objects in room r
            _objects_nx, object_idx = get_objects_in_room(dsg_component, r)

            for oc in nx.connected_components(_objects_nx):

                # extracting object sub-graph
                object_graph_component = dsg_nx.subgraph(oc).copy()

                # extracting H-tree of the object graph component
                object_component_jth, _object_root_nodes = \
                    generate_component_jth(dsg_nx_component=object_graph_component,
                                           component_type="objects",
                                           room_node_data=(r, dsg_component.nodes[r]),
                                           verbose=verbose)

                # adding to _htree
                _htree.add_object_jth(object_jth=object_component_jth,
                                      object_root_nodes=_object_root_nodes,
                                      room_idx=r)

        htree_list.append(_htree)

    return htree_list


if __name__ == "__main__":

    dset = MP3D(complete=False)

    # data = dset[2]
    # dsg_torch = data['dsg_torch']
    #
    # dsg_jth_list = generate_htree(dsg_torch, verbose=False)
    # dsg_jth_nx = dsg_jth_list[0].jth
    # dsg_jth_torch = nx_dsg_jth_to_torch(dsg_jth_nx)

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


