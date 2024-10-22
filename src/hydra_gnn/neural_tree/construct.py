"""Module to construct a H-Tree."""
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from hydra_gnn.neural_tree.generate_junction_tree_hierarchies import (
    sample_and_generate_jth,
)


# h-tree data structure
HTREE_NODE_TYPES = ["object", "room", "object-room", "room-room"]
HTREE_EDGE_TYPES = [
    ("object", "o_to_or", "object-room"),
    ("object-room", "or_to_o", "object"),
    ("room", "r_to_or", "object-room"),
    ("object-room", "or_to_r", "room"),
    ("room", "r_to_rr", "room-room"),
    ("room-room", "rr_to_r", "room"),
    ("object-room", "or_ro_rr", "room-room"),
    ("room-room", "rr_ro_or", "object-room"),
    ("object-room", "or_to_or", "object-room"),
    ("room-room", "rr_to_rr", "room-room"),
]

# auxilary virtual nodes and edges for pre and post learning feature extraction
HTREE_VIRTUAL_NODE_TYPES = ["object_virtual", "room_virtual"]
HTREE_INIT_EDGE_TYPES = [
    ("object_virtual", "ov_to_or", "object-room"),
    ("room_virtual", "rv_to_or", "object-room"),
    ("room_virtual", "rv_to_rr", "room-room"),
]
HTREE_POOL_EDGE_TYPES = [
    ("object", "o_to_ov", "object_virtual"),
    ("room", "r_to_rv", "room_virtual"),
]


def torch_dsg_to_nx(dsg_torch):
    """
    Converts heterogeneous dsg_nx to a networkx graph, with object and room nodes labeled with node_type
    """
    # convert to networkx
    _g = dsg_torch.to_homogeneous()
    node_type = dsg_torch.node_types

    attr_keys = ["x", "pos", "label", "node_type"]
    if "node_ids" in _g:
        attr_keys.append("node_ids")
    dsg_nx = to_networkx(_g, node_attrs=attr_keys).to_undirected()

    # change node_type from 0/1 to object/room
    for _, data in dsg_nx.nodes.items():
        # -1: for removing 's' from 'objects' and 'rooms'
        data["node_type"] = node_type[data["node_type"]][:-1]

    return dsg_nx


def get_room_graph(dsg_nx):
    """
    Extracts room graph from a networkx dsg.
    """
    room_idx_list = [
        idx for idx in dsg_nx.nodes if dsg_nx.nodes[idx]["node_type"] == "room"
    ]

    return dsg_nx.subgraph(room_idx_list), room_idx_list


def get_object_graph(dsg_nx, room_idx):
    """
    Extracts object graph from a networkx dsg for a given room index.
    """
    object_idx_list = [
        idx
        for idx in dsg_nx.neighbors(room_idx)
        if dsg_nx.nodes[idx]["node_type"] == "object"
    ]

    return dsg_nx.subgraph(object_idx_list), object_idx_list


def generate_component_jth(
    dsg_nx_component,
    component_type,
    num_zero_padding,
    room_node_data=None,
    treewidth_bound=1000,
    verbose=False,
):
    """
    Generates jth and root_nodes, given either room graph or objects graph in dsg_nx format
    """
    assert component_type in ["rooms", "objects"]
    zero_feature = [0.0] * num_zero_padding
    zero_pos = [0.0] * 3

    if component_type == "rooms":
        # extracting H-tree of the room graph
        _, jth, root_nodes = sample_and_generate_jth(
            dsg_nx_component,
            k=treewidth_bound,
            zero_feature=zero_feature,
            copy_node_attributes=["x", "pos", "label", "node_type"],
            need_root_tree=True,
            remove_edges_every_layer=True,
            verbose=verbose,
        )

        # if room graph has only one node, "sample_and_generate_jth" returns nothing for room_root_nodes
        if len(dsg_nx_component) == 1:
            root_nodes = [0]

        # filling node_type with 'room-room' for clique nodes
        for _, data_dict in jth.nodes.items():
            if data_dict["type"] == "clique":
                data_dict["node_type"] = "room-room"

        return jth, root_nodes

    else:  # component_type == "objects"
        if room_node_data is None:
            raise ValueError("room_node_data not specified.")
        else:
            r, room_data_dict = room_node_data

        # extracting H-tree of the single room room-object graph
        _, jth, root_nodes = sample_and_generate_jth(
            dsg_nx_component,
            k=treewidth_bound,
            zero_feature=zero_feature,
            copy_node_attributes=["x", "pos", "label", "node_type"],
            need_root_tree=True,
            remove_edges_every_layer=True,
            verbose=verbose,
        )

        # if object graph has only one node, "sample_and_generate_jth" returns nothing for room_root_nodes
        if len(dsg_nx_component) == 1:
            jth.add_node(
                1,
                x=zero_feature,
                pos=zero_pos,
                type="clique",
                clique_has=[jth.nodes[0]["clique_has"]],
            )
            jth.add_edge(0, 1)
            root_nodes = [1]

        # the clique nodes should all contain the room node, fill node_type with "object-room" and add room node r
        for _, data in jth.nodes.items():
            if data["type"] == "clique":
                data[
                    "node_type"
                ] = "object-room"  # object cliques, identified as node_type=object-room
                data["clique_has"].append(r)  # add room node to clique

        # adding room node r to all lowest level cliques
        # finding the lowest level cliques
        leaf_nodes = [
            idx for idx, data_dict in jth.nodes.items() if data_dict["type"] == "node"
        ]
        lowest_level_clique_nodes = set(
            sum([[n for n in jth.neighbors(leaf_idx)] for leaf_idx in leaf_nodes], [])
        )

        # adding a new compy of the room node, and an edge
        idx_count = jth.number_of_nodes()
        for clique_idx in lowest_level_clique_nodes:
            jth.add_node(
                idx_count,
                x=room_data_dict["x"],
                pos=room_data_dict["pos"],
                label=room_data_dict["label"],
                node_type=room_data_dict["node_type"],
                type="node",
                clique_has=r,
            )

            jth.add_edge(clique_idx, idx_count)

            # increase _idx_count by 1
            idx_count += 1

        return jth, root_nodes


class HTree:
    """
    Helper class to merge room tree and room-object trees.
    """

    def __init__(self, room_jth, room_root_nodes):
        self.jth = room_jth
        self.root_nodes = room_root_nodes
        self.num_nodes = self.jth.number_of_nodes()

    def _reindex_and_compose(self, g, _root_nodes):
        # relabeling object_jth
        new_labels = dict(
            zip(
                range(g.number_of_nodes()),
                range(self.num_nodes, self.num_nodes + g.number_of_nodes()),
            )
        )

        g = nx.relabel_nodes(g, new_labels)
        _roots = [new_labels[i] for i in _root_nodes]

        # adding object_jth to jth
        self.jth = nx.compose(self.jth, g)
        self.num_nodes = self.jth.number_of_nodes()

        return g, _roots

    def add_object_jth(self, object_jth, object_root_nodes, room_idx):
        # disconnecting all edges between object_root_nodes in object_jth
        H = object_jth.subgraph(object_root_nodes)
        for edge in H.edges:
            object_jth.remove_edge(edge[0], edge[1])

        # for root node that contains room_idx
        for _root_node in self.root_nodes:
            if not isinstance(self.jth.nodes[_root_node]["clique_has"], list):
                self.jth.nodes[_root_node]["clique_has"] = [
                    self.jth.nodes[_root_node]["clique_has"]
                ]

            if room_idx in self.jth.nodes[_root_node]["clique_has"]:
                g, _roots = self._reindex_and_compose(
                    object_jth.copy(), object_root_nodes
                )

                for _object_root_idx in _roots:
                    self.jth.add_edge(_root_node, _object_root_idx)


def generate_htree(dsg_torch, verbose=False):
    """
    Takes in heterogeneous dsg_torch and generates h-tree using the sequential procedure.
    """

    # converting to networkx
    dsg_nx = torch_dsg_to_nx(dsg_torch)

    # iterating over each connected component in dsg_nx
    htree_nx = (
        nx.Graph()
    )  # add connected component to this nx graph object for final output
    for i, c in enumerate(nx.connected_components(dsg_nx)):
        # extracting connected component, from node index list c
        dsg_component = dsg_nx.subgraph(c).copy()

        # extracting room graph
        dsg_component_room, room_idx = get_room_graph(dsg_component)

        # extracting H-tree of the room graph
        dsg_component_room_jth, room_root_nodes = generate_component_jth(
            dsg_nx_component=dsg_component_room,
            component_type="rooms",
            num_zero_padding=dsg_torch["rooms"].num_features,
            verbose=verbose,
        )
        # create a Htree
        _htree = HTree(room_jth=dsg_component_room_jth, room_root_nodes=room_root_nodes)

        # extracting jth of the object graphs in a room
        for r in room_idx:
            # extracting all objects in room r
            dsg_component_object_r, object_idx = get_object_graph(dsg_component, r)

            for oc in nx.connected_components(dsg_component_object_r):
                # extracting object sub-graph
                object_graph_component = dsg_nx.subgraph(oc).copy()

                # extracting H-tree of the object graph component
                object_component_jth, object_root_nodes = generate_component_jth(
                    dsg_nx_component=object_graph_component,
                    component_type="objects",
                    room_node_data=(r, dsg_component.nodes[r]),
                    num_zero_padding=dsg_torch["objects"].num_features,
                    verbose=verbose,
                )

                # adding to _htree
                _htree.add_object_jth(
                    object_jth=object_component_jth,
                    object_root_nodes=object_root_nodes,
                    room_idx=r,
                )

        # if there is only one room node in this connected component, change room node clique_has attribute to int
        if len(room_idx) == 1 and _htree.jth.number_of_nodes() > 1:
            assert _htree.jth.nodes[0]["type"] == "node"
            assert len(_htree.jth.nodes[0]["clique_has"]) == 1
            _htree.jth.nodes[0]["clique_has"] = _htree.jth.nodes[0]["clique_has"][0]

        if verbose:
            print(
                f"Component {i}: H-tree contains {_htree.jth.number_of_nodes()} nodes "
                f"and {_htree.jth.number_of_edges()} edges."
            )

        # add htree component
        htree_nx = nx.disjoint_union(htree_nx, _htree.jth)

    return htree_nx


def add_virtual_nodes_to_htree(htree_nx):
    """
    Add virtual nodes which are copies of the original dsg, connect corresponding leaf nodes in htree to the
    virtual nodes via pooling edges, and connect virtual nodes to the clique nodes via init edges.
    The pooling edges are used to pool final node hidden states and the init edges are to initialize clique features.
    """
    assert "virtual" not in [
        data_dict["type"] for data_dict in htree_nx.nodes.values()
    ], "Virtual node already in input graph."

    # convert input graph to a directed graph, since virtual edges are directed
    htree_output = htree_nx.to_directed()

    # get leaf and clique nodes indices in the origianl input htree data
    leaf_nodes = [
        idx
        for idx, data_dict in htree_output.nodes.items()
        if data_dict["type"] == "node"
    ]
    clique_nodes = [
        idx
        for idx, data_dict in htree_output.nodes.items()
        if data_dict["type"] == "clique"
    ]

    idx_offset = htree_output.number_of_nodes()  # index offset for virtual nodes
    assert all(
        (idx in range(idx_offset) for idx in htree_output.nodes)
    ), "Input graph must have nodes labeled using consecutive integers, try nx.convert_node_labels_to_integers."

    # add virtual nodes and connect leaf nodes to virtual nodes
    for leaf_idx in leaf_nodes:
        node_data_dict = htree_output.nodes[leaf_idx]
        virtual_node_idx = node_data_dict["clique_has"] + idx_offset

        if virtual_node_idx not in htree_output.nodes:
            kwargs = {
                "x": node_data_dict["x"],
                "pos": node_data_dict["pos"],
                "label": node_data_dict["label"],
                "node_type": f"{node_data_dict['node_type']}_virtual",
                "type": "virtual",
                "clique_has": node_data_dict["clique_has"],
            }
            if "node_ids" in node_data_dict:
                kwargs["node_ids"] = node_data_dict["node_ids"]

            htree_output.add_node(virtual_node_idx, **kwargs)

        htree_output.add_edge(leaf_idx, virtual_node_idx)

    # connect virtual nodes to clique nodes
    for clique_idx in clique_nodes:
        node_data_dict = htree_output.nodes[clique_idx]
        for node_idx in node_data_dict["clique_has"]:
            virtual_node_idx = node_idx + idx_offset
            htree_output.add_edge(virtual_node_idx, clique_idx)

    return htree_output


def nx_htree_to_torch(
    htree_nx,
    node_type_names=HTREE_NODE_TYPES + HTREE_VIRTUAL_NODE_TYPES,
    edge_type_names=HTREE_EDGE_TYPES + HTREE_POOL_EDGE_TYPES + HTREE_INIT_EDGE_TYPES,
    double_precision: bool = False,
):
    """
    Converts an networkx htree to heterogeneous torch data.
    Clique nodes will have 'label' and 'clique_has' attributes set to -1.
    """
    if not all((idx in range(htree_nx.number_of_nodes()) for idx in htree_nx.nodes)):
        raise Warning("Relabeling the input graph nodes using consecutive integers.")

    # output torch tensor data types
    if double_precision:
        dtype_int = torch.int64
        dtype_float = torch.float64
    else:
        dtype_int = torch.int32
        dtype_float = torch.float32

    # node attributes and node types
    x_ = []
    pos_ = []
    label_ = []
    node_type_ = []
    clique_has_ = []
    node_ids_ = []

    read_pos_from = "room_virtual" if "room_virtual" in node_type_names else "room"
    for i in range(htree_nx.number_of_nodes()):
        data_dict = htree_nx.nodes[i]
        node_type_.append(node_type_names.index(data_dict["node_type"]))

        if data_dict["type"] == "clique":
            # clique node position is the average position of neighboring room nodes
            room_predecessor_pos = [
                htree_nx.nodes[idx]["pos"]
                for idx in htree_nx.predecessors(i)
                if htree_nx.nodes[idx]["node_type"] == read_pos_from
            ]
            if room_predecessor_pos:
                pos_.append(
                    torch.tensor(
                        np.mean(room_predecessor_pos, axis=0), dtype=dtype_float
                    )
                )
            else:  # will not get to this if read_pos_from == room_virtual
                pos_.append(torch.zeros(3, dtype=dtype_float))
            x_.append(
                torch.hstack(
                    (pos_[-1], torch.tensor(data_dict["x"][3:], dtype=dtype_float))
                )
            )
            label_.append(-1)
            node_ids_.append(-1)
            clique_has_.append(-1)
        else:
            x_.append(torch.tensor(data_dict["x"], dtype=dtype_float))
            pos_.append(torch.tensor(data_dict["pos"], dtype=dtype_float))
            label_.append(data_dict["label"])
            node_ids_.append(data_dict.get("node_ids", -1))
            clique_has_.append(data_dict["clique_has"])

    x_ = torch.vstack(x_)
    pos_ = torch.vstack(pos_)
    label_ = torch.tensor(label_, dtype=dtype_int)
    node_type_ = torch.tensor(node_type_)
    clique_has_ = torch.tensor(clique_has_, dtype=dtype_int)
    node_ids_ = torch.tensor(node_ids_, dtype=torch.int64)

    # edge types
    edges = list(htree_nx.edges)
    edge_index = torch.tensor(edges).t().contiguous()

    edge_type_ = []
    for i, j in edges:
        node_type_i = htree_nx.nodes[i]["node_type"]
        node_type_j = htree_nx.nodes[j]["node_type"]

        edge_type_idx = next(
            idx
            for idx, edge_type in enumerate(edge_type_names)
            if (edge_type[0] == node_type_i and edge_type[2] == node_type_j)
        )
        edge_type_.append(edge_type_idx)

    edge_type_ = torch.tensor(edge_type_)

    # generate homogeneous graph
    htree_homogeneous = Data(
        x=x_,
        edge_index=edge_index,
        label=label_,
        pos=pos_,
        clique_has=clique_has_,
        node_type=node_type_,
        edge_type=edge_type_,
        node_ids=node_ids_,
    )

    # extracting the heterogeneous graph
    htree_torch = htree_homogeneous.to_heterogeneous(
        node_type=node_type_,
        edge_type=edge_type_,
        node_type_names=node_type_names,
        edge_type_names=edge_type_names,
    )

    return htree_torch
