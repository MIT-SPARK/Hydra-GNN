"""Utilities for handling H-Trees during training."""
import networkx as nx
import torch
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data
from copy import deepcopy
from neural_tree.h_tree import generate_jth, generate_node_labels


class HTreeDataset:
    """
    H-Tree dataset
        data_list: a list of torch_geometric.data.Data instances (graph classification) or a list of three such lists
         corresponding to train, val, test split (node classification)
        num_node_features:  int
        num_classes:        int
        name:               string
        task:               string (node or graph)
    """

    def __init__(
        self,
        data_list,
        num_node_features,
        num_classes,
        name,
        task,
        data_list_original=None,
    ):
        assert isinstance(num_node_features, int)
        if data_list_original is not None:
            assert isinstance(data_list_original[0], Data)
        assert isinstance(num_classes, int) or isinstance(num_classes, tuple)
        assert task == "graph" or task == "node" or task == "synthetic"
        if task == "graph" or task == "synthetic":
            assert isinstance(data_list[0], Data)
        else:
            assert len(data_list) == 3
            for i in range(3):
                assert len(data_list) == 0 or isinstance(data_list[i][0], Data)

        self.dataset_jth = data_list
        self.dataset_original = data_list_original
        self.num_node_features = num_node_features
        self.num_classes = num_classes
        self.name = name
        self.task = task


def convert_to_networkx_jth(data: Data, task="graph", node_id=None, radius=None):
    """
    Helper function: convert loopy graph to junction tree hierarchy
    """
    # Convert to networkx graph
    G = pyg_utils.to_networkx(data, node_attrs=["x"])
    G = nx.to_undirected(G)

    if task == "graph" or task == "synthetic":
        if nx.is_connected(G) is False:
            print("[Input graph] is disconnected.")

    else:  # task == 'node'
        if radius is not None:
            G_subgraph = nx.ego_graph(G, node_id, radius=radius, undirected=False)
            extracted_id = [i for i in G_subgraph.nodes.keys()]
            G_subgraph = nx.relabel_nodes(
                G_subgraph,
                dict(zip(extracted_id, list(range(len(G_subgraph))))),
                copy=True,
            )
            G = generate_node_labels(G_subgraph)
        else:
            extracted_id = [i for i in G.nodes.keys()]
            G = generate_node_labels(G)
        # index of the classification node in the extracted graph, for computing leaf_mask
        classification_node_id = extracted_id.index(node_id)

    is_clique_graph = (
        True
        if len(list(G.edges)) == G.number_of_nodes() * (G.number_of_nodes() - 1) / 2
        else False
    )

    # Create junction tree hierarchy
    G.graph = {"original": True}
    if task == "synthetic":
        zero_feature = 0.0
    else:
        zero_feature = [0.0] * data.num_node_features
    G_jth, root_nodes = generate_jth(G, zero_feature=zero_feature)

    # Convert back to torch Data (change first clique_has value to avoid TypeError when calling pyg_utils.from_networkx
    if is_clique_graph:  # clique graph
        G_jth.nodes[0]["clique_has"] = 0
    else:
        G_jth.nodes[0]["clique_has"] = [0]
    data_jth = pyg_utils.from_networkx(G_jth)

    try:  # todo: collect failure cases where input graph is not disconnected but output is (failure mode)
        data_jth["diameter"] = nx.diameter(G_jth)
    except nx.NetworkXError:
        data_jth["diameter"] = 0
        print("junction tree hierarchy disconnected.")
        return data_jth

    if task == "node":
        data_jth["classification_node"] = classification_node_id

    return data_jth, G_jth, root_nodes


def convert_to_junction_tree_hierarchy(
    data: Data, task="graph", node_id=None, radius=None
) -> Data:
    """
    Convert a loopy graph (torch_geometric.data.Data) to junction tree hierarchy.
    In addition to input feature x and label y, the output Data instance have additional task specific attributes.
    If task = 'graph':
        data_jth.root_mask, a BoolTensor of dimension [data_jth.num_nodes] specifying top-level root nodes
    else (task = 'node'):
        data_jth.leaf_mask, a BoolTensor of dimension [data_jth.num_nodes] specifying leaf nodes
        data_jth.leaf_cluster, a LongTensor of dimension [data_jth.num_nodes] mapping nodes in data_jth to nodes in data
         (the original node id at non-leaf nodes will not be used)
    """
    assert isinstance(data, Data)
    assert task == "graph" or "node"
    if task == "node":
        assert node_id is not None

    data_jth, G_jth, root_nodes = convert_to_networkx_jth(data, task, node_id, radius)

    # Save node mask/cluster based on task
    if task == "graph":  # save top-level root nodes
        root_mask = torch.zeros(data_jth.num_nodes, dtype=torch.bool)
        root_mask[root_nodes] = True
        data_jth["root_mask"] = root_mask
        data_jth.y = data.y
    elif task == "synthetic":
        data_jth.x = data_jth.x.reshape(data_jth.num_nodes, 1)
        data_jth.y = data.y
    else:  # save leaf nodes
        leaf_mask = torch.zeros(data_jth.num_nodes, dtype=torch.bool)
        for v, attr in G_jth.nodes("type"):
            if (
                attr == "node"
                and G_jth.nodes[v]["clique_has"] == data_jth["classification_node"]
            ):
                leaf_mask[v] = True
        data_jth["leaf_mask"] = leaf_mask
        data_jth.y = data.y[node_id]
        if "y_room" in data.keys:
            data_jth.y_room = data.y_room[node_id]
            data_jth.y_object = data.y_object[node_id]

    # remove clique_has
    data_jth.clique_has = None

    return data_jth


def convert_to_same_junction_tree_hierarchy(
    data: Data, min_diameter=None, max_diameter=None
):
    """
    For node classification, convert the entire graph to junction tree hierarchy. From this junction tree hierarchy,
    create three lists corresponding to nodes in train_mask, val_mask and tes_mask.
    """
    assert isinstance(data, Data)
    train_list = []
    val_list = []
    test_list = []

    data_jth, G_jth, root_nodes = convert_to_networkx_jth(data, "node", 0, None)

    # return empty lists if diameter is beyond specified bound
    try:
        data_jth["diameter"] = nx.diameter(G_jth)
    except nx.NetworkXError:
        data_jth["diameter"] = 0
        print("junction tree hierarchy disconnected.")
        return data_jth
    if (min_diameter is not None and data_jth.diameter < min_diameter) or (
        max_diameter is not None and data_jth.diameter > max_diameter
    ):
        return train_list, val_list, test_list

    # prepare modified copies of data_jth based on train/val/test masks
    data_jth.clique_has = None
    for node_id in range(data.num_nodes):
        if (
            data.train_mask[node_id].item()
            or data.val_mask[node_id].item()
            or data.test_mask[node_id].item()
        ):
            # create a copy of data_jth and related attributes
            data_jth_i = deepcopy(data_jth)
            leaf_mask = torch.zeros(data_jth.num_nodes, dtype=torch.bool)
            for v, attr in G_jth.nodes("type"):
                if attr == "node" and G_jth.nodes[v]["clique_has"] == node_id:
                    leaf_mask[v] = True
            data_jth_i["leaf_mask"] = leaf_mask
            data_jth_i.y = data.y[node_id]
            if "y_room" in data.keys:
                data_jth_i.y_room = data.y_room[node_id]
                data_jth_i.y_object = data.y_object[node_id]
            # save to lists
            if data.train_mask[node_id].item() is True:
                train_list.append(data_jth_i)
            if data.val_mask[node_id].item() is True:
                val_list.append(data_jth_i)
            if data.test_mask[node_id].item() is True:
                test_list.append(data_jth_i)
    return train_list, val_list, test_list


def convert_dataset_to_junction_tree_hierarchy(
    dataset, task, min_diameter=None, max_diameter=None, radius=None
):
    """
    Convert a torch.dataset object or a list of torch.Data to a junction tree hierarchies.
    :param dataset:     a iterable collection of torch.Data objects
    :param task:            str, 'graph' or 'node'
    :param min_diameter:    int
    :param max_diameter:    int
    :param radius:          int, maximum radius of extracted sub-graphs for node classification
    :return: if task = 'graph', return a list of torch.Data objects in the same order as in dataset;
     else (task = 'node'), return a list of three such lists, for nodes and the corresponding subgraph in train_mask,
     val_mask, and test_mask respectively.
    """
    if task == "graph" or task == "synthetic":
        output_list = []
        for data in dataset:
            data_jth = convert_to_junction_tree_hierarchy(data, task)
            if (min_diameter is None or data_jth.diameter >= min_diameter) and (
                max_diameter is None or data_jth.diameter <= max_diameter
            ):
                output_list.append(data_jth)
        return output_list
    elif task == "node":
        train_list = []
        val_list = []
        test_list = []
        for data in dataset:
            if (
                radius is None
            ):  # for nodes in the same graph, use the same junction tree hierarchy
                (
                    train_graphs,
                    val_graphs,
                    test_graphs,
                ) = convert_to_same_junction_tree_hierarchy(
                    data, min_diameter, max_diameter
                )
                train_list += train_graphs
                val_list += val_graphs
                test_list += test_graphs
            else:  # otherwise, create jth for each node separately
                progress_threshold = 0
                print("Begin converting ego graphs to JTH.")
                for i in range(data.num_nodes):
                    if 100.0 * i / data.num_nodes > progress_threshold + 1:
                        progress_threshold += 1
                        print(
                            "Progress converting ego graphs to JTH: ",
                            progress_threshold,
                            "/",
                            "100 %",
                        )
                    if (
                        data.train_mask[i].item()
                        or data.val_mask[i].item()
                        or data.test_mask[i].item()
                    ):
                        data_jth = convert_to_junction_tree_hierarchy(
                            data, task, node_id=i, radius=radius
                        )
                        if (
                            min_diameter is None or data_jth.diameter >= min_diameter
                        ) and (
                            max_diameter is None or data_jth.diameter <= max_diameter
                        ):
                            if data.train_mask[i].item() is True:
                                train_list.append(data_jth)
                            elif data.val_mask[i].item() is True:
                                val_list.append(data_jth)
                            elif data.test_mask[i].item() is True:
                                test_list.append(data_jth)
                print("Progress converting ego graphs to JTH: 100 / 100 %")
        return [train_list, val_list, test_list]
    else:
        raise Exception("must specify if task is 'graph' or 'node' classification")


def get_subtrees_from_jth(data, G_jth, radius):
    """
    Segment sub-trees from JTH where each tree correspond to a label node in the original graph. Note: if the original
     graph is disconnected, G_jth is the JTH of one of the connected component of the original graph.
    :param data:    torch.data, original graph
    :param G_jth:   nx.Graph, junction tree hierarchy of the (subsampled) original graph or one of the connected
    (subsampled) component of the original graph
    :param radius:  int, furthest neighbor node from leaf nodes corresponding to a label node
    :return:
    """
    # save leaf node indices for each node in the original graph
    leaf_nodes_list = [None] * data.num_nodes
    original_idx_set = set()  # indices of original nodes in data that are in G_jth
    for i, attr in G_jth.nodes("type"):
        if attr == "node":
            original_idx = G_jth.nodes[i]["clique_has"]
            original_idx_set.add(original_idx)
            if leaf_nodes_list[original_idx] is None:
                leaf_nodes_list[original_idx] = [i]
            else:
                leaf_nodes_list[original_idx].append(i)
    original_idx_list = list(original_idx_set)
    num_original_nodes = len(original_idx_list)

    # loop through each node in the original graph
    train_list = []
    val_list = []
    test_list = []
    data_mask = data.train_mask + data.val_mask + data.test_mask  # classification nodes
    progress_threshold = 0
    max_num_nodes = 0
    for j in range(num_original_nodes):
        original_idx = original_idx_list[j]
        if 100.0 * (j + 1) / num_original_nodes > progress_threshold + 10:
            progress_threshold += 10
            # print("Progress Segmenting Subtree: ", int(100.0 * j / num_original_nodes), "/", "100 %")
        if data_mask[original_idx].item() is True:
            # segment subtree from the complete jth using specified radius from leaf nodes
            leaf_nodes = leaf_nodes_list[original_idx]
            G_subtree = nx.ego_graph(
                G_jth, leaf_nodes[0], radius=radius, undirected=False
            )
            for leaf_node in leaf_nodes[
                1:
            ]:  # add other subtrees if there are multiple leaf nodes
                if G_subtree.number_of_nodes() == G_jth.number_of_nodes():
                    break
                H_subtree = nx.ego_graph(
                    G_jth, leaf_node, radius=radius, undirected=False
                )
                G_subtree = nx.compose(G_subtree, H_subtree)
            extracted_id = [i for i in G_subtree.nodes.keys()]
            G_subtree = nx.relabel_nodes(
                G_subtree,
                dict(zip(extracted_id, list(range(len(G_subtree))))),
                copy=True,
            )

            # convert G_subtree to torch data
            leaf_mask = torch.zeros(G_subtree.number_of_nodes(), dtype=torch.bool)
            for v, attr in G_subtree.nodes("type"):
                if attr == "node" and G_subtree.nodes[v]["clique_has"] == original_idx:
                    leaf_mask[v] = True
                del G_subtree.nodes[v]["clique_has"]
                del G_subtree.nodes[v]["type"]
            data_subtree = pyg_utils.from_networkx(G_subtree)
            data_subtree.leaf_mask = leaf_mask
            data_subtree.y = data.y[original_idx]
            if nx.is_connected(G_subtree):
                data_subtree.diameter = nx.diameter(G_subtree)
            else:
                data_subtree.diameter = 0
            if data_subtree.num_nodes > max_num_nodes:
                max_num_nodes = data_subtree.num_nodes
                # print('max number of nodes: {}'.format(max_num_nodes))

            # save subtree
            if data.train_mask[original_idx].item() is True:
                train_list.append(data_subtree)
            elif data.val_mask[original_idx].item() is True:
                val_list.append(data_subtree)
            else:
                test_list.append(data_subtree)
    # print("Progress Segmenting Subtree: 100 / 100 %")
    return train_list, val_list, test_list
