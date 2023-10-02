"""Path utilities."""
from hydra_gnn.neural_tree.construct import HTREE_NODE_TYPES, HTREE_EDGE_TYPES
import numpy as np
import torch_geometric
from torch_geometric.utils import to_networkx
from torch_geometric.data import HeteroData, Data
import torch
import networkx as nx
from networkx.algorithms.approximation.treewidth import (
    treewidth_min_degree,
    treewidth_min_fill_in,
)
import plotly.graph_objects as go
import plotly.express as px
from typing import Any, Sequence, Iterator
from io import IOBase
import itertools
import pathlib


def project_dir():
    """Get project directory."""
    pkg_path = pathlib.Path(__file__).absolute().parent
    return pkg_path.parent.parent


def data_dir():
    """Get data directory."""
    return project_dir() / "data"


PROJECT_DIR = str(project_dir())
DATA_DIR = str(data_dir())
MP3D_BENCHMARK_DIR = str(data_dir() / "mp3d_benchmark")
MP3D_HOUSE_DIR = str(data_dir() / "house_files")
MP3D_OBJECT_LABEL_DATA_PATH = str(data_dir() / "mpcat40.tsv")
HYDRA_TRAJ_DIR = str(data_dir() / "tro_graphs_2022_09_24")
COLORMAP_DATA_PATH = str(data_dir() / "colormap.csv")
STANFORD3DSG_DATA_DIR = str(data_dir() / "Stanford3DSceneGraph/tiny/verified_graph")
STANFORD3DSG_GRAPH_PATH = str(data_dir() / "Stanford3DSG.pkl")
WORD2VEC_MODEL_PATH = str(data_dir() / "GoogleNews-vectors-negative300.bin")


def print_log(string, file):
    """Print and log message to file."""
    print(string)
    print(string, file=file)


def update_existing_keys(dict_to_update, dict_input):
    """Copy values over if key previously existed."""
    dict_to_update.update(
        {k: v for k, v in dict_input.items() if k in dict_to_update.keys()}
    )


# -----------------------------------------------------------------------------
# Data analysis: graph treewidth, htree diameter
# -----------------------------------------------------------------------------
def extract_object_graph(graph_torch, to_nx=True):
    if not graph_torch["objects"]:
        object_graph = Data()
        if to_nx:
            object_graph = nx.empty_graph()
    else:
        if graph_torch["objects", "objects_to_objects", "objects"]:
            object_graph = Data(
                x=graph_torch["objects"].x,
                pos=graph_torch["objects"].pos,
                y=graph_torch["objects"].label,
                edge_index=graph_torch["objects", "objects_to_objects", "objects"][
                    "edge_index"
                ],
            )
        else:
            object_graph = Data(
                x=graph_torch["objects"].x,
                pos=graph_torch["objects"].pos,
                y=graph_torch["objects"].label,
                edge_index=torch.zeros(2, 0).to(dtype=torch.long),
            )
        if to_nx:
            object_graph = to_networkx(object_graph).to_undirected()

    return object_graph


def extract_room_graph(graph_torch, to_nx=True):
    if not graph_torch["rooms"]:
        room_graph = Data()
        if to_nx:
            room_graph = nx.empty_graph()
    else:
        if graph_torch["rooms", "rooms_to_rooms", "rooms"]:
            room_graph = Data(
                x=graph_torch["rooms"].x,
                pos=graph_torch["rooms"].pos,
                y=graph_torch["rooms"].label,
                edge_index=graph_torch["rooms", "rooms_to_rooms", "rooms"][
                    "edge_index"
                ],
            )
        else:
            room_graph = Data(
                x=graph_torch["rooms"].x,
                pos=graph_torch["rooms"].pos,
                y=graph_torch["rooms"].label,
                edge_index=torch.zeros(2, 0).to(dtype=torch.long),
            )
        if to_nx:
            room_graph = to_networkx(room_graph).to_undirected()

    return room_graph


def get_treewidth(torch_data, graph_type="full"):
    """
    Find treewidth upper bound of the full input PyG graph, or the object subgraphs, or the room subgraph,
    using approximate algorithms.
    """
    assert graph_type in ["full", "object", "room"]
    if graph_type == "full":
        nx_graph = (
            to_networkx(torch_data.to_homogeneous()).to_undirected()
            if isinstance(torch_data, HeteroData)
            else to_networkx(torch_data).to_undirected()
        )
    elif graph_type == "object":
        nx_graph = extract_object_graph(torch_data, to_nx=True)
    else:
        nx_graph = extract_room_graph(torch_data, to_nx=True)

    num_nodes = nx_graph.number_of_nodes()
    num_edges = nx_graph.number_of_edges()
    if num_nodes:
        if num_edges:
            tw1, _ = treewidth_min_degree(nx_graph)
            tw2, _ = treewidth_min_fill_in(nx_graph)
            return max([tw1, tw2])
        else:
            return 0
    else:
        return 0


def get_diameters(torch_data, valid_room_labels=None):
    """
    Get diameter of an augmented htree by first creating a copy of the htree without the virtual nodes.
    """
    assert isinstance(torch_data, HeteroData)

    # generate a data copy with just htree nodes (i.e. remove all virtual nodes)
    data_htree = HeteroData()
    for node_type in HTREE_NODE_TYPES:
        data_htree[node_type].pos = torch_data[node_type].pos
        data_htree[node_type].label = torch_data[node_type].label
        data_htree[node_type].clique_has = torch_data[node_type].clique_has
    for edge_type in HTREE_EDGE_TYPES:
        data_htree[edge_type].edge_index = torch_data[edge_type].edge_index
    data_htree = data_htree.to_homogeneous()

    # find diameter of each connected component using networkx
    nx_data = to_networkx(
        data_htree, node_attrs=["label", "clique_has"]
    ).to_undirected()
    diameters = []
    valid_room_ids = []
    for c in nx.connected_components(nx_data):
        subgraph = nx_data.subgraph(c)
        if valid_room_labels is None:
            diameters.append(nx.diameter(subgraph))
        else:  # skip subgraphs where all rooms are unlabeld -- i.e. ignored by training
            # todo: this code does not distinguish room and object nodes -- but works on hydra labels
            room_idx = [
                idx
                for idx, data_dict in subgraph.nodes.items()
                if data_dict["label"] in valid_room_labels
            ]
            # this assumes objects are saved before rooms in htree construction
            offset = torch_data["object_virtual"].num_nodes
            if len(room_idx) != 0:
                diameters.append(nx.diameter(subgraph))
                valid_room_ids.append(
                    set(subgraph.nodes[idx]["clique_has"] - offset for idx in room_idx)
                )

    if valid_room_labels is not None:
        return diameters, valid_room_ids
    else:
        return diameters


# -----------------------------------------------------------------------------
# Plot heterogenous torch data
# -----------------------------------------------------------------------------
def _filter_empty_nodes(torch_data):
    return (
        lambda node_type: torch_data[node_type].num_nodes is not None
        and torch_data[node_type].num_nodes > 0
    )


def plot_heterogeneous_graph(
    torch_data, node_filter_func=_filter_empty_nodes, **kwargs
):
    assert isinstance(torch_data, torch_geometric.data.HeteroData)

    # plot style params
    marker_size = 6 if "marker_size" not in kwargs else kwargs["marker_size"]
    title = "Heterogeneous Scene Graph" if "title" not in kwargs else kwargs["title"]
    z_axis_offset = 3.0 if "z_offset" not in kwargs else kwargs["z_offset"]

    fig = go.Figure()
    fig.layout.title = title

    # node and types to be plotted
    node_types = list(filter(node_filter_func(torch_data), torch_data.node_types))
    edge_types = torch_data.edge_types

    # scene nodes
    for i, node_type in enumerate(node_types):
        pos = torch_data[node_type]["pos"]
        if "label" in torch_data.keys:
            label = torch_data[node_type].label.tolist()
            node_label = [f"label: {l}" for l in label]
        else:
            node_label = None
        plotly_nodes = go.Scatter3d(
            x=pos[:, 0],
            y=pos[:, 1],
            z=pos[:, 2] + i * z_axis_offset,
            mode="markers",
            marker=dict(
                size=marker_size, opacity=0.8, color=px.colors.qualitative.Plotly[i * 2]
            ),
            hovertext=node_label,
            name=node_type,
        )
        fig.add_trace(plotly_nodes)

    # scene edges
    for source_node_type, edge_name, target_node_type in edge_types:
        edge_index = torch_data[
            source_node_type, edge_name, target_node_type
        ].edge_index

        source_node_pos = torch_data[source_node_type]["pos"][edge_index[0, :], :]
        target_node_pos = torch_data[target_node_type]["pos"][edge_index[1, :], :]

        num_edges = edge_index.shape[1]

        source_offset = node_types.index(source_node_type) * z_axis_offset
        target_offset = node_types.index(target_node_type) * z_axis_offset
        x_list = sum(
            [
                [source_node_pos[i, 0].item(), target_node_pos[i, 0].item(), None]
                for i in range(num_edges)
            ],
            [],
        )
        y_list = sum(
            [
                [source_node_pos[i, 1].item(), target_node_pos[i, 1].item(), None]
                for i in range(num_edges)
            ],
            [],
        )
        z_list = sum(
            [
                [
                    source_node_pos[i, 2].item() + source_offset,
                    target_node_pos[i, 2].item() + target_offset,
                    None,
                ]
                for i in range(num_edges)
            ],
            [],
        )
        if source_node_type == target_node_type:
            color = px.colors.qualitative.Plotly[
                node_types.index(source_node_type) + len(node_types)
            ]
        else:
            color = px.colors.qualitative.Plotly[node_types.index(source_node_type)]
        plotly_edge = go.Scatter3d(
            x=x_list,
            y=y_list,
            z=z_list,
            mode="lines",
            line=dict(width=2, color=color),
            name=edge_name,
        )
        fig.add_trace(plotly_edge)

    return fig


# -----------------------------------------------------------------------------
# Parameter sweep
# https://github.com/elcorto/psweep
# -----------------------------------------------------------------------------
def is_seq(seq) -> bool:
    if isinstance(seq, str) or isinstance(seq, IOBase) or isinstance(seq, dict):
        return False
    else:
        try:
            iter(seq)
            return True
        except TypeError:
            return False


def flatten(seq):
    for item in seq:
        if not is_seq(item):
            yield item
        else:
            for subitem in flatten(item):
                yield subitem


def plist(name: str, seq: Sequence[Any]):
    return [{name: entry} for entry in seq]


def merge_dicts(args: Sequence[dict]):
    dct = {}
    assert is_seq(args), f"input args={args} is no sequence"
    for entry in args:
        assert isinstance(entry, dict), f"entry={entry} is no dict"
        dct.update(entry)
    return dct


def itr2params(loops: Iterator[Any]):
    ret = [merge_dicts(flatten(entry)) for entry in loops]
    lens = list(map(len, ret))
    assert len(np.unique(lens)) == 1, f"not all psets have same length"
    return ret


def pgrid(*plists):
    assert is_seq(plists), f"input plists={plists} is no sequence"
    return itr2params(itertools.product(*plists))
