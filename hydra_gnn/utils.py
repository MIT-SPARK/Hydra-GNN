import plotly.graph_objects as go
import plotly.express as px
import torch_geometric
import os.path
from typing import Any, Sequence, Iterator
from io import IOBase
import itertools
import numpy as np


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MP3D_BENCHMARK_DIR = os.path.join(DATA_DIR, "mp3d_benchmark")
MP3D_HOUSE_DIR = os.path.join(DATA_DIR, "house_files")
HYDRA_TRAJ_DIR = os.path.join(DATA_DIR, "tro_graphs_2022_09_24")
COLORMAP_DATA_PATH = os.path.join(DATA_DIR, "colormap.csv")
WORD2VEC_MODEL_PATH = os.path.join(DATA_DIR, "GoogleNews-vectors-negative300.bin")


def print_log(string, file):
    print(string)
    print(string, file=file)


def update_existing_keys(dict_to_update, dict_input):
    dict_to_update.update({k:v for k, v in dict_input.items() if k in dict_to_update.keys()})


# -----------------------------------------------------------------------------
# Plot heterogenous torch data
# -----------------------------------------------------------------------------
def _filter_empty_nodes(torch_data):
    return lambda node_type: torch_data[node_type].num_nodes is not None and torch_data[node_type].num_nodes > 0


def plot_heterogeneous_graph(torch_data, node_filter_func=_filter_empty_nodes, **kwargs):
    assert isinstance(torch_data, torch_geometric.data.HeteroData)

    # plot style params
    marker_size = 6 if 'marker_size' not in kwargs else kwargs['marker_size']
    title = 'Heterogeneous Scene Graph' if 'title' not in kwargs else kwargs['title']
    z_axis_offset = 3.0 if 'z_offset' not in kwargs else kwargs['z_offset']

    fig = go.Figure()
    fig.layout.title = title

    # node and types to be plotted
    node_types = list(filter(node_filter_func(torch_data), torch_data.node_types))
    edge_types = torch_data.edge_types

    # scene nodes
    for i, node_type in enumerate(node_types):
        pos = torch_data[node_type]['pos']
        if 'label' in torch_data.keys:
            label = torch_data[node_type].label.tolist()
            node_label = [f"label: {l}" for l in label]
        else:
            node_label = None
        plotly_nodes = go.Scatter3d(x=pos[:, 0],
                                    y=pos[:, 1],
                                    z=pos[:, 2] + i * z_axis_offset,
                                    mode='markers',
                                    marker=dict(size=marker_size, 
                                                opacity=0.8,
                                                color=px.colors.qualitative.Plotly[i*2]),
                                    hovertext=node_label,
                                    name=node_type)
        fig.add_trace(plotly_nodes)

    # scene edges
    for source_node_type, edge_name, target_node_type in edge_types:
        edge_index = torch_data[source_node_type, edge_name, target_node_type].edge_index

        source_node_pos = torch_data[source_node_type]['pos'][edge_index[0, :], :]
        target_node_pos = torch_data[target_node_type]['pos'][edge_index[1, :], :]

        num_edges = edge_index.shape[1]

        source_offset = node_types.index(source_node_type) * z_axis_offset
        target_offset = node_types.index(target_node_type) * z_axis_offset
        x_list = sum([[source_node_pos[i, 0].item(), target_node_pos[i, 0].item(), None]
                      for i in range(num_edges)], [])
        y_list = sum([[source_node_pos[i, 1].item(), target_node_pos[i, 1].item(), None]
                      for i in range(num_edges)], [])
        z_list = sum([[source_node_pos[i, 2].item() + source_offset, target_node_pos[i, 2].item() + target_offset, None]
                      for i in range(num_edges)], [])
        if source_node_type == target_node_type:
            color = px.colors.qualitative.Plotly[node_types.index(source_node_type) + len(node_types)]
        else:
            color = px.colors.qualitative.Plotly[node_types.index(source_node_type)]
        plotly_edge = go.Scatter3d(x=x_list,
                                   y=y_list,
                                   z=z_list,
                                   mode='lines',
                                   line=dict(width=2, color=color),
                                   name=edge_name)
        fig.add_trace(plotly_edge)

    return fig


# -----------------------------------------------------------------------------
# Parameter sweep
# https://github.com/elcorto/psweep
# -----------------------------------------------------------------------------
def is_seq(seq) -> bool:
    if (
        isinstance(seq, str)
        or isinstance(seq, IOBase)
        or isinstance(seq, dict)
    ):
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
    assert (
        len(np.unique(lens)) == 1
    ), f"not all psets have same length"
    return ret


def pgrid(*plists):
    assert is_seq(plists), f"input plists={plists} is no sequence"
    return itr2params(itertools.product(*plists))
