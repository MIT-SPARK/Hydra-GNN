"""Make stanford3d data."""
from hydra_gnn.utils import (
    project_dir,
    WORD2VEC_MODEL_PATH,
    STANFORD3DSG_DATA_DIR,
    STANFORD3DSG_GRAPH_PATH,
)
from hydra_gnn.Stanford3DSG_dataset import (
    Stanford3DSG_data,
    Stanford3DSG_htree_data,
    Stanford3DSG_object_feature_converter,
    Stanford3DSG_room_feature_converter,
)
from hydra_gnn.preprocess_dsgs import dsg_node_converter
import pathlib
import os
import click
import shutil
import numpy as np
import gensim
import pickle
import yaml


# data params (used only with --from_raw_data flag)
threshold_near = 1.5
max_near = 2.0
max_on = 0.2


def _make_zero_feature(x):
    return np.zeros(0)


@click.command()
@click.option("-n", "--output_filename", default=None, help="output file name")
@click.option(
    "-o",
    "--output_dir",
    default=str(project_dir() / "output/preprocessed_Stanford3DSG"),
    help="output directory (defaults to output/preprocessed_Stanford3DSG",
)
@click.option(
    "--from_raw_data",
    is_flag=True,
    help="use raw Stanford3DSG and specified edge data params to construct graphs"
    " (this will result in 35 multi-room graphs instead of 482 single room graphs)",
)
@click.option("--save_htree", is_flag=True, help="store htree data")
@click.option(
    "--save_word2vec",
    is_flag=True,
    help="store word2vec vectors as node features",
)
def main(output_filename, output_dir, from_raw_data, save_htree, save_word2vec):
    """Prepare stanford3d data for training."""
    param_filename = "params.yaml"
    if output_filename is None:
        output_filename = "htree" if save_htree else "data"
        output_filename += ".pkl" if from_raw_data else "_nt.pkl"

    print(f"Computing torch graphs from raw Stanford3DSG data: {from_raw_data}")
    print(f"Saving torch graphs as htree: {save_htree}")
    print(f"Saving torch graphs with word2vec features: {save_word2vec}")
    print(f"Output directory: {output_dir}")
    print(f"Output data files: {output_filename}, ({param_filename})")
    
    output_path = pathlib.Path(output_dir).expanduser().absolute() / output_filename
    if output_path.exists():
        input("Output data file exists. Press any key to proceed...")
    
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    if save_word2vec:
        word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
            WORD2VEC_MODEL_PATH, binary=True
        )
        object_feature_converter = Stanford3DSG_object_feature_converter(word2vec_model)
        room_feature_converter = Stanford3DSG_room_feature_converter(word2vec_model)
    else:
        object_feature_converter = _make_zero_feature
        room_feature_converter = _make_zero_feature

    node_converter = dsg_node_converter(
        object_feature_converter, room_feature_converter
    )

    # process dataset as a list of torch data
    data_list = []
    htree_construction_time = 0.0
    max_htree_construction_time = 0.0
    if from_raw_data:
        data_files = os.listdir(STANFORD3DSG_DATA_DIR)
        for i, data_file in enumerate(data_files):
            if save_htree:
                data = Stanford3DSG_htree_data(
                    os.path.join(STANFORD3DSG_DATA_DIR, data_file)
                )
            else:
                data = Stanford3DSG_data(os.path.join(STANFORD3DSG_DATA_DIR, data_file))
            data.add_object_edges(
                threshold_near=threshold_near, max_near=max_near, max_on=max_on
            )
            htree_time = data.compute_torch_data(
                use_heterogeneous=True, node_converter=node_converter
            )
            if save_htree:
                max_htree_construction_time = max(
                    max_htree_construction_time, htree_time
                )
                htree_construction_time += htree_time

            data.clear_dsg()
            data_list.append(data)
    else:
        with open(STANFORD3DSG_GRAPH_PATH, "rb") as input_file:
            saved_data_list, semantic_dict, num_labels = pickle.load(input_file)

        for i in range(len(saved_data_list["x_list"])):
            data_dict = {
                "x": saved_data_list["x_list"][i],
                "y": saved_data_list["y_list"][i],
                "edge_index": saved_data_list["edge_index_list"][i],
                "room_mask": saved_data_list["room_mask_list"][i],
            }
            if save_htree:
                data = Stanford3DSG_htree_data(
                    data_dict=data_dict,
                    room_semantic_dict=semantic_dict["room"],
                    object_semantic_dict=semantic_dict["object"],
                )
            else:
                data = Stanford3DSG_data(
                    data_dict=data_dict,
                    room_semantic_dict=semantic_dict["room"],
                    object_semantic_dict=semantic_dict["object"],
                )

            htree_time = data.compute_torch_data(
                use_heterogeneous=True, node_converter=node_converter
            )

            if save_htree:
                max_htree_construction_time = max(
                    max_htree_construction_time, htree_time
                )
                htree_construction_time += htree_time

            data.clear_dsg()
            data_list.append(data)

    print(f"Number of node features: {data.num_node_features()}")
    if save_htree:
        stats = f"{htree_construction_time: 0.2f}. (max: {max_htree_construction_time})"
        print(f"Total h-tree construction time: {stats}")

    with output_path.open("wb") as output_file:
        pickle.dump(data_list, output_file)

    # save dataset stat: room connectivity threshold (if applicable); label mapping
    data_stat_dir = output_path.parent / f"{output_path.stem}_stat"
    if data_stat_dir.exists():
        shutil.rmtree(data_stat_dir)

    data_stat_dir.mkdir(parents=True)
    print(data_stat_dir)
    label_dict = data.get_label_dict()

    with (data_stat_dir / param_filename).open("w") as output_file:
        if from_raw_data:
            output_params = dict(
                threshold_near=threshold_near, max_near=max_near, max_on=max_on
            )
            yaml.dump(output_params, output_file, default_flow_style=False)
        yaml.dump(label_dict, output_file, default_flow_style=False)


if __name__ == "__main__":
    main()
