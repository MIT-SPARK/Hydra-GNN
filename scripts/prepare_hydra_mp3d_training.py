"""Script to prepare mp3d data."""
from hydra_gnn.utils import (
    project_dir,
    HYDRA_TRAJ_DIR,
    MP3D_HOUSE_DIR,
    COLORMAP_DATA_PATH,
    WORD2VEC_MODEL_PATH,
)
from hydra_gnn.mp3d_dataset import Hydra_mp3d_data, Hydra_mp3d_htree_data
from hydra_gnn.preprocess_dsgs import hydra_object_feature_converter, dsg_node_converter
from spark_dsg.mp3d import load_mp3d_info
import spark_dsg as dsg
import shutil
import click
import numpy as np
import gensim
import pandas as pd
import pathlib
import pickle
import yaml


# data params
threshold_near = 1.5
max_near = 2.0
max_on = 0.2
object_synonyms = []
room_synonyms = [("a", "t"), ("z", "Z", "x", "p", "\x15")]


def room_removal_func(room):
    """Return whether or not to remove rooms."""
    return not (len(room.children()) > 1 or room.has_siblings())


def _empty_feature(x):
    return np.zeros(0)


def _empty_w2v(x):
    return np.zeros(300)


@click.command()
@click.option("-n", "--output_filename", default=None, help="output file name")
@click.option(
    "--min_iou",
    default=0.6,
    type=float,
    help="minimum IoU threshold for room label assignment",
)
@click.option(
    "--output_dir",
    default=str(project_dir() / "output/preprocessed_mp3d"),
    help="training and validation ratio",
)
@click.option(
    "--expand_rooms",
    is_flag=True,
    help="expand room segmentation from existing places segmentation",
)
@click.option(
    "--repartition_rooms",
    is_flag=True,
    help="re-segment rooms using gt segmentation in the dataset file",
)
@click.option("--save_htree", is_flag=True, help="store htree data")
@click.option("--save_homogeneous", is_flag=True, help="store torch data as HeteroData")
def main(
    output_filename,
    min_iou,
    output_dir,
    expand_rooms,
    repartition_rooms,
    save_htree,
    save_homogeneous,
):
    """Prepare an mp3d dataset."""
    param_filename = "params.yaml"
    skipped_filename = "skipped_partial_scenes.yaml"
    if output_filename is None:
        output_prefix = "htree" if save_htree else "data"
        output_filename = "{}_gt{}.pkl".format(output_prefix, int(min_iou*100)) \
            if repartition_rooms else "{}_{}.pkl".format(output_prefix, int(min_iou*100))

    print(f"Saving torch graphs as htree:  {save_htree}")
    print(f"Saving torch graphs as homogeneous torch data: {save_homogeneous}")
    print(f"Saving torch graphs with expand_rooms: {expand_rooms}")
    print(f"Saving torch graphs with room repartioning: {repartition_rooms}")
    print(f"Min IoU threshold for room label assignment: {min_iou}")
    print(f"Output directory: {output_dir}")
    print(f"Output files: {output_filename}, ({param_filename}, {skipped_filename})")

    output_path = pathlib.Path(output_dir).expanduser().absolute() / output_filename
    if output_path.exists():
        input("Output data file exists. Press any key to proceed...")

    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    colormap_data = pd.read_csv(COLORMAP_DATA_PATH, delimiter=",")
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
        WORD2VEC_MODEL_PATH, binary=True
    )
    object_feature_converter = hydra_object_feature_converter(
        colormap_data, word2vec_model
    )
    if save_htree or save_homogeneous:
        room_feature_converter = _empty_w2v
    else:
        room_feature_converter = _empty_feature

    trajectory_dirs = [x.name for x in pathlib.Path(HYDRA_TRAJ_DIR).iterdir()]
    skipped_json_files = {"none": [], "no room": [], "no object": []}
    data_list = []
    htree_construction_time = 0.0
    max_htree_construction_time = 0.0
    for i, trajectory_name in enumerate(trajectory_dirs):
        trajectory_dir = pathlib.Path(HYDRA_TRAJ_DIR) / trajectory_name
        scene_id, _, trajectory_id = trajectory_name.split("_")
        # Load gt house segmentation for room labeling
        gt_house_file = f"{MP3D_HOUSE_DIR}/{scene_id}.house"
        gt_house_info = load_mp3d_info(gt_house_file)

        json_file_names = [x.name for x in trajectory_dir.iterdir()]
        for json_file_name in json_file_names:
            if json_file_name[0:3] == "est":
                continue

            num_frames = json_file_name[15:-5]
            file_path = trajectory_dir / json_file_name
            assert file_path.exists()

            if save_htree:
                data = Hydra_mp3d_htree_data(
                    scene_id=scene_id,
                    trajectory_id=trajectory_id,
                    num_frames=num_frames,
                    file_path=str(file_path),
                    expand_rooms=expand_rooms,
                )
            else:
                data = Hydra_mp3d_data(
                    scene_id=scene_id,
                    trajectory_id=trajectory_id,
                    num_frames=num_frames,
                    file_path=str(file_path),
                    expand_rooms=expand_rooms,
                )

            curr_traj_path = str(pathlib.Path(trajectory_name) / json_file_name)
            # skip dsg without room node or without object node
            if data.get_room_object_dsg().num_nodes() == 0:
                skipped_json_files["none"].append(curr_traj_path)
                continue

            G_curr = data.get_room_object_dsg()
            if G_curr.get_layer(dsg.DsgLayers.ROOMS).num_nodes() == 0:
                skipped_json_files["no room"].append(curr_traj_path)
                continue

            if G_curr.get_layer(dsg.DsgLayers.OBJECTS).num_nodes() == 0:
                skipped_json_files["no object"].append(curr_traj_path)
                continue

            # parepare torch data
            data.add_dsg_room_labels(
                gt_house_info,
                angle_deg=-90,
                room_removal_func=room_removal_func,
                min_iou_threshold=min_iou,
                repartition_rooms=repartition_rooms,
            )

            num_objects = (
                data.get_room_object_dsg().get_layer(dsg.DsgLayers.OBJECTS).num_nodes()
            )
            if repartition_rooms and num_objects == 0:
                skipped_json_files["no object"].append(curr_traj_path)
                continue

            data.add_object_edges(
                threshold_near=threshold_near, max_near=max_near, max_on=max_on
            )
            htree_time = data.compute_torch_data(
                use_heterogeneous=(not save_homogeneous),
                node_converter=dsg_node_converter(
                    object_feature_converter, room_feature_converter
                ),
                object_synonyms=object_synonyms,
                room_synonyms=room_synonyms,
            )
            if save_htree:
                max_htree_construction_time = max(
                    max_htree_construction_time, htree_time
                )
                htree_construction_time += htree_time

            data.clear_dsg()  # remove hydra dsg for output
            data_list.append(data)

        if i == 0:
            print(f"Number of node features: {data.num_node_features()}")
        print(f"Done converting {i + 1}/{len(trajectory_dirs)} trajectories. ")

    if save_htree:
        print(
            f"Total h-tree construction time: {htree_construction_time: 0.2f}. (max: {max_htree_construction_time})"
        )

    with output_path.open("wb") as output_file:
        pickle.dump(data_list, output_file)
    print(
        f"Saved {len(data_list)} scene graphs with at least one room and one object to {output_path}."
    )

    # save dataset stat
    data_stat_dir = output_path.parent / f"{output_path.stem}_stat"
    if data_stat_dir.exists():
        shutil.rmtree(data_stat_dir)

    data_stat_dir.mkdir(parents=True)
    # save room connectivity threshold and label mapping
    output_params = dict(
        threshold_near=threshold_near, max_near=max_near, max_on=max_on
    )
    label_dict = data.get_label_dict()

    with (data_stat_dir / param_filename).open("w") as output_file:
        yaml.dump(output_params, output_file, default_flow_style=False)
        yaml.dump(label_dict, output_file, default_flow_style=False)

    # save skipped trajectories
    with (data_stat_dir / skipped_filename).open("w") as output_file:
        yaml.dump(skipped_json_files, output_file, default_flow_style=False)


if __name__ == "__main__":
    main()
