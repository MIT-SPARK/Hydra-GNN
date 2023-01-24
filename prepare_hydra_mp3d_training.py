from hydra_gnn.utils import PROJECT_DIR, HYDRA_TRAJ_DIR, MP3D_HOUSE_DIR, COLORMAP_DATA_PATH, WORD2VEC_MODEL_PATH
from hydra_gnn.mp3d_dataset import Hydra_mp3d_data, Hydra_mp3d_htree_data
from hydra_gnn.preprocess_dsgs import hydra_object_feature_converter, hydra_node_converter
from spark_dsg.mp3d import load_mp3d_info
import spark_dsg as dsg
import os
import argparse
import numpy as np
import gensim
import pandas as pd
import pickle
import yaml


# data params
threshold_near=2.0
threshold_on=1.0
max_near=2.0
object_synonyms=[]
room_synonyms=[('a', 't'), ('z', 'Z', 'x', 'p', '\x15')]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('output_filename', default='data.pkl',
                        help="output file name")
    parser.add_argument('--output_dir', default=os.path.join(PROJECT_DIR, 'output/preprocessed_mp3d'),
                        help="training and validation ratio")
    parser.add_argument('--save_htree', action='store_true', 
                        help="store htree data")
    parser.add_argument('--save_homogeneous', action='store_true', 
                        help="store torch data as HeteroData")
    args = parser.parse_args()
    print("Saving torch graphs as htree:", args.save_htree)
    print("Saving torch graphs as homogeneous torch data:", args.save_homogeneous)
    print("Output directory:", args.output_dir)
    print("Output data files:", args.output_filename, ', params.yaml', ', skipped_partial_scenes.yaml')
    input("Press any key to proceed...")

    colormap_data = pd.read_csv(COLORMAP_DATA_PATH, delimiter=',')
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(WORD2VEC_MODEL_PATH, binary=True)
    object_feature_converter=hydra_object_feature_converter(colormap_data, word2vec_model)
    room_feature_converter = lambda i: np.zeros(300)

    trajectory_dirs = os.listdir(HYDRA_TRAJ_DIR)
    skipped_json_files = {'none': [], 'no room': [], 'no object': []}
    data_list = []
    for i, trajectory_name in enumerate(trajectory_dirs):
        trajectory_dir = os.path.join(HYDRA_TRAJ_DIR, trajectory_name)
        scene_id, _, trajectory_id = trajectory_name.split('_')
        # Load gt house segmentation for room labeling
        gt_house_file = f"{MP3D_HOUSE_DIR}/{scene_id}.house"
        gt_house_info = load_mp3d_info(gt_house_file)
        
        json_file_names = os.listdir(trajectory_dir)
        for json_file_name in json_file_names:
            if json_file_name[0:3] == 'est':
                continue
            num_frames = json_file_name[15:-5]
            file_path = os.path.join(HYDRA_TRAJ_DIR, trajectory_name, json_file_name)
            assert os.path.exists(file_path)

            if args.save_htree:
                data = Hydra_mp3d_htree_data(scene_id=scene_id, trajectory_id=trajectory_id, \
                    num_frames=num_frames, file_path=file_path)
            else:
            data = Hydra_mp3d_data(scene_id=scene_id, trajectory_id=trajectory_id, \
                num_frames=num_frames, file_path=file_path)

            # skip dsg without room node or without object node
            if data.get_room_object_dsg().num_nodes() == 0:
                skipped_json_files['none'].append(os.path.join(trajectory_name, json_file_name))
                continue
            if data.get_room_object_dsg().get_layer(dsg.DsgLayers.ROOMS).num_nodes() == 0:
                skipped_json_files['no room'].append(os.path.join(trajectory_name, json_file_name))
                continue
            if data.get_room_object_dsg().get_layer(dsg.DsgLayers.OBJECTS).num_nodes() == 0:
                skipped_json_files['no object'].append(os.path.join(trajectory_name, json_file_name))
                continue

            # parepare torch data
            data.add_room_labels(gt_house_info, angle_deg=-90)
            data.add_object_edges(
                threshold_near=threshold_near, threshold_on=threshold_on, max_near=max_near)
            data.compute_torch_data(
                use_heterogeneous=(not args.save_homogeneous),
                node_converter=hydra_node_converter(object_feature_converter, room_feature_converter),
                object_synonyms=object_synonyms, 
                room_synonyms=room_synonyms)
            data.clear_dsg()    # remove hydra dsg for output
            data_list.append(data)
        print(f"Done converting {i + 1}/{len(trajectory_dirs)} trajectories.")
    
    output_filename = args.output_filename
    with open(os.path.join(args.output_dir, output_filename), 'wb') as output_file:
        pickle.dump(data_list, output_file)

    param_filename = 'params.yaml'
    output_params = dict(threshold_near=threshold_near, threshold_on=threshold_on, max_near=max_near)
    label_dict = data.get_label_dict()
    with open(os.path.join(args.output_dir, param_filename), 'w') as output_file:
        yaml.dump(output_params, output_file, default_flow_style=False)
        yaml.dump(label_dict, output_file, default_flow_style=False)

    skipped_filename = 'skipped_partial_scenes.yaml'
    with open(os.path.join(args.output_dir, skipped_filename), 'w') as output_file:
        yaml.dump(skipped_json_files, output_file, default_flow_style=False)
