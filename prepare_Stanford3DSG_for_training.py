from hydra_gnn.utils import PROJECT_DIR, WORD2VEC_MODEL_PATH, STANFORD3DSG_DATA_DIR, STANFORD3DSG_GRAPH_PATH
from hydra_gnn.Stanford3DSG_dataset import Stanford3DSG_data, Stanford3DSG_htree_data, \
    Stanford3DSG_object_feature_converter, Stanford3DSG_room_feature_converter
from hydra_gnn.preprocess_dsgs import dsg_node_converter
import os
import shutil
import argparse
import numpy as np
import gensim
import pickle
import yaml


# data params (used only with --from_raw_data flag)
threshold_near=1.5
max_near=2.0
max_on=0.2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('output_filename', default='data.pkl',
                        help="output file name")
    parser.add_argument('--output_dir', default=os.path.join(PROJECT_DIR, 'output/preprocessed_Stanford3DSG'),
                        help="output data directory")
    parser.add_argument('--from_raw_data', action='store_true',
                        help="use raw Stanford3DSG to construct training graphs instead of pre-saved graphs")
    parser.add_argument('--save_htree', action='store_true', 
                        help="store htree data")
    parser.add_argument('--save_word2vec', action='store_true', 
                        help="store word2vec vectors as node features")
    args = parser.parse_args()

    param_filename = 'params.yaml'
    skipped_filename = 'skipped_partial_scenes.yaml'
    
    print("Computing torch graphs from raw Stanford3DSG data:", args.from_raw_data)
    print("Saving torch graphs as htree:", args.save_htree)
    print("Saving torch graphs with word2vec features:", args.save_word2vec)
    print("Output directory:", args.output_dir) 
    print("Output data files:", f"{args.output_filename}, ({param_filename})")
    if os.path.exists(os.path.join(args.output_dir, args.output_filename)):
        input("Output data file exists. Press any key to proceed...")

    if args.save_word2vec:
        word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(WORD2VEC_MODEL_PATH, binary=True)
        object_feature_converter = Stanford3DSG_object_feature_converter(word2vec_model)
        room_feature_converter = Stanford3DSG_room_feature_converter(word2vec_model)
    else:
        object_feature_converter = lambda i: np.zeros(0)
        room_feature_converter = lambda i: np.zeros(0)
    node_converter = dsg_node_converter(object_feature_converter, room_feature_converter)

    # process dataset as a list of torch data
    data_list = []
    htree_construction_time = 0.0
    max_htree_construction_time = 0.0
    if args.from_raw_data:
        data_files = os.listdir(STANFORD3DSG_DATA_DIR)
        for i, data_file in enumerate(data_files):
            if args.save_htree:
                data = Stanford3DSG_htree_data(os.path.join(STANFORD3DSG_DATA_DIR, data_file))
            
            else:
                data = Stanford3DSG_data(os.path.join(STANFORD3DSG_DATA_DIR, data_file))
            data.add_object_edges(threshold_near=threshold_near, max_near=max_near, max_on=max_on)
            data.compute_torch_data(use_heterogeneous=True, node_converter=node_converter)
            data.clear_dsg()
            data_list.append(data)
    else:
        with open(STANFORD3DSG_GRAPH_PATH, 'rb') as input_file:
            saved_data_list, semantic_dict, num_labels = pickle.load(input_file)
            room_label_dict = semantic_dict['room']
            object_label_dict = semantic_dict['object']
        for i in range(len(saved_data_list['x_list'])):
            data_dict = {'x': saved_data_list['x_list'][i],
                         'y': saved_data_list['y_list'][i],
                         'edge_index': saved_data_list['edge_index_list'][i],
                         'room_mask': saved_data_list['room_mask_list'][i]}
            if args.save_htree:
                data = Stanford3DSG_htree_data(data_dict=data_dict, 
                                               room_semantic_dict=semantic_dict['room'], 
                                               object_semantic_dict=semantic_dict['object'])
            else:
                data = Stanford3DSG_data(data_dict=data_dict, 
                                         room_semantic_dict=semantic_dict['room'], 
                                         object_semantic_dict=semantic_dict['object'])
            htree_time = data.compute_torch_data(use_heterogeneous=True, node_converter=node_converter)
            if args.save_htree:
                max_htree_construction_time = max(max_htree_construction_time, htree_time)
                htree_construction_time += htree_time
            
            data.clear_dsg()
            data_list.append(data)

    print(f"Number of node features: {data.num_node_features()}")
    if args.save_htree:
        print(f"Totla h-tree construction time: {htree_construction_time: 0.2f}. (max: {max_htree_construction_time})")

    with open(os.path.join(args.output_dir, args.output_filename), 'wb') as output_file:
        pickle.dump(data_list, output_file)

    # save dataset stat: room connectivity threshold (if applicable); label mapping
    data_stat_dir = os.path.join(args.output_dir, os.path.splitext(args.output_filename)[0] + '_stat')
    if os.path.exists(data_stat_dir):
        shutil.rmtree(data_stat_dir)
    else:
        os.mkdir(data_stat_dir)
        print(data_stat_dir)
    label_dict = data.get_label_dict()
    with open(os.path.join(data_stat_dir, param_filename), 'w') as output_file:
        if args.from_raw_data:
            output_params = dict(threshold_near=threshold_near, max_near=max_near, max_on=max_on)
            yaml.dump(output_params, output_file, default_flow_style=False)
        yaml.dump(label_dict, output_file, default_flow_style=False)
