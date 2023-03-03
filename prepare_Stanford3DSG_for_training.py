from hydra_gnn.utils import PROJECT_DIR, WORD2VEC_MODEL_PATH, STANFORD3DSG_DATA_DIR
from hydra_gnn.Stanford3DSG_dataset import Stanford3DSG_data, Stanford3DSG_htree_data, \
    Stanford3DSG_object_feature_converter, Stanford3DSG_room_feature_converter
from hydra_gnn.preprocess_dsgs import hydra_node_converter
import os
import argparse
import numpy as np
import gensim
import pickle
import yaml


# data params
threshold_near=1.5
max_near=2.0
max_on=0.2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('output_filename', default='data.pkl',
                        help="output file name")
    parser.add_argument('--output_dir', default=os.path.join(PROJECT_DIR, 'output/preprocessed_Stanford3DSG'),
                        help="training and validation ratio")
    parser.add_argument('--save_htree', action='store_true', 
                        help="store htree data")
    parser.add_argument('--save_homogeneous', action='store_true', 
                        help="store torch data as HeteroData")
    parser.add_argument('--save_word2vec', action='store_true', 
                        help="store word2vec vectors as node features")
    args = parser.parse_args()

    param_filename = 'params.yaml'
    skipped_filename = 'skipped_partial_scenes.yaml'
    
    print("Saving torch graphs as htree:", args.save_htree)
    print("Saving torch graphs as homogeneous torch data:", args.save_homogeneous)
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

    data_files = os.listdir(STANFORD3DSG_DATA_DIR)
    data_list = []
    for i, data_file in enumerate(data_files):

        if args.save_htree:
            data = Stanford3DSG_htree_data(os.path.join(STANFORD3DSG_DATA_DIR, data_file))
        
        else:
            data = Stanford3DSG_data(os.path.join(STANFORD3DSG_DATA_DIR, data_file))

        # parepare torch data
        data.add_object_edges(threshold_near=threshold_near, max_near=max_near, max_on=max_on)
        data.compute_torch_data(
            use_heterogeneous=(not args.save_homogeneous),
            node_converter=hydra_node_converter(object_feature_converter, room_feature_converter))
        data.clear_dsg()    # remove hydra dsg for output
        data_list.append(data)
        
        if i == 0:
            print(f"Number of node features: {data.num_node_features()}")
        print(f"Done converting {i + 1}/{len(data_files)} data.")

    with open(os.path.join(args.output_dir, args.output_filename), 'wb') as output_file:
        pickle.dump(data_list, output_file)

    # save dataset stat
    data_stat_dir = os.path.join(args.output_dir, os.path.splitext(args.output_filename)[0] + '_stat')
    if os.path.exists(data_stat_dir):
        os.rmdir(data_stat_dir)
    else:
        os.mkdir(data_stat_dir)
        print(data_stat_dir)
    # save room connectivity threshold and label mapping
    output_params = dict(threshold_near=threshold_near, max_near=max_near, max_on=max_on)
    label_dict = data.get_label_dict()
    with open(os.path.join(data_stat_dir, param_filename), 'w') as output_file:
        yaml.dump(output_params, output_file, default_flow_style=False)
        yaml.dump(label_dict, output_file, default_flow_style=False)
