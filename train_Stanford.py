from hydra_gnn.utils import PROJECT_DIR, STANFORD3DSG_GRAPH_DIR, \
    plist, pgrid, update_existing_keys
from hydra_gnn.Stanford3DSG_dataset import Stanford3DSG_htree_data, Stanford3DSG_data, \
    Stanford3DSG_dataset
from hydra_gnn.semisupervised_training_job import SemiSupervisedTrainingJob
from hydra_gnn.preprocess_dsgs import dsg_node_converter
from statistics import mean, stdev
import os
import shutil
import argparse
import pickle
import yaml
import torch
import numpy as np
import pandas as pd
from pprint import pprint


# parameter sweep setup (note: last GAT hidden_dims is label dim and therefore not specified)
GAT_hidden_dims = plist('GAT_hidden_dims', [[128], [128, 128], [128, 128, 128]])
dropout = plist('dropout', [0.25, 0.5])
lr = plist('lr', [0.001])
weight_decay = plist('weight_decay', [0.0])
param_dict_list = pgrid(lr, weight_decay, GAT_hidden_dims, dropout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', default=0, type=int, help="slurm array task ID")
    parser.add_argument('--num_tasks', default=1, type=int, help="total number of slurm array tasks")
    parser.add_argument('--gpu_index', default=-1, type=int, help="gpu index (default: task_id)")
    parser.add_argument('--config_file', default=os.path.join(PROJECT_DIR, 'config/mp3d_default_config.yaml'),
                        help='training config file')
    parser.add_argument('--use_htree', action='store_true',
                        help='use htree instead of original graph for learning')
    args = parser.parse_args()

    print(f"cuda available: {torch.cuda.is_available()}")
    
    # parse config file
    with open(args.config_file, 'r') as input_file:
        config = yaml.safe_load(input_file)
    task_id = args.task_id
    num_tasks = args.num_tasks
    gpu_index = args.gpu_index if args.gpu_index != -1 else task_id
    dataset_path = os.path.join(PROJECT_DIR, config['data']['file_path'])
    output_dir = os.path.join(PROJECT_DIR, config['logger']['output_dir'])
    assert task_id < num_tasks

    # setup log folder, accuracy files
    print(f"output direcotry: {output_dir}")
    if os.path.exists(output_dir):
        print("Existing contents might be over-written.")
        # if task_id == 0:
        #     input("Output directory exists. Existing contents might be over-written. Press any key to proceed...")
    else:
        os.mkdir(output_dir)
    
    # create torch data lists from saved graph data
    with open(STANFORD3DSG_GRAPH_DIR, 'rb') as input_file:
        saved_data_list, semantic_dict, num_labels = pickle.load(input_file)
        room_label_dict = semantic_dict['room']
        object_label_dict = semantic_dict['object']
    
    object_feature_converter = lambda i: np.zeros(0)
    room_feature_converter = lambda i: np.zeros(0)
    node_converter = dsg_node_converter(object_feature_converter, room_feature_converter)
    data_list = []
    for i in range(len(saved_data_list['x_list'])):
        data_dict = {'x': saved_data_list['x_list'][i],
                     'y': saved_data_list['y_list'][i],
                     'edge_index': saved_data_list['edge_index_list'][i],
                     'room_mask': saved_data_list['room_mask_list'][i]}
        if args.use_htree:
            data = Stanford3DSG_htree_data(data_dict=data_dict, 
                                           room_semantic_dict=semantic_dict['room'], 
                                           object_semantic_dict=semantic_dict['object'])
        else:
            data = Stanford3DSG_data(data_dict=data_dict, 
                                     room_semantic_dict=semantic_dict['room'], 
                                     object_semantic_dict=semantic_dict['object'])
            
        # use hetero data first, compute_relative_pose() does not work with homogeneous data
        data.compute_torch_data(use_heterogeneous=True, node_converter=node_converter)
        
        if config['network']['conv_block'] == 'GAT_edge':
            data.compute_relative_pos()
        if config['data']['type'] == 'homogeneous':
            data.to_homogeneous()
        
        # set clique node features to zero to be consistent with NT paper
        if isinstance(data, Stanford3DSG_htree_data) and config['data']['type'] == 'homogeneous':
            data.get_torch_data().x[data.get_torch_data().clique_has == -1] = 0 
        data_list.append(data)

    # preprare dataset
    dataset = Stanford3DSG_dataset()
    for data in data_list:
        dataset.add_data(data)

    # log resutls
    log_params = list(param_dict_list[0].keys())
    df = pd.DataFrame(columns=['log_dir'] + log_params + \
        ['val_' + str(i) for i in range(config['run_control']['num_runs'])] + \
            ['test_' + str(i) for i in range(config['run_control']['num_runs'])] + \
                ['avg num_epochs', 'avg training_time/ecpho (s)', 'avg test_time (s)'])

    # update parameter
    num_param_set = len(param_dict_list)
    experiment_id_list = list(range(num_param_set))[task_id:num_param_set:num_tasks]
    for experiment_i in experiment_id_list:
        print(f"\nExperiment {experiment_i+1} / {len(param_dict_list)}")
        param_dict = param_dict_list[experiment_i]
        
        if config['network']['conv_block'][:3] == 'GAT':
            # todo: this is for temporary GAT tuning with single attention head
            param_dict['GAT_heads'] = len(param_dict['GAT_hidden_dims']) * [6] + [6]
            param_dict['GAT_concats'] = len(param_dict['GAT_hidden_dims']) * [True] + [True]

            # check GAT params
            if len(param_dict['GAT_heads']) != len(param_dict['GAT_hidden_dims']) + 1 or \
                len(param_dict['GAT_concats']) != len(param_dict['GAT_hidden_dims']) + 1:
                print('  Skip - invalid GAT param')
                continue

        update_existing_keys(config['network'], param_dict)
        update_existing_keys(config['optimization'], param_dict)
        pprint('config:')
        pprint(config)

        # clean up output directory
        experiment_output_dir_i = os.path.join(output_dir, f"experiment_{experiment_i}")
        if os.path.exists(experiment_output_dir_i):
            shutil.rmtree(experiment_output_dir_i)
            print(f"Overwritting experiment output folder {experiment_output_dir_i}")
        os.mkdir(experiment_output_dir_i)

        # save config
        with open(os.path.join(experiment_output_dir_i, 'config.yaml'), 'w') as output_file:
            yaml.dump(config, output_file, default_flow_style=False)

        # run experiment
        val_accuracy_list = []
        val_room_accuracy_list = []
        val_object_accuracy_list = []
        test_accuracy_list = []
        test_room_accuracy_list = []
        test_object_accuracy_list = []
        training_time_list = []
        training_epoch_list = []
        test_time_list = []
        for j in range(config['run_control']['num_runs']):
            print(f"\nRun {j + 1} / {config['run_control']['num_runs']}:")
            dataset.generate_node_split(0.7, 0.1, 0.2, seed=j)
            train_job = SemiSupervisedTrainingJob(dataset=dataset,
                                                  network_params=config['network'])
            model, best_acc, info = train_job.train(f"{experiment_output_dir_i}/{j}", 
                                            optimization_params=config['optimization'],
                                            early_stop_window=config['run_control']['early_stop_window'],
                                            gpu_index=gpu_index,
                                            verbose=True)
            val_accuracy_list.append(best_acc[0] * 100)
            test_accuracy_list.append(best_acc[1] * 100)
            training_time_list.append(info['training_time'])
            training_epoch_list.append(info['num_epochs'])
            test_time_list.append(info['test_time'])

        # print type separated accuracy
        val_room_accuracy, val_object_accuracy = train_job.test(mask_name='val_mask', get_type_separated_accuracy=True)
        test_room_accuracy, test_object_accuracy = train_job.test(mask_name='test_mask', get_type_separated_accuracy=True)
        val_room_accuracy_list.append(val_room_accuracy * 100)
        val_object_accuracy_list.append(val_object_accuracy * 100)
        test_room_accuracy_list.append(test_room_accuracy * 100)
        test_object_accuracy_list.append(test_object_accuracy * 100)
        print(f"\nValidation accuracy: {mean(val_accuracy_list)} +/- {stdev(val_accuracy_list)}")
        print(f"    room: {mean(val_room_accuracy_list)} +/- {stdev(val_room_accuracy_list)}")
        print(f"    object: {mean(val_object_accuracy_list)} +/- {stdev(val_object_accuracy_list)}")

        print(f"Test accuracy: {mean(test_accuracy_list)} +/- {stdev(test_accuracy_list)}")
        print(f"    room: {mean(test_room_accuracy_list)} +/- {stdev(test_room_accuracy_list)}")
        print(f"    object: {mean(test_object_accuracy_list)} +/- {stdev(test_object_accuracy_list)}")

        # save param and accuracy
        accuracy_file_path = os.path.join(output_dir, f"accuracy-{task_id}.csv")
        output_data_list = [f"experiment_{experiment_i}"] + [param_dict[key] for key in log_params] + \
            val_accuracy_list + test_accuracy_list + \
                [mean(training_epoch_list), sum(training_time_list) / sum(training_epoch_list), mean(test_time_list)]
        df = pd.concat([df, pd.DataFrame(data=[output_data_list], columns = df.columns)])
        df.to_csv(accuracy_file_path, index=False)
