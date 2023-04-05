from hydra_gnn.utils import PROJECT_DIR, MP3D_BENCHMARK_DIR, HYDRA_TRAJ_DIR, \
    plist, pgrid, update_existing_keys
from hydra_gnn.mp3d_utils import generate_mp3d_split, read_mp3d_split
from hydra_gnn.mp3d_dataset import Hydra_mp3d_dataset
from hydra_gnn.base_training_job import BaseTrainingJob
from torch_geometric.loader import DataLoader
import random
from statistics import mean
import os
import shutil
import argparse
import pickle
import yaml
import torch
import numpy as np
import pandas as pd
from pprint import pprint


# parameter sweep setup
GAT_hidden_dims = plist('GAT_hidden_dims', [[32, 32], [64, 64]])
GAT_head = plist('GAT_head', [3])    # same number of head in each conv layer
GAT_concat = plist('GAT_concat', [True, False])    # same concate in each conv layer except the last (always False)
dropout = plist('dropout', [0.4])
lr = plist('lr', [0.001])
weight_decay = plist('weight_decay', [0.0001, 0.001])
param_dict_list = pgrid(lr, weight_decay, GAT_hidden_dims, GAT_head, GAT_concat, dropout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', default=0, type=int, help="slurm array task ID")
    parser.add_argument('--num_tasks', default=1, type=int, help="total number of slurm array tasks")
    parser.add_argument('--gpu_index', default=-1, type=int, help="gpu index (default: task_id)")
    parser.add_argument('--config_file', default=os.path.join(PROJECT_DIR, 'config/mp3d_default_config.yaml'),
                        help='training config file')
    parser.add_argument('--train_val_ratio', default=None, type=float, nargs=2, 
                        help='training and validation ratio')
    parser.add_argument('--same_val_test', action='store_true', 
                        help="use the same scenes for validation and testing (trajctory 0-1 for validation, 2-4 for testing)")
    parser.add_argument('--remove_word2vec', action='store_true', 
                        help="remove word2vec features from node features (assuming it is at the end of the feature vector)")
    parser.add_argument('--single_experiment', action='store_true',
                        help="run single experiment using config file instead of parameter sweep")
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
    
    # load data and data split
    if args.train_val_ratio is None:
        print("Preparing training data using default mp3d split.")
        split_dict = read_mp3d_split(MP3D_BENCHMARK_DIR)
    else:
        print("Preparing training data using specified split ratio.")
        random.seed(0)
        split_dict = generate_mp3d_split(HYDRA_TRAJ_DIR, args.train_val_ratio[0], args.train_val_ratio[1])
    if args.same_val_test:
        print("Regenerating val/test split using trajectories -- 0-1 for training and 2-4 for testing.")

    # create data lists
    dataset_dict = {'train': Hydra_mp3d_dataset('train', remove_short_trajectories=False),
                    'val': Hydra_mp3d_dataset('val'),
                    'test': Hydra_mp3d_dataset('test')}
    with open(dataset_path, 'rb') as input_file:
        data_list = pickle.load(input_file)
    if config['network']['conv_block'] == 'GAT_edge':
        [data.compute_relative_pos() for data in data_list]
    if config['data']['type'] == 'homogeneous':
        [data.to_homogeneous() for data in data_list]
    if args.remove_word2vec:
        [data.remove_last_features(300) for data in data_list]
    
    if args.same_val_test:
        for data in data_list:
            if data.get_data_info()['scene_id'] in split_dict['scenes_train']:
                dataset_dict['train'].add_data(data)
            else:
                if data.get_data_info()['trajectory_id'] in ['0', '1']:
                    dataset_dict['val'].add_data(data)
                elif data.get_data_info()['trajectory_id'] in ['2', '3', '4']:
                    dataset_dict['test'].add_data(data)
                else:
                    raise RuntimeError(f"Found invalid trajectory id in input data file {dataset_path}")
    else:
        for data in data_list:
            if data.get_data_info()['scene_id'] in split_dict['scenes_train']:
                dataset_dict['train'].add_data(data)
            elif data.get_data_info()['scene_id'] in split_dict['scenes_val']:
                dataset_dict['val'].add_data(data)
            elif data.get_data_info()['scene_id'] in split_dict['scenes_test']:
                dataset_dict['test'].add_data(data)
            else:
                raise RuntimeError(f"Founnd invalid scene id in input data file {dataset_path}.")
    print(f"  training: {dataset_dict['train'].num_scenes()} scenes {len(dataset_dict['train'])} graphs\n"
          f"  validation: {dataset_dict['val'].num_scenes()} scenes {len(dataset_dict['val'])} graphs\n"
          f"  test: {dataset_dict['test'].num_scenes()} scenes {len(dataset_dict['test'])} graphs")

    # master output directory 
    # dt_str = datetime.datetime.now().strftime('%m%d%H%M')
    # experiment_output_dir = os.path.join(output_dir, dt_str)
    # os.mkdir(experiment_output_dir)

    # log resutls
    log_params = list(param_dict_list[0].keys())
    df = pd.DataFrame(columns=['log_dir'] + log_params + \
        ['val_' + str(i) for i in range(config['run_control']['num_runs'])] + \
            ['test_' + str(i) for i in range(config['run_control']['num_runs'])] + \
                ['avg num_epochs', 'avg training_time/ecpho (s)', 'avg test_time (s)'])

    # run experiment
    if args.single_experiment:
        experiment_id_list = [0]
    else:
        num_param_set = len(param_dict_list)
        experiment_id_list = list(range(num_param_set))[task_id:num_param_set:num_tasks]

    for experiment_i in experiment_id_list:
        if args.single_experiment:
            print(f"\nUse all parameters in config: {args.config_file}")
            experiment_output_dir_i = os.path.join(output_dir, "experiment")
            config_output_path_i = os.path.join(output_dir, "config.yaml")
            accuracy_file_path = os.path.join(output_dir, "accuracy.csv")
            param_list = ['-'] * len(log_params)
        else:
            print(f"\nExperiment {experiment_i+1} / {len(param_dict_list)}")
            experiment_output_dir_i = os.path.join(output_dir, f"experiment_{experiment_i}")
            config_output_path_i = os.path.join(experiment_output_dir_i, 'config.yaml')
            accuracy_file_path = os.path.join(output_dir, f"accuracy-{task_id}.csv")
            param_dict = param_dict_list[experiment_i]
            param_list = [param_dict[key] for key in log_params]
            if config['network']['conv_block'][:3] == 'GAT':
                # todo: this is for temporary GAT tuning
                param_dict['GAT_heads'] = len(param_dict['GAT_hidden_dims']) * [param_dict['GAT_head']] + [param_dict['GAT_head']]
                param_dict['GAT_concats'] = len(param_dict['GAT_hidden_dims']) * [param_dict['GAT_concat']] + [False]

                # check GAT params
                if len(param_dict['GAT_heads']) != len(param_dict['GAT_hidden_dims']) + 1 or \
                    len(param_dict['GAT_concats']) != len(param_dict['GAT_hidden_dims']) + 1:
                    print('  Skip - invalid GAT param')
                    continue
            update_existing_keys(config['network'], param_dict)
            update_existing_keys(config['optimization'], param_dict)
            pprint('config:')
            pprint(config)

        # clean up tensorboard output directory
        if os.path.exists(experiment_output_dir_i):
            shutil.rmtree(experiment_output_dir_i)
            print(f"Overwritting experiment output folder {experiment_output_dir_i}")
        os.mkdir(experiment_output_dir_i)

        # save config
        with open(config_output_path_i, 'w') as output_file:
            yaml.dump(config, output_file, default_flow_style=False)

        # run experiment
        test_accuracy_list = []
        val_accuracy_list = []
        training_time_list = []
        training_epoch_list = []
        test_time_list = []
        for j in range(config['run_control']['num_runs']):
            print(f"\nRun {j + 1} / {config['run_control']['num_runs']}:")
            train_job = BaseTrainingJob(dataset_dict=dataset_dict, 
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

            # log per label accuracy for single experiment
            if args.single_experiment:
                train_accuracy, train_accuracy_matrix = \
                    train_job.test(DataLoader(train_job.get_dataset('train'), batch_size=4096), True)
                val_accuracy, val_accuracy_matrix = \
                    train_job.test(DataLoader(train_job.get_dataset('val'), batch_size=4096), True)
                test_accuracy, test_accuracy_matrix = \
                    train_job.test(DataLoader(train_job.get_dataset('test'), batch_size=4096), True)
                assert abs(val_accuracy - best_acc[0]) < 0.001, f"{val_accuracy}, {best_acc[0]}"
                assert abs(test_accuracy - best_acc[1]) < 0.001, f"{test_accuracy}, {best_acc[1]}"
                # output_header = "num_labels[train],num_tp[train],num_fp[train],num_labels[val],num_tp[val],num_fp[val],num_labels[test],num_tp[test],num_fp[test]"
                num_labels = train_accuracy_matrix.shape[1]
                output_header = [f"{l}[train]" for l in range(num_labels)] + \
                                [f"{l}[val]" for l in range(num_labels)] + \
                                [f"{l}[test]" for l in range(num_labels)]
                output_header = ','.join(output_header)
                np.savetxt(f"{experiment_output_dir_i}/{j}/accuracy_matrix.csv", 
                        np.concatenate((train_accuracy_matrix, val_accuracy_matrix, test_accuracy_matrix), axis=1),
                        fmt='%i',
                        delimiter=',', 
                        comments='',
                        header=output_header)

        # save param and accuracy
        output_data_list = [f"experiment_{experiment_i}"] + param_list + val_accuracy_list + test_accuracy_list + \
            [mean(training_epoch_list), sum(training_time_list) / sum(training_epoch_list), mean(test_time_list)]
        df = pd.concat([df, pd.DataFrame(data=[output_data_list], columns = df.columns)])
        if not args.single_experiment:
            df["GAT_concat"] = df["GAT_concat"].astype(bool)
        df.to_csv(accuracy_file_path, index=False)
