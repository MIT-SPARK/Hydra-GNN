from hydra_gnn.utils import PROJECT_DIR, MP3D_BENCHMARK_DIR, HYDRA_TRAJ_DIR, \
    plist, pgrid, update_existing_keys
from hydra_gnn.mp3d_utils import generate_mp3d_split, read_mp3d_split
from hydra_gnn.mp3d_dataset import Hydra_mp3d_dataset
from hydra_gnn.base_training_job import BaseTrainingJob
import random
from statistics import mean
import os
import shutil
import argparse
import pickle
import yaml
import torch
import pandas as pd
from pprint import pprint


# parameter sweep setup
GAT_hidden_dims = plist('GAT_hidden_dims', [[32, 32, 32]])
dropout = plist('dropout', [0.2])
lr = plist('lr', [0.002])
weight_decay = plist('weight_decay', [0.0])
param_dict_list = pgrid(lr, weight_decay, GAT_hidden_dims, dropout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', default=0, type=int, help="slurm array task ID")
    parser.add_argument('--num_tasks', default=1, type=int, help="total number of slurm array tasks")
    parser.add_argument('--config_file', default=os.path.join(PROJECT_DIR, 'config/mp3d_default_config.yaml'),
                        help='training config file')
    parser.add_argument('--train_val_ratio', default=None, type=float, nargs=2, 
                        help='training and validation ratio')
    args = parser.parse_args()

    print(f"cuda available: {torch.cuda.is_available()}")
    
    # parse config file
    with open(args.config_file, 'r') as input_file:
        config = yaml.safe_load(input_file)
    task_id = args.task_id
    num_tasks = args.num_tasks
    dataset_path = os.path.join(PROJECT_DIR, config['data']['file_path'])
    output_dir = os.path.join(PROJECT_DIR, config['logger']['output_dir'])
    assert task_id < num_tasks

    # setup log folder, accuracy files
    print(f"output direcotry: {output_dir}")
    if os.path.exists(output_dir):
        if task_id == 0:
            input("Output directory exists. Existing contents might be over-written. Press any key to proceed...")
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

    # create data lists
    dataset_dict = {'train': Hydra_mp3d_dataset('train'),
                    'val': Hydra_mp3d_dataset('val'),
                    'test': Hydra_mp3d_dataset('test')}
    with open(dataset_path, 'rb') as input_file:
        data_list = pickle.load(input_file)
    if config['network']['conv_block'] == 'GAT_edge':
        assert config['data']['type'] != 'homogeneous'
        [data.compute_relative_pos() for data in data_list]
    elif config['data']['type'] == 'homogeneous':
        [data.to_homogeneous() for data in data_list]
    
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

    # update parameter
    num_param_set = len(param_dict_list)
    experiment_id_list = list(range(num_param_set))[task_id:num_param_set:num_tasks]
    for experiment_i in experiment_id_list:
        print(f"{experiment_i+1} / {len(param_dict_list)}")
        param_dict = param_dict_list[experiment_i]
        
        # todo: this is for temporary GAT tuning with single attention head
        if config['network']['conv_block'][:3] == 'GAT':
            param_dict['GAT_heads'] = len(param_dict['GAT_hidden_dims']) * [1] + [1]
            param_dict['GAT_concats'] = len(param_dict['GAT_hidden_dims']) * [False] + [False]

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
        os.mkdir(experiment_output_dir_i)

        accuracy_file_path = os.path.join(output_dir, f"accuracy-{task_id}.csv")
        if os.path.exists(accuracy_file_path):
            os.remove(accuracy_file_path)

        # save config
        with open(os.path.join(experiment_output_dir_i, 'config.yaml'), 'w') as output_file:
            yaml.dump(config, output_file, default_flow_style=False)

        # run experiment
        test_accuracy_list = []
        val_accuracy_list = []
        training_time_list = []
        training_epoch_list = []
        test_time_list = []
        for j in range(config['run_control']['num_runs']):
            train_job = BaseTrainingJob(network_type=config['data']['type'],
                                        dataset_dict=dataset_dict, 
                                        network_params=config['network'])
            model, best_acc, info = train_job.train(experiment_output_dir_i + '/' + str(j), 
                                            optimization_params=config['optimization'],
                                            early_stop_window=config['run_control']['early_stop_window'], 
                                            verbose=True)

            val_accuracy_list.append(best_acc[0] * 100)
            test_accuracy_list.append(best_acc[1] * 100)
            training_time_list.append(info['training_time'])
            training_epoch_list.append(info['num_epochs'])
            test_time_list.append(info['test_time'])

        # save param and accuracy
        output_data_list = [f"experiment_{experiment_i}"] + [param_dict[key] for key in log_params] + \
            val_accuracy_list + test_accuracy_list + \
                [mean(training_epoch_list), sum(training_time_list) / sum(training_epoch_list), mean(test_time_list)]
        df = pd.concat([df, pd.DataFrame(data=[output_data_list], columns = df.columns)])
        df.to_csv(accuracy_file_path, index=False)
