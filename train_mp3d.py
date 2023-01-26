from hydra_gnn.utils import PROJECT_DIR, MP3D_BENCHMARK_DIR, HYDRA_TRAJ_DIR, \
    plist, pgrid, update_existing_keys
from hydra_gnn.mp3d_utils import generate_mp3d_split, read_mp3d_split
from hydra_gnn.mp3d_dataset import Hydra_mp3d_dataset
from hydra_gnn.base_training_job import BaseTrainingJob
import datetime
import random
import os
import argparse
import pickle
import yaml
import pandas as pd
from pprint import pprint


# parameter sweep setup
GAT_hidden_dims = plist('GAT_hidden_dims', [[16], [32], [16, 16], [32, 32]])
dropout = plist('dropout', [0.2, 0.4, 0.6])
lr = plist('lr', [0.001])
weight_decay = plist('weight_decay', [0.0])
param_dict_list = pgrid(lr, weight_decay, GAT_hidden_dims, dropout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default=os.path.join(PROJECT_DIR, 'config/mp3d_default_config.yaml'),
                        help='traiing config file')
    parser.add_argument('--train_val_ratio', default=None, type=float, nargs=2, 
                        help='training and validation ratio')
    args = parser.parse_args()
    
    # parse config file
    with open(args.config_file, 'r') as input_file:
        config = yaml.safe_load(input_file)
    dataset_path = os.path.join(PROJECT_DIR, config['data']['file_path'])
    output_dir = os.path.join(PROJECT_DIR, config['logger']['output_dir'])

    # setup log folder, accuracy files
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # shutil.copy(args.config_file, output_dir)
    
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
    dt_str = datetime.datetime.now().strftime('%m%d%H%M')
    experiment_output_dir = os.path.join(output_dir, dt_str)
    os.mkdir(experiment_output_dir)
    print(f"Saving output to {experiment_output_dir}")

    # log resutls
    log_params = list(param_dict_list[0].keys())
    df = pd.DataFrame(columns=['log_dir'] + log_params + \
        ['val_' + str(i) for i in range(config['run_control']['num_runs'])] + \
            ['test_' + str(i) for i in range(config['run_control']['num_runs'])])

    # update parameter
    for experiment_i, param_dict in enumerate(param_dict_list):
        print(f"{experiment_i+1} / {len(param_dict_list)}")
        
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

        # save config
        experiment_output_dir_i = os.path.join(experiment_output_dir, f"experiment_{experiment_i}")
        os.mkdir(experiment_output_dir_i)
        with open(os.path.join(experiment_output_dir_i, 'config.yaml'), 'w') as output_file:
            yaml.dump(config, output_file, default_flow_style=False)

        # run experiment
        test_accuracy_list = []
        val_accuracy_list = []
        for j in range(config['run_control']['num_runs']):
            train_job = BaseTrainingJob(network_type=config['data']['type'],
                                        dataset_dict=dataset_dict, 
                                        network_params=config['network'])
            model, best_acc = train_job.train(experiment_output_dir_i + '/' + str(j), 
                                            optimization_params=config['optimization'],
                                            early_stop_window=config['run_control']['early_stop_window'], 
                                            verbose=True)

            val_accuracy_list.append(best_acc[0] * 100)
            test_accuracy_list.append(best_acc[1] * 100)
            val_accuracy_list.append(0 * 100)
            test_accuracy_list.append(0 * 100)

        # save param and accuracy
        output_data_list = [f"experiment_{experiment_i}"] + [param_dict[key] for key in log_params] + \
            val_accuracy_list + test_accuracy_list
        df = df.append(pd.DataFrame(data=[output_data_list], columns = df.columns))
        df.to_csv(os.path.join(experiment_output_dir, 'accuracy.csv'), index=False)
