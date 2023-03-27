from hydra_gnn.utils import PROJECT_DIR, plist, pgrid, update_existing_keys
from hydra_gnn.Stanford3DSG_dataset import Stanford3DSG_htree_data, Stanford3DSG_dataset
from hydra_gnn.semisupervised_training_job import SemiSupervisedTrainingJob
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
GAT_hidden_dims = plist('GAT_hidden_dims', [[128], [128, 128]])
dropout = plist('dropout', [0.1, 0.2, 0.3])
lr = plist('lr', [0.001])
weight_decay = plist('weight_decay', [0.0])
param_dict_list = pgrid(lr, weight_decay, GAT_hidden_dims, dropout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', default=0, type=int, help="slurm array task ID")
    parser.add_argument('--num_tasks', default=1, type=int, help="total number of slurm array tasks")
    parser.add_argument('--gpu_index', default=-1, type=int, help="gpu index (default: task_id)")
    parser.add_argument('--config_file', default=os.path.join(PROJECT_DIR, 'config/Stanford3DSG_default_config.yaml'),
                        help='training config file')
    parser.add_argument('--train_val_ratio', default=(0.7, 0.1), type=float, nargs=2, 
                        help='training and validation ratio')
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
    
    # load data and prepare dataset
    dataset = Stanford3DSG_dataset()
    with open(dataset_path, 'rb') as input_file:
        data_list = pickle.load(input_file)
    for data in data_list:
        if config['network']['conv_block'] == 'GAT_edge':
            data.compute_relative_pos()
        if config['data']['type'] == 'homogeneous':
            data.to_homogeneous()
        
        # set clique node features to zero to be consistent with NT paper
        if isinstance(data, Stanford3DSG_htree_data) and config['data']['type'] == 'homogeneous':
            data.get_torch_data().x[data.get_torch_data().clique_has == -1] = 0 
        dataset.add_data(data)

    # log resutls
    log_params = list(param_dict_list[0].keys())
    df = pd.DataFrame(columns=['log_dir'] + log_params + \
        ['val_' + str(i) for i in range(config['run_control']['num_runs'])] + \
            ['test_' + str(i) for i in range(config['run_control']['num_runs'])] + \
                ['avg num_epochs', 'avg training_time/ecpho (s)', 'avg test_time (s)'])

    # node split
    train_ratio, val_ratio = args.train_val_ratio[0], args.train_val_ratio[1]
    test_ratio = 1 - train_ratio - val_ratio
    print(f"Train val test split ratio: {train_ratio} : {val_ratio} : {test_ratio}")

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
            config_output_path = os.path.join(output_dir, "config.yaml")
            accuracy_file_path = os.path.join(output_dir, "accuracy.csv")
            param_list = ['-'] * len(log_params)
        else:
            print(f"\nExperiment {experiment_i+1} / {len(param_dict_list)}")
            experiment_output_dir_i = os.path.join(output_dir, f"experiment_{experiment_i}")
            config_output_path = os.path.join(experiment_output_dir_i, "config.yaml")
            accuracy_file_path = os.path.join(output_dir, f"accuracy-{task_id}.csv")
            # update parameter
            param_dict = param_dict_list[experiment_i]
            param_list = [param_dict[key] for key in log_params]
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

        # clean up tensorboard output directory
        if os.path.exists(experiment_output_dir_i):
            shutil.rmtree(experiment_output_dir_i)
            print(f"Overwritting experiment output folder {experiment_output_dir_i}")
        os.mkdir(experiment_output_dir_i)
        
        # save config
        with open(config_output_path, 'w') as output_file:
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
            dataset.generate_node_split(train_ratio, val_ratio, test_ratio, seed=j)
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

            val_room_accuracy, val_object_accuracy = train_job.test(mask_name='val_mask', get_type_separated_accuracy=True)
            test_room_accuracy, test_object_accuracy = train_job.test(mask_name='test_mask', get_type_separated_accuracy=True)
            val_room_accuracy_list.append(val_room_accuracy * 100)
            val_object_accuracy_list.append(val_object_accuracy * 100)
            test_room_accuracy_list.append(test_room_accuracy * 100)
            test_object_accuracy_list.append(test_object_accuracy * 100)        

        # save param and accuracy
        output_data_list = [f"experiment_{experiment_i}"] + param_list + val_accuracy_list + test_accuracy_list + \
                [mean(training_epoch_list), sum(training_time_list) / sum(training_epoch_list), mean(test_time_list)]
        df = pd.concat([df, pd.DataFrame(data=[output_data_list], columns = df.columns)])
        df.to_csv(accuracy_file_path, index=False)

        # save type separated accuracy for single experiment
        if args.single_experiment:
            with open(accuracy_file_path, 'a') as output_file:
                print(f"\nValidation accuracy: {mean(val_accuracy_list)} +/- {stdev(val_accuracy_list)}", file=output_file)
                print(f"    room: {mean(val_room_accuracy_list)} +/- {stdev(val_room_accuracy_list)}", file=output_file)
                print(f"    object: {mean(val_object_accuracy_list)} +/- {stdev(val_object_accuracy_list)}", file=output_file)

                print(f"Test accuracy: {mean(test_accuracy_list)} +/- {stdev(test_accuracy_list)}", file=output_file)
                print(f"    room: {mean(test_room_accuracy_list)} +/- {stdev(test_room_accuracy_list)}", file=output_file)
                print(f"    object: {mean(test_object_accuracy_list)} +/- {stdev(test_object_accuracy_list)}", file=output_file)
