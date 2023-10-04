"""Run training for mp3d."""
from hydra_gnn.utils import (
    project_dir,
    MP3D_BENCHMARK_DIR,
    HYDRA_TRAJ_DIR,
    plist,
    pgrid,
    update_existing_keys,
)
from hydra_gnn.mp3d_utils import generate_mp3d_split, read_mp3d_split
from hydra_gnn.mp3d_dataset import Hydra_mp3d_dataset
from hydra_gnn.base_training_job import BaseTrainingJob
from torch_geometric.loader import DataLoader
import random
from statistics import mean
import click
import shutil
import pickle
import yaml
import torch
import numpy as np
import pandas as pd
import pathlib
from pprint import pprint


# parameter sweep setup
GAT_hidden_dims = plist("GAT_hidden_dims", [[32, 32], [64, 64]])
GAT_head = plist("GAT_head", [3])  # same number of head in each conv layer
GAT_concat = plist(
    "GAT_concat", [True, False]
)  # same concate in each conv layer except the last (always False)
dropout = plist("dropout", [0.4])
lr = plist("lr", [0.001])
weight_decay = plist("weight_decay", [0.0001, 0.001])
param_dict_list = pgrid(
    lr, weight_decay, GAT_hidden_dims, GAT_head, GAT_concat, dropout
)


@click.command()
@click.option("--task_id", default=0, type=int, help="slurm array task ID")
@click.option(
    "--num_tasks", default=1, type=int, help="total number of slurm array tasks"
)
@click.option("--gpu_index", default=-1, type=int, help="gpu index (default: task_id)")
@click.option(
    "--config_dir", default=str(project_dir() / "config"), help="training config dir"
)
@click.option(
    "-c",
    "--config_name",
    default="mp3d_default_config.yaml",
    help="training config file",
)
@click.option(
    "--train_val_ratio",
    default=None,
    type=float,
    nargs=2,
    help="training and validation ratio",
)
@click.option(
    "--same_val_test",
    is_flag=True,
    help="split val/test by trajectory (trajctory 0-1 for validation, 2-4 for testing)",
)
@click.option(
    "--remove_word2vec",
    is_flag=True,
    help="remove word2vec features from node features",
)
@click.option(
    "--single_experiment",
    is_flag=True,
    help="run single experiment using config file instead of parameter sweep",
)
def main(
    task_id,
    num_tasks,
    gpu_index,
    config_dir,
    config_name,
    train_val_ratio,
    same_val_test,
    remove_word2vec,
    single_experiment,
):
    """Run training for mp3d."""
    print(f"cuda available: {torch.cuda.is_available()}")
    config_file = pathlib.Path(config_dir).expanduser().absolute() / config_name

    # parse config file
    with open(config_file, "r") as input_file:
        config = yaml.safe_load(input_file)
    gpu_index = gpu_index if gpu_index != -1 else task_id
    dataset_path = project_dir() / config["data"]["file_path"]
    output_dir = project_dir() / config["logger"]["output_dir"]
    assert task_id < num_tasks

    # setup log folder, accuracy files
    print(f"output directory: {output_dir}")
    if output_dir.exists():
        print("Existing contents might be over-written.")
    else:
        output_dir.mkdir(parents=True)

    # load data and data split
    if train_val_ratio is None:
        print("Preparing training data using default mp3d split.")
        split_dict = read_mp3d_split(MP3D_BENCHMARK_DIR)
    else:
        print("Preparing training data using specified split ratio.")
        random.seed(0)
        split_dict = generate_mp3d_split(
            HYDRA_TRAJ_DIR, train_val_ratio[0], train_val_ratio[1]
        )

    if same_val_test:
        print("Regenerating val/test split using trajectories!")
        print("Using 0-1 for training and 2-4 for testing")

    # create data lists
    dataset_dict = {
        "train": Hydra_mp3d_dataset("train", remove_short_trajectories=False),
        "val": Hydra_mp3d_dataset("val"),
        "test": Hydra_mp3d_dataset("test"),
    }
    with open(dataset_path, "rb") as input_file:
        data_list = pickle.load(input_file)

    if config["network"]["conv_block"] == "GAT_edge":
        [data.compute_relative_pos() for data in data_list]

    if config["data"]["type"] == "homogeneous":
        [data.to_homogeneous() for data in data_list]

    if remove_word2vec:
        [data.remove_last_features(300) for data in data_list]

    if same_val_test:
        for data in data_list:
            if data.get_data_info()["scene_id"] in split_dict["scenes_train"]:
                dataset_dict["train"].add_data(data)
            else:
                if data.get_data_info()["trajectory_id"] in ["0", "1"]:
                    dataset_dict["val"].add_data(data)
                elif data.get_data_info()["trajectory_id"] in ["2", "3", "4"]:
                    dataset_dict["test"].add_data(data)
                else:
                    raise RuntimeError(
                        f"Found invalid trajectory id in input data file {dataset_path}"
                    )
    else:
        for data in data_list:
            if data.get_data_info()["scene_id"] in split_dict["scenes_train"]:
                dataset_dict["train"].add_data(data)
            elif data.get_data_info()["scene_id"] in split_dict["scenes_val"]:
                dataset_dict["val"].add_data(data)
            elif data.get_data_info()["scene_id"] in split_dict["scenes_test"]:
                dataset_dict["test"].add_data(data)
            else:
                raise RuntimeError(
                    f"Founnd invalid scene id in input data file {dataset_path}."
                )

    def _stats(datasets, split):
        return f"{datasets[split].num_scenes()} scenes {len(datasets[split])} graphs"

    print(f"  training: {_stats(dataset_dict, 'train')}")
    print(f"  validation: {_stats(dataset_dict, 'val')}")
    print(f"  test: {_stats(dataset_dict, 'test')}")

    # log resutls
    log_params = list(param_dict_list[0].keys())
    df = pd.DataFrame(
        columns=["log_dir"]
        + log_params
        + ["val_" + str(i) for i in range(config["run_control"]["num_runs"])]
        + ["test_" + str(i) for i in range(config["run_control"]["num_runs"])]
        + ["avg num_epochs", "avg training_time/ecpho (s)", "avg test_time (s)"]
    )

    # run experiment
    if single_experiment:
        experiment_id_list = [0]
    else:
        num_param_set = len(param_dict_list)
        experiment_id_list = list(range(num_param_set))[task_id:num_param_set:num_tasks]

    for experiment_i in experiment_id_list:
        if single_experiment:
            print(f"\nUse all parameters in config: {config_file}")
            experiment_output_dir_i = output_dir / "experiment"
            config_output_path_i = output_dir / "config.yaml"
            accuracy_file_path = output_dir / "accuracy.csv"
            param_list = ["-"] * len(log_params)
        else:
            print(f"\nExperiment {experiment_i+1} / {len(param_dict_list)}")
            experiment_output_dir_i = output_dir / f"experiment_{experiment_i}"
            config_output_path_i = experiment_output_dir_i / "config.yaml"
            accuracy_file_path = output_dir / f"accuracy-{task_id}.csv"
            param_dict = param_dict_list[experiment_i]
            param_list = [param_dict[key] for key in log_params]
            if config["network"]["conv_block"][:3] == "GAT":
                # todo: this is for temporary GAT tuning
                param_dict["GAT_heads"] = len(param_dict["GAT_hidden_dims"]) * [
                    param_dict["GAT_head"]
                ] + [param_dict["GAT_head"]]
                param_dict["GAT_concats"] = len(param_dict["GAT_hidden_dims"]) * [
                    param_dict["GAT_concat"]
                ] + [False]

                # check GAT params
                if (
                    len(param_dict["GAT_heads"])
                    != len(param_dict["GAT_hidden_dims"]) + 1
                    or len(param_dict["GAT_concats"])
                    != len(param_dict["GAT_hidden_dims"]) + 1
                ):
                    print("  Skip - invalid GAT param")
                    continue
            update_existing_keys(config["network"], param_dict)
            update_existing_keys(config["optimization"], param_dict)
            pprint("config:")
            pprint(config)

        # clean up tensorboard output directory
        if experiment_output_dir_i.exists():
            shutil.rmtree(experiment_output_dir_i)
            print(f"Overwritting experiment output folder {experiment_output_dir_i}")

        experiment_output_dir_i.mkdir(parents=True)

        # save config
        with config_output_path_i.open("w") as output_file:
            yaml.dump(config, output_file, default_flow_style=False)

        # run experiment
        test_accuracy_list = []
        val_accuracy_list = []
        training_time_list = []
        training_epoch_list = []
        test_time_list = []
        for j in range(config["run_control"]["num_runs"]):
            print(f"\nRun {j + 1} / {config['run_control']['num_runs']}:")
            train_job = BaseTrainingJob(
                dataset_dict=dataset_dict, network_params=config["network"]
            )
            model, best_acc, info = train_job.train(
                f"{experiment_output_dir_i}/{j}",
                optimization_params=config["optimization"],
                early_stop_window=config["run_control"]["early_stop_window"],
                gpu_index=gpu_index,
                verbose=True,
            )
            val_accuracy_list.append(best_acc[0] * 100)
            test_accuracy_list.append(best_acc[1] * 100)
            training_time_list.append(info["training_time"])
            training_epoch_list.append(info["num_epochs"])
            test_time_list.append(info["test_time"])

            # log per label accuracy for single experiment
            if single_experiment:
                train_accuracy, train_accuracy_matrix = train_job.test(
                    DataLoader(train_job.get_dataset("train"), batch_size=4096), True
                )
                val_accuracy, val_accuracy_matrix = train_job.test(
                    DataLoader(train_job.get_dataset("val"), batch_size=4096), True
                )
                test_accuracy, test_accuracy_matrix = train_job.test(
                    DataLoader(train_job.get_dataset("test"), batch_size=4096), True
                )
                assert (
                    abs(val_accuracy - best_acc[0]) < 0.001
                ), f"{val_accuracy}, {best_acc[0]}"
                assert (
                    abs(test_accuracy - best_acc[1]) < 0.001
                ), f"{test_accuracy}, {best_acc[1]}"
                num_labels = train_accuracy_matrix.shape[1]
                output_header = (
                    [f"{lab}[train]" for lab in range(num_labels)]
                    + [f"{lab}[val]" for lab in range(num_labels)]
                    + [f"{lab}[test]" for lab in range(num_labels)]
                )
                output_header = ",".join(output_header)
                np.savetxt(
                    f"{experiment_output_dir_i}/{j}/accuracy_matrix.csv",
                    np.concatenate(
                        (
                            train_accuracy_matrix,
                            val_accuracy_matrix,
                            test_accuracy_matrix,
                        ),
                        axis=1,
                    ),
                    fmt="%i",
                    delimiter=",",
                    comments="",
                    header=output_header,
                )

        # save param and accuracy
        output_data_list = (
            [f"experiment_{experiment_i}"]
            + param_list
            + val_accuracy_list
            + test_accuracy_list
            + [
                mean(training_epoch_list),
                sum(training_time_list) / sum(training_epoch_list),
                mean(test_time_list),
            ]
        )
        df = pd.concat([df, pd.DataFrame(data=[output_data_list], columns=df.columns)])
        if not single_experiment:
            df["GAT_concat"] = df["GAT_concat"].astype(bool)

        df.to_csv(accuracy_file_path, index=False)


if __name__ == "__main__":
    main()
