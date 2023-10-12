"""Train stanford3d."""
from hydra_gnn.utils import project_dir, plist, pgrid
from hydra_gnn.Stanford3DSG_dataset import Stanford3DSG_htree_data, Stanford3DSG_dataset
from hydra_gnn.semisupervised_training_job import SemiSupervisedTrainingJob
from statistics import mean, stdev
import click
import shutil
import pickle
import yaml
import torch
import pathlib
import pandas as pd
from pprint import pprint


# parameter sweep setup (note: last GAT hidden_dims is label dim and not specified)
GAT_hidden_dims = plist("GAT_hidden_dims", [[64, 64], [128, 128]])
GAT_head = plist("GAT_head", [6])  # same number of head in each conv layer
GAT_concat = plist(
    "GAT_concat", [True]
)  # same concate in each conv layer except the last (always False)
dropout = plist("dropout", [0, 0.25])
lr = plist("lr", [0.001, 0.0005])
weight_decay = plist("weight_decay", [0.0, 0.0001, 0.001])
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
    "--config_dir", 
    default=str(project_dir() / "config/Stanford3D"), 
    help="training config dir"
)
@click.option(
    "-c",
    "--config_name",
    default="baseline_GraphSAGE.yaml",
    help="training config file",
)
@click.option(
    "--train_val_ratio",
    default=(0.7, 0.1),
    type=float,
    nargs=2,
    help="training and validation ratio",
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
    single_experiment,
):
    """Run stanford3d training."""
    print(f"cuda available: {torch.cuda.is_available()}")
    config_file = pathlib.Path(config_dir).expanduser().absolute() / config_name

    # parse config file
    with config_file.open("r") as input_file:
        config = yaml.safe_load(input_file)
    gpu_index = gpu_index if gpu_index != -1 else task_id
    dataset_path = project_dir() / config["data"]["file_path"]
    output_dir = project_dir() / config["logger"]["output_dir"]
    assert task_id < num_tasks

    # setup log folder, accuracy files
    print(f"output directory: {output_dir}")
    if output_dir.exists():
        print("Existing contents may be overwritten.")
    else:
        output_dir.mkdir(parents=True)

    # load data and prepare dataset
    dataset = Stanford3DSG_dataset()
    with open(dataset_path, "rb") as input_file:
        data_list = pickle.load(input_file)
    for data in data_list:
        if config["network"]["conv_block"] == "GAT_edge":
            data.compute_relative_pos()
        if config["data"]["type"] == "homogeneous":
            data.to_homogeneous()

        # set clique node features to zero to be consistent with NT paper
        if (
            isinstance(data, Stanford3DSG_htree_data)
            and config["data"]["type"] == "homogeneous"
        ):
            data.get_torch_data().x[data.get_torch_data().clique_has == -1] = 0
        dataset.add_data(data)

    # log resutls
    log_params = list(param_dict_list[0].keys())
    df = pd.DataFrame(
        columns=["log_dir"]
        + log_params
        + ["val_" + str(i) for i in range(config["run_control"]["num_runs"])]
        + ["test_" + str(i) for i in range(config["run_control"]["num_runs"])]
        + ["avg num_epochs", "avg training_time/ecpho (s)", "avg test_time (s)"]
    )

    # node split
    train_ratio, val_ratio = train_val_ratio[0], train_val_ratio[1]
    test_ratio = 1 - train_ratio - val_ratio
    print(f"Train val test split ratio: {train_ratio} : {val_ratio} : {test_ratio}")

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
            config_output_path = output_dir / "config.yaml"
            accuracy_file_path = output_dir / "accuracy.csv"
            param_list = ["-"] * len(log_params)
        else:
            print(f"\nExperiment {experiment_i+1} / {len(param_dict_list)}")
            experiment_output_dir_i = output_dir / f"experiment_{experiment_i}"
            config_output_path = experiment_output_dir_i / "config.yaml"
            accuracy_file_path = output_dir / f"accuracy-{task_id}.csv"
            # update parameter
            param_dict = param_dict_list[experiment_i]
            param_list = [param_dict[key] for key in log_params]
            if config["network"]["conv_block"][:3] == "GAT":
                # todo: this is for temporary GAT tuning with single attention head
                if config["data"]["type"] == "homogeneous":
                    param_dict["GAT_heads"] = len(param_dict["GAT_hidden_dims"]) * [
                        param_dict["GAT_head"]
                    ]
                    param_dict["GAT_concats"] = len(param_dict["GAT_hidden_dims"]) * [
                        param_dict["GAT_concat"]
                    ]
                else:
                    param_dict["GAT_heads"] = len(param_dict["GAT_hidden_dims"]) * [
                        param_dict["GAT_head"]
                    ] + [param_dict["GAT_head"]]
                    param_dict["GAT_concats"] = len(param_dict["GAT_hidden_dims"]) * [
                        param_dict["GAT_concat"]
                    ] + [False]

        pprint("config:")
        pprint(config)

        # clean up tensorboard output directory
        if experiment_output_dir_i.exists():
            shutil.rmtree(experiment_output_dir_i)
            print(f"Overwritting experiment output folder {experiment_output_dir_i}")
        experiment_output_dir_i.mkdir(parents=True)

        # save config
        with config_output_path.open("w") as output_file:
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
        for j in range(config["run_control"]["num_runs"]):
            print(f"\nRun {j + 1} / {config['run_control']['num_runs']}:")
            dataset.generate_node_split(train_ratio, val_ratio, test_ratio, seed=j)
            train_job = SemiSupervisedTrainingJob(
                dataset=dataset, network_params=config["network"]
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

            # save type separated accuracy for single experiment
            if single_experiment:
                val_room_accuracy, val_object_accuracy = train_job.test(
                    mask_name="val_mask", get_type_separated_accuracy=True
                )
                test_room_accuracy, test_object_accuracy = train_job.test(
                    mask_name="test_mask", get_type_separated_accuracy=True
                )
                val_room_accuracy_list.append(val_room_accuracy * 100)
                val_object_accuracy_list.append(val_object_accuracy * 100)
                test_room_accuracy_list.append(test_room_accuracy * 100)
                test_object_accuracy_list.append(test_object_accuracy * 100)

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
        df.to_csv(str(accuracy_file_path), index=False)

        # save type separated accuracy for single experiment
        if single_experiment:
            with accuracy_file_path.open("a") as output_file:
                print(
                    f"\nValidation accuracy: {mean(val_accuracy_list)} +/- {stdev(val_accuracy_list)}",
                    file=output_file,
                )
                print(
                    f"    room: {mean(val_room_accuracy_list)} +/- {stdev(val_room_accuracy_list)}",
                    file=output_file,
                )
                print(
                    f"    object: {mean(val_object_accuracy_list)} +/- {stdev(val_object_accuracy_list)}",
                    file=output_file,
                )

                print(
                    f"Test accuracy: {mean(test_accuracy_list)} +/- {stdev(test_accuracy_list)}",
                    file=output_file,
                )
                print(
                    f"    room: {mean(test_room_accuracy_list)} +/- {stdev(test_room_accuracy_list)}",
                    file=output_file,
                )
                print(
                    f"    object: {mean(test_object_accuracy_list)} +/- {stdev(test_object_accuracy_list)}",
                    file=output_file,
                )


if __name__ == "__main__":
    main()
