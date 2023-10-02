# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: venv_hydra_gnn
#     language: python
#     name: python3
# ---

# %%
from hydra_gnn.utils import PROJECT_DIR, MP3D_BENCHMARK_DIR
from hydra_gnn.mp3d_utils import read_mp3d_split
from hydra_gnn.mp3d_dataset import Hydra_mp3d_dataset
from hydra_gnn.base_training_job import BaseTrainingJob
import os
import pickle
import yaml
from statistics import mean, stdev

# %%
import torch

torch.cuda.is_available()

# %%
# Parameters
import datetime

t = datetime.datetime.now()
output_name = t.strftime("%m%d%H%M")

config = dict()
config["data"] = {
    "file_path": "output/preprocessed_mp3d/htree_gt60.pkl",
    "type": "heterogeneous",
}
config["run_control"] = {"num_runs": 1, "early_stop_window": 300}
config["network"] = {
    "conv_block": "GAT_edge",
    "dropout": 0.4,
    "GAT_hidden_dims": [32, 32],
    "GAT_heads": [1, 1, 1],
    "GAT_concats": [True, True, False],
}
config["optimization"] = {
    "lr": 0.001,
    "num_epochs": 800,
    "weight_decay": 0.0,
    "batch_size": 2048,
}
config["logger"] = {"output_dir": "output/log_test/" + output_name}

# %%
dataset_path = os.path.join(PROJECT_DIR, config["data"]["file_path"])
output_dir = os.path.join(PROJECT_DIR, config["logger"]["output_dir"])
num_runs = config["run_control"]["num_runs"]
network_type = config["data"]["type"]
early_stop_window = config["run_control"]["early_stop_window"]
network_params = config["network"]
optimization_params = config["optimization"]

print(network_type)
print(dataset_path)
print(output_dir)

# %% [markdown]
# ## Load dataset

# %%
split_dict = read_mp3d_split(MP3D_BENCHMARK_DIR)
with open(dataset_path, "rb") as input_file:
    data_list = pickle.load(input_file)

# %%
torch_data = data_list[0].get_torch_data()
for node_type in torch_data.x_dict:
    if "y" in torch_data[node_type]:
        print(node_type, torch_data[node_type].y)

# %%
dataset_dict = {
    "train": Hydra_mp3d_dataset("train", remove_short_trajectories=True),
    "val": Hydra_mp3d_dataset("val"),
    "test": Hydra_mp3d_dataset("test"),
}

if config["network"]["conv_block"] == "GAT_edge":
    [data.compute_relative_pos() for data in data_list]
if network_type[:11] == "homogeneous":
    [data.to_homogeneous() for data in data_list]
# [data.remove_last_features(300) for data in data_list]

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
print(
    f"  training: {dataset_dict['train'].num_scenes()} scenes {len(dataset_dict['train'])} graphs\n"
    f"  validation: {dataset_dict['val'].num_scenes()} scenes {len(dataset_dict['val'])} graphs\n"
    f"  test: {dataset_dict['test'].num_scenes()} scenes {len(dataset_dict['test'])} graphs"
)

# %% [markdown]
# ## Run experiment

# %%
experiment_output_dir = os.path.join(PROJECT_DIR, config["logger"]["output_dir"])
assert not os.path.exists(experiment_output_dir), "Output directory exists"
os.mkdir(experiment_output_dir)

test_accuracy_list = []
val_accuracy_list = []
training_time_list = []
training_epoch_list = []
test_time_list = []
for j in range(config["run_control"]["num_runs"]):
    train_job = BaseTrainingJob(
        dataset_dict=dataset_dict, network_params=config["network"]
    )
    model, best_acc, info = train_job.train(
        experiment_output_dir + "/" + str(j),
        optimization_params=config["optimization"],
        early_stop_window=config["run_control"]["early_stop_window"],
        verbose=True,
    )

    val_accuracy_list.append(best_acc[0] * 100)
    test_accuracy_list.append(best_acc[1] * 100)
    training_time_list.append(info["training_time"])
    training_epoch_list.append(info["num_epochs"])
    test_time_list.append(info["test_time"])

# %%
if config["run_control"]["num_runs"] > 2:
    print(
        f"Validation accuracy: {mean(val_accuracy_list)} +/- {stdev(val_accuracy_list)}"
    )
    print(f"Test accuracy: {mean(test_accuracy_list)} +/- {stdev(test_accuracy_list)}")

# %% [markdown]
# ## Save last model

# %%
# save model hyper-parameters
network_params = train_job.get_network_params()
graph_type, network_type = train_job.train_job_type().split(" ")
with open(os.path.join(experiment_output_dir, "model.yaml"), "w") as output_file:
    yaml.dump(
        {
            "graph_type": graph_type,
            "network_type": network_type,
            "network_params": network_params,
        },
        output_file,
        default_flow_style=False,
    )

# save last model weights
model_weights_path = os.path.join(
    experiment_output_dir + "/" + str(j), "model_weights.pth"
)
torch.save(model.state_dict(), model_weights_path)
print("model params saved to:", experiment_output_dir + "/" + str(j))
