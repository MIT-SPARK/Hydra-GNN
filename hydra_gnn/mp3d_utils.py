from hydra_gnn.utils import MP3D_BENCHMARK_DIR, HYDRA_TRAJ_DIR
import random
import os
import argparse


def generate_mp3d_split(hydra_dataset_dir, train_ratio, val_ratio):
    test_ratio = 1 - train_ratio - val_ratio
    assert test_ratio > 0, "Invalid train/val/test split."

    # read all scene names
    trajectory_dirs = os.listdir(hydra_dataset_dir)
    scene_names = list(set(full_name.split("_")[0] for full_name in trajectory_dirs))
    num_scenes = len(scene_names)
    assert num_scenes == 90

    # randomly permute scene names to generate split
    random.shuffle(scene_names)    
    scenes_train = scene_names[0: round(train_ratio * num_scenes)]
    scenes_val = scene_names[round(train_ratio * num_scenes): round(-test_ratio * num_scenes)]
    scenes_test = scene_names[round(-test_ratio * num_scenes): ]

    return {'scenes_train': scenes_train, 'scenes_test': scenes_test, 'scenes_val': scenes_val}


def read_mp3d_split(benchmark_dir):
    with open(benchmark_dir + "/scenes_train.txt") as input_file:
        data = input_file.read()
        scenes_train = data.split('\n')[0:-1]
    
    with open(benchmark_dir + "/scenes_test.txt") as input_file:
        data = input_file.read()
        scenes_test = data.split('\n')[0:-1]

    with open(benchmark_dir + "/scenes_val.txt") as input_file:
        data = input_file.read()
        scenes_val = data.split('\n')[0:-1]
    
    return {'scenes_train': scenes_train, 'scenes_test': scenes_test, 'scenes_val': scenes_val}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_val_ratio', default=None, type=float, nargs=2, 
                        help='training and validation ratio')
    args = parser.parse_args()

    if args.train_val_ratio is None:
        print("Preparing training data using default mp3d split.")
        split_dict = read_mp3d_split(MP3D_BENCHMARK_DIR)
    else:
        print("Preparing training data using specified split ratio.")
        split_dict = generate_mp3d_split(HYDRA_TRAJ_DIR, args.train_val_ratio[0], args.train_val_ratio[1])
