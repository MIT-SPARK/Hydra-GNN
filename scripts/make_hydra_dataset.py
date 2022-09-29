from hydra_python._plugins.habitat import HabitatInterface
from hydra_python._plugins.onnx import OnnxSegmenter
from scipy.spatial.transform import Rotation as R
import hydra_python as hydra
import numpy as np
import subprocess
import pathlib
import click


def _make_log_directories(log_path, incremental, include_vio=False, exist_ok=False):
    if log_path.exists():
        return False

    log_path.mkdir(exist_ok=exist_ok, parents=True)
    (log_path / "logs").mkdir(exist_ok=exist_ok)

    if not incremental:
        return True

    leaves = ["backend", "frontend", "pgmo", "lcd"]
    if include_vio:
        leaves.append("vio")

    for leaf in leaves:
        (log_path / leaf).mkdir(exist_ok=exist_ok)

    return True


def _setup_pipeline(camera, colormap, output_path):
    semantic_path = output_path / "colormap.csv"
    with semantic_path.open("w") as fout:
        colormap.save(fout)

    pipeline = hydra.load_pipeline(
        hydra.get_config_path() / "hydra",
        str(semantic_path),
        log_path=output_path,
        object_labels=[
            3,  # chair
            5,  # table
            6,  # picture
            7,  # cabinet
            8,  # cushion
            10,  # sofa
            11,  # bed
            12,  # curtain
            13,  # chest of drawers
            14,  # plant
            15,  # sink
            18,  # toiled
            19,  # stool
            20,  # towel
            21,  # mirror
            22,  # tv_monitor
            23,  # shower
            25,  # bathtub
            27,  # fireplace
            28,  # lighting
            30,  # shelving
            31,  # blinds
            32,  # seating
            33,  # board_panel
            34,  # furniture
            35,  # clothes
            36,  # objects
            37,  # misc
        ],
        dynamic_labels=[],
    )

    pipeline.set_log_dir(str(output_path / "logs"))
    pipeline.set_camera(camera)
    return pipeline


@click.group()
def cli():
    """Primary target for click."""
    pass


@cli.command(name="single")
@click.argument("model-file")
@click.argument("scene-path")
@click.argument("output-path")
@click.argument("seed", type=int)
@click.argument("length", type=float)
@click.option("--save-step", default=100, type=int)
def run_est_gt(model_file, scene_path, output_path, seed, length, save_step=100):

    click.secho(f"Using seed {seed}", fg="green")
    scene_path = pathlib.Path(scene_path)
    output_path = pathlib.Path(output_path)

    colormap_path = hydra.get_config_path() / "colormaps" / "ade150_mp3d_config.yaml"
    color_info = hydra.SegmentationColormap.from_yaml(colormap_path)
    segmenter = OnnxSegmenter(model_file, color_info)

    data = HabitatInterface(scene_path)
    camera = hydra.Camera(**data.camera_info)
    segmenter.colormap.set_names(data.colormap.names)

    poses = data.get_random_trajectory(
        inflation_radius=0.25, seed=seed, target_length_m=length
    )
    click.secho(f"Trajectory is {poses.get_path_length()} meters long", fg="green")

    gt_output_path = output_path / "gt"
    est_output_path = output_path / "est"

    _make_log_directories(gt_output_path, True)
    _make_log_directories(est_output_path, True)

    gt_pipeline = _setup_pipeline(camera, data.colormap, gt_output_path)
    est_pipeline = _setup_pipeline(camera, segmenter.colormap, est_output_path)

    thresholds = np.arange(save_step, len(poses), save_step)

    with click.progressbar(poses) as bar:
        for index, pose in enumerate(bar):
            timestamp, world_t_body, q_wxyz = pose
            q_xyzw = np.roll(q_wxyz, -1)

            world_T_body = np.eye(4)
            world_T_body[:3, 3] = world_t_body
            world_T_body[:3, :3] = R.from_quat(q_xyzw).as_matrix()
            data.set_pose(timestamp, world_T_body)

            gt_pipeline.step(
                timestamp, world_t_body, q_wxyz, data.semantics, data.depth
            )
            est_pipeline.step(
                timestamp, world_t_body, q_wxyz, segmenter(data.rgb), data.depth
            )

            if index in thresholds or index == len(poses) - 1:
                suffix = f"partial_dsg_{index}.json"
                gt_pipeline.graph.save(str(gt_output_path / f"gt_{suffix}"), True)
                est_pipeline.graph.save(str(est_output_path / f"est_{suffix}"), True)

    gt_pipeline.save(str(gt_output_path))
    est_pipeline.save(str(est_output_path))


@cli.command(name="all")
@click.argument("dataset_path", type=click.Path(exists=True))
@click.argument("model_file")
@click.argument("output")
@click.option("-l", "--length", default=100.0, type=float)
def make_dataset(dataset_path, model_file, output, length):
    """Make a mp3d dataset using hydra_python."""
    output_path = pathlib.Path(output).expanduser().absolute()
    dataset_path = pathlib.Path(dataset_path)

    scenes = [x for x in dataset_path.iterdir() if x.is_dir()]
    trajectory_seeds = [0, 12345, 24690, 49380, 98760]

    for scene_idx, scene in enumerate(scenes):
        scene_name = scene.stem
        click.echo(f"Starting {scene_name} ({scene_idx + 1} of {len(scenes)})")
        scene_output = output_path / scene.stem
        scene_mesh = scene / f"{scene.stem}.glb"

        for index, seed in enumerate(trajectory_seeds):
            traj_name = f"trajectory_{index}"
            trajectory_output = scene_output / traj_name
            if trajectory_output.exists():
                click.secho(
                    f"Skipping existing trajectory {traj_name} for {scene_name}",
                    fg="green",
                )
                continue

            click.secho(f"Running {traj_name} for {scene_name}", fg="green")
            trajectory_output.mkdir(parents=True)
            subprocess.run(
                [
                    "python",
                    __file__,
                    "single",
                    str(model_file),
                    str(scene_mesh),
                    str(trajectory_output),
                    str(seed),
                    str(length),
                ]
            )


if __name__ == "__main__":
    cli()
