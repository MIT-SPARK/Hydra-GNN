import multiprocessing as mp
import spark_dsg as dsg
import pathlib
import shutil
import click


def _copy_graph(args):
    trajectory, output_path, include_mesh = args
    scene_name = trajectory.parent.stem
    trajectory_name = trajectory.stem

    new_path = output_path / f"{scene_name}_{trajectory_name}"
    new_path.mkdir(parents=True)

    for dsg_path in trajectory.rglob("*_partial_dsg*"):
        G = dsg.DynamicSceneGraph.load(str(dsg_path))
        new_dsg_path = new_path / dsg_path.name
        G.save(str(new_dsg_path), include_mesh=include_mesh)


@click.command()
@click.argument("result_dir")
@click.argument("output_dir")
@click.option("-m", "--include-mesh", default=False, is_flag=True)
@click.option("-t", "--num-threads", default=None, type=int)
def main(result_dir, output_dir, include_mesh, num_threads):
    """Copy resulting scene graphs to a single directory."""
    output_path = pathlib.Path(output_dir).expanduser().absolute()
    if output_path.exists():
        click.secho(f"{output_path} already exists", fg="yellow")
        click.confirm(
            f"remove contents under {output_path}?", abort=True, default=False
        )
        shutil.rmtree(output_path)

    output_path.mkdir(parents=True)
    result_path = pathlib.Path(result_dir).expanduser().absolute()

    all_results = []

    scenes = [path for path in result_path.iterdir() if path.is_dir()]
    for scene in scenes:
        all_results += [path for path in scene.iterdir() if path.is_dir()]

    if num_threads is None:
        num_threads = mp.cpu_count()

    tasks = [(path, output_path, include_mesh) for path in all_results]
    with mp.Pool(num_threads) as pool:
        with click.progressbar(
            pool.imap_unordered(_copy_graph, tasks), length=len(tasks)
        ) as bar:
            for _ in bar:
                pass


if __name__ == "__main__":
    main()
