"""Script to expand rooms for a single graph."""
import spark_dsg as dsg
import spark_dsg.mp3d as dsg_mp3d
import pathlib
import click


@click.command()
@click.argument("filepath", type=click.Path(exists=True))
@click.option("-o", "--output", type=str, default=None)
def main(filepath, output):
    filepath = pathlib.Path(filepath).expanduser().absolute()
    G_orig = dsg.DynamicSceneGraph.load(str(filepath))
    G_new = dsg_mp3d.expand_rooms(G_orig, verbose=True)

    if output:
        output = pathlib.Path(output).expanduser().absolute()
    else:
        output = filepath.parent / (f"{filepath.stem}_new_places.json")
    G_new.save(str(output))


if __name__ == "__main__":
    main()
