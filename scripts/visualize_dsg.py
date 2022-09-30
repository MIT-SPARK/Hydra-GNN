import spark_dsg as dsg
import click


@click.command()
@click.argument("dsg_path", type=click.Path(exists=True))
def main(dsg_path):
    G = dsg.DynamicSceneGraph.load(dsg_path)
    dsg.render_to_open3d(G)
