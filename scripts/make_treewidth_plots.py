"""Plot object and room treewidths."""
import click
import pickle
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pathlib
from run_treewidth_analysis import data_dir


def _load_df(data_files, labels):
    data_ = []
    length = dict()
    for file_, label_ in zip(data_files, labels):

        with open(file_, "rb") as f:
            d = pickle.load(f)
            length[label_] = len(d["num_nodes"])
            data_.append(d)

    data = dict()

    data["type"] = []
    for key in data_[0].keys():
        data[key] = []

    for i in range(len(labels)):
        for key in data_[0].keys():
            data[key] = [*data[key], *data_[i][key]]
            len_ = len(data_[i][key])
        for j in range(len_):
            data["type"].append(labels[i])

    df = pd.DataFrame.from_dict(data)
    return df


def _plot_treewidth(df, output_path, filename):
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    sns.set()
    sns.set_style("white")
    sns.set_context("paper", font_scale=1.9, rc={"lines.linewidth": 2.0})

    fig, ax = plt.subplots()
    sns.histplot(
        x="treewidth_ub",
        data=df,
        hue="type",
        binwidth=0.8,
        common_norm=False,
        stat="percent",
        ax=ax,
        color=sns.color_palette()[0],
    )

    ax.set_xlabel("Treewidth Upper-Bound")
    ax.get_legend().remove()
    sns.despine()

    fig.set_size_inches([6, 4])
    fig.tight_layout()
    if output_path:
        fig.savefig(str(output_path / filename), bbox_inches="tight")
    else:
        plt.show()


@click.command()
@click.option("-o", "--output_dir", default=None)
def main(output_dir):
    """Plot room and object tree-width."""
    object_df = _load_df(["mp3dnew_objects.pkl"], ["Matterport3D"])
    room_df = _load_df(["mp3dnew_rooms.pkl"], ["Matterport3D"])

    if output_dir:
        output_path = pathlib.Path(output_dir).expanduser().absolute()
    else:
        output_path = data_dir() / "treewidth"

    _plot_treewidth(object_df, output_path, "object_tw.pdf")
    _plot_treewidth(room_df, output_path, "room_tw.pdf")


if __name__ == "__main__":
    main()
