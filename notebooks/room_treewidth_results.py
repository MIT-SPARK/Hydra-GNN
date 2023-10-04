# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pickle
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=2)
# sns.set(font_scale=2)

# %% [markdown]
# # Analysis of 3D-DSGs: Room Graph

# %%
# data_files = ['data/mp3d_room_analysis_complete.pkl',
#               'data/mp3d_room_analysis_trajectory.pkl']

# labels = ['complete', 'trajectory']
data_files = ["data/mp3dnew_rooms.pkl"]
labels = ["Matterport3D"]

data_ = []
length = dict()
for file_, label_ in zip(data_files, labels):
    with open(file_, "rb") as f:
        d = pickle.load(f)
        length[label_] = len(d["num_nodes"])
        data_.append(d)

# %%
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

# %%
df = pd.DataFrame.from_dict(data)

# %% [markdown]
# ## Number of Nodes

# %%
sns.histplot(data=df, x="num_nodes", hue="type", common_norm=False, stat="density")

# %% [markdown]
# ## Trees or Not

# %%
sns.histplot(
    x="is_tree", data=df, hue="type", binwidth=0.8, common_norm=False, stat="density"
)

# %% [markdown]
# ## Connected or Not

# %%
sns.histplot(
    x="is_disconnected",
    data=df,
    hue="type",
    binwidth=0.8,
    common_norm=False,
    stat="density",
)

# %% [markdown]
# ## Treewidth Upper-Bound

# %%
sns_plot = sns.histplot(
    x="treewidth_ub",
    data=df,
    hue="type",
    binwidth=0.8,
    common_norm=False,
    stat="density",
)
sns_plot.set(xlabel="Treewidth Upper-Bound")
sns_plot.legend_.set_title("")

# %%
# Saving figure
fig = sns_plot.get_figure()
fig.savefig("runs/room_tw.png", bbox_inches="tight")

# %%
table_ = dict()
table_["complete"] = data_[0]["treewidth_ub"].count(2)
table_["trajectory"] = data_[1]["treewidth_ub"].count(2)

# %%
table_

# %%
length

# %% [markdown]
# ## Graph Degree

# %%
sns.histplot(
    x="degree", data=df, hue="type", binwidth=0.8, common_norm=False, stat="density"
)

# %% [markdown]
# ## Planarity

# %%
sns.histplot(x="is_planar", data=df, hue="type", common_norm=False, stat="density")

# %% [markdown]
# This shows that all room graphs are planar.

# %% [markdown]
# # Conclusion
#
# - There isn’t much difference between CNN-estimated and GT-semantic estimated scene graphs.
#
# - Room segmentation can be improved by “assuming” that treewidth=1 (or imposing this).
#
# - Empirically, we have tw(room-graph) <= 2.
#
# - Sequential data vs Final state:
#     - tw=2 for 1% of all sequentially generated graphs
#     - tw=2 for 5% of all sequentially generated graphs
#

# %%

# %%
