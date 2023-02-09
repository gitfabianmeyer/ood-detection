import os

import pandas as pd
from ood_detection.config import Config
import wandb


def get_history_from_project(project, drop_subs=True):
    api = wandb.Api()
    # Project is specified by <entity/project-name>
    runs = api.runs(project)
    histories = []
    for run in runs:
        # run.summary are the output key/values like accuracy.
        # We call ._json_dict to omit large files
        history = run.history()
        history['name'] = run.name
        histories.append(history)

    concat = pd.concat(histories)
    if drop_subs:
        concat = concat.drop([name for name in concat.columns if name.startswith("_")], axis=1).reset_index(drop=True)
    return concat


def save_csv(dataframe, name):
    csv_path = os.path.join(Config.DATAPATH, 'csvs')
    os.makedirs(csv_path, exist_ok=True)
    dataframe.to_csv(os.path.join(csv_path, name))


def save_plot(plot, name, dpi=1200):
    plot_path = os.path.join(Config.DATAPATH, 'plots')
    os.makedirs(plot_path, exist_ok=True)
    plot.savefig(os.path.join(plot_path, name), bbox_inches='tight', dpi=dpi)


def plot_corruption_distances(df, split_by, group_by, metric):
    unique_vals = len(df[split_by].unique())
    if unique_vals < 4:
        fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharey=True)
    else:
        fig, axs = plt.subplots(unique_vals // 2, int(np.ceil(unique_vals / 2)), figsize=(20, 10), sharey=True)

    for i, name in enumerate(df[split_by].unique()):
        ax = axs[i % 2][i // 2]
        ax.set_title(name.upper())
        ax.set_ylabel(metric)
        ax.set_xticks([1, 3, 5])
        ax.grid(visible=True)

        for cname, group in df[df[split_by] == name].groupby(group_by):
            group.plot(x='severity', y=metric, ax=axs[i % 2][i // 2], label=cname)
    if unique_vals % 2 != 0:
        axs[-1, -1].axis('off')
    fig.suptitle(f"{metric.upper()} for all {split_by}")
