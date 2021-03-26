from packaging import version

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import tensorboard as tb
from src.utils.paths import DOCS_DIR
from src.utils.plots import save_show_fig


def load_stats(experiment_id: str = "LCXAHo9bQZqEzqXZJ9cAUA"):
    experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
    df = experiment.get_scalars()
    return df


def plot_epoch_loss(df: pd.DataFrame, fig_location: str = None,
                    show_figure: bool = False):
    df_epoch_loss = df[df['tag'] == 'epoch_loss']
    sns.set_style("whitegrid", {'axes.grid': False})
    fig, ax = plt.subplots(figsize=(5, 3))
    graycolors = sns.mpl_palette('Greys_r', 2)
    g = sns.lineplot(ax=ax, data=df_epoch_loss, x='step', y='value', hue='run')
    #                 palette=graycolors)
    # g.lines[0].set_linestyle("-")
    # g.lines[1].set_linestyle("--")
    g.set_ylabel('Loss')
    g.set_xlabel('Epoch')
    g.set(yscale='log')
    fig.tight_layout()
    ax.legend(title='')
    save_show_fig(fig, fig_location, show_figure)


if __name__ == "__main__":
    location = DOCS_DIR.joinpath('images/tiny_yolov3_epoch_loss.png')
    df = load_stats()
    plot_epoch_loss(df, fig_location=location, show_figure=True)
    pass
