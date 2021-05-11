import pandas as pd
import seaborn as sns
import tensorboard as tb
from matplotlib import pyplot as plt

from src.utils.paths import DOCS_DIR
from src.utils.plots import save_show_fig


def load_stats(experiment_id: str):
    """
    Reads data into a dataframe from tensorboard.

    Parameters
    ----------
    experiment_id
        Id of the experiment, see https://www.tensorflow.org/tensorboard/dataframe_api.
    """
    experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
    df = experiment.get_scalars()
    return df


def plot_epoch_loss(df: pd.DataFrame, fig_location: str = None,
                    show_figure: bool = False):
    """
    Plots loss from dataframe from a tensorboard data.
    """

    df_epoch_loss = df[df['tag'] == 'epoch_loss']
    sns.set_style("whitegrid", {'axes.grid': True})
    fig, ax = plt.subplots(figsize=(5, 3))
    g = sns.lineplot(ax=ax, data=df_epoch_loss, x='step', y='value', hue='run')
    g.set_ylabel('Loss')
    g.set_xlabel('Epoch')
    g.set(yscale='log')
    g.set_ylim(top=0.03)
    g.set_yticks([0.03, 0.02, 0.01, 0.005, 0.0025])
    g.set_yticklabels([0.03, 0.02, 0.01, 0.005, 0.0025])
    fig.tight_layout()
    ax.legend(title='')
    save_show_fig(fig, fig_location, show_figure)


if __name__ == "__main__":
    """
    The logdir has to be uploaded to tensorboard dev:
    tensorboard dev upload --logdir ./logs/20210315-143811/
    See: https://www.tensorflow.org/tensorboard/dataframe_api
    """
    location = DOCS_DIR.joinpath('figures/implementation/tiny_yolov3_epoch_loss.pdf')
    df = load_stats(experiment_id='A5l9wDz1QaSxTlDpIdMaRQ')
    plot_epoch_loss(df, fig_location=location, show_figure=True)
    pass
