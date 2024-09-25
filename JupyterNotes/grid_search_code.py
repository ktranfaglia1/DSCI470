import numpy as np
from sklearn.datasets import make_blobs
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
from plot_2d_separator import (plot_2d_separator, plot_2d_classification, plot_2d_scores)
from plot_helpers import cm2 as cm, discrete_scatter


def plot_threefold_split():
    plt.figure(figsize=(15, 1))
    axis = plt.gca()
    bars = axis.barh([0, 0, 0], [11.9, 2.9, 4.9], left=[0, 12, 15], color=[
                     'white', 'grey', 'grey'], hatch="//", edgecolor='k',
                     align='edge')
    bars[2].set_hatch(r"")
    axis.set_yticks(())
    axis.set_frame_on(False)
    axis.set_ylim(-.1, .8)
    axis.set_xlim(-0.1, 20.1)
    axis.set_xticks([6, 13.3, 17.5])
    axis.set_xticklabels(["training set", "validation set",
                          "test set"], fontdict={'fontsize': 20})
    axis.tick_params(length=0, labeltop=True, labelbottom=False)
    axis.text(6, -.3, "Model fitting",
              fontdict={'fontsize': 13}, horizontalalignment="center")
    axis.text(13.3, -.3, "Parameter selection",
              fontdict={'fontsize': 13}, horizontalalignment="center")
    axis.text(17.5, -.3, "Evaluation",
              fontdict={'fontsize': 13}, horizontalalignment="center")
    
def plot_grid_search_overview():
    plt.figure(figsize=(10, 3), dpi=70)
    axes = plt.gca()
    axes.yaxis.set_visible(False)
    axes.xaxis.set_visible(False)
    axes.set_frame_on(False)

    def draw(ax, text, start, target=None):
        if target is not None:
            patchB = target.get_bbox_patch()
            end = target.get_position()
        else:
            end = start
            patchB = None
        annotation = ax.annotate(text, end, start, xycoords='axes pixels',
                                 textcoords='axes pixels', size=20,
                                 arrowprops=dict(
                                     arrowstyle="-|>", fc="w", ec="k",
                                     patchB=patchB,
                                     connectionstyle="arc3,rad=0.0"),
                                 bbox=dict(boxstyle="round", fc="w"),
                                 horizontalalignment="center",
                                 verticalalignment="center")
        plt.draw()
        return annotation
    step = 100
    grr = 400

    final_evaluation = draw(axes, "final evaluation", (5 * step, grr - 3 *
                                                       step))
    retrained_model = draw(axes, "retrained model", (3 * step, grr - 3 * step),
                           final_evaluation)
    best_parameters = draw(axes, "best parameters", (.5 * step, grr - 3 *
                                                     step), retrained_model)
    cross_validation = draw(axes, "cross-validation", (.5 * step, grr - 2 *
                                                       step), best_parameters)
    draw(axes, "parameter grid", (0.0, grr - 0), cross_validation)
    training_data = draw(axes, "training data", (2 * step, grr - step),
                         cross_validation)
    draw(axes, "training data", (2 * step, grr - step), retrained_model)
    test_data = draw(axes, "test data", (5 * step, grr - step),
                     final_evaluation)
    draw(axes, "data set", (3.5 * step, grr - 0.0), training_data)
    draw(axes, "data set", (3.5 * step, grr - 0.0), test_data)
    plt.ylim(0, 1)
    plt.xlim(0, 1.5)


def heatmap(values, xlabel, ylabel, xticklabels, yticklabels, cmap=None,
            vmin=None, vmax=None, ax=None, fmt="%0.2f"):
    if ax is None:
        ax = plt.gca()
    # plot the mean cross-validation scores
    img = ax.pcolor(values, cmap=cmap, vmin=vmin, vmax=vmax)
    img.update_scalarmappable()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(xticklabels)) + .5)
    ax.set_yticks(np.arange(len(yticklabels)) + .5)
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)
    ax.set_aspect(1)

    for p, color, value in zip(img.get_paths(), img.get_facecolors(),
                               img.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.mean(color[:3]) > 0.5:
            c = 'k'
        else:
            c = 'w'
        ax.text(x, y, fmt % value, color=c, ha="center", va="center")
    return img