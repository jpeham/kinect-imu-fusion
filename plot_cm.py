import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def remove_chartjunk(ax, spines, grid=None, ticklabels=None):
    """
Removes "chartjunk", such as extra lines of axes and tick marks.

If grid="y" or "x", will add a white grid at the "y" or "x" axes,
respectively

If ticklabels="y" or "x", or ['x', 'y'] will remove ticklabels from that
axis
"""
    all_spines = ['top', 'bottom', 'right', 'left']
    for spine in spines:
        ax.spines[spine].set_visible(False)

    # For the remaining spines, make their line thinner and a slightly
    # off-black dark grey
    for spine in all_spines:
        if spine not in spines:
            ax.spines[spine].set_linewidth(0.5)
    x_pos = {'top', 'bottom'}
    y_pos = {'left', 'right'}
    xy_pos = [x_pos, y_pos]
    xy_ax_names = ['xaxis', 'yaxis']

    for ax_name, pos in zip(xy_ax_names, xy_pos):
        axis = ax.__dict__[ax_name]
        # axis.set_tick_params(color=almost_black)
        if axis.get_scale() == 'log':
            # if this spine is not in the list of spines to remove
            for p in pos.difference(spines):
                axis.set_ticks_position(p)
                # axis.set_tick_params(which='both', p)
        else:
            axis.set_ticks_position('none')

    if grid is not None:
        for g in grid:
            assert g in ('x', 'y')
            ax.grid(axis=grid, color='white', linestyle='-', linewidth=0.5)

    if ticklabels is not None:
        if type(ticklabels) is str:
            assert ticklabels in {'x', 'y'}
            if ticklabels == 'x':
                ax.set_xticklabels([])
            if ticklabels == 'y':
                ax.set_yticklabels([])
        else:
            assert set(ticklabels) | {'x', 'y'} > 0
            if 'x' in ticklabels:
                ax.set_xticklabels([])
            elif 'y' in ticklabels:
                ax.set_yticklabels([])


def plot_cm(y_true,
            y_pred,
            order=None,
            rename=None,
            hypo_names_rotation='vertical',
            hypo_names_above=True,
            textsize_values=None,
            textsize_labels=None,
            textsize_expl=None,
            lang='en',
            plotlegend=False,
            diag_color=True,
            fig_width=8.5,
            name='test'):
    """
    Compute a confusion matrix and plot it.

    :param y_true: list or 1D-np.array of true labels
    :param y_pred: list or 1D-np.array of predicted labels. Must be of same length as y_true.
    :param order: Optional: list of the class labels used in y_true and y_pred. May be used to order the labels or select a subset.
    :param rename: Optional: A dict mapping class labels used in y_true and y_pred to new names that will be displayed instead.
    :param hypo_names_rotation: 'vertical' or 'horizontal' to select rotation of hypothesis labels
    :param hypo_names_above: if True, show hypothesis labels above the plot instead of below
    :param textsize_values: textsize of values in the matrix
    :param textsize_labels: textsize of the class names labeling the rows/columns
    :param textsize_expl: textsize of explanations ('Hypothesis', 'False positives', ...)
    :param lang: language of explanations. 'en' or 'de'
    :param plotlegend: if True, plot the colormap legend.
    :param diag_color: if True, paint the main diagonal (correct classifications) in a different color
    :param fig_width: width of figure (height is selected by fixed aspect ratio)
    :param name: name to save plot under
    :return: 
    """

    fig_aspect = 7.5 / 8.5
    figsize = (fig_width, fig_width * fig_aspect)

    labeldict = {
        'hypothesis': {'de': 'Hypothese', 'en': 'Hypothesis'},
        'reference': {'de': 'Referenz', 'en': 'Reference'},
        'falsepos': {'de': 'Falsch-positive', 'en': 'False positives'},
        'falseneg': {'de': 'Falsch-negative', 'en': 'False negatives'}
    }

    if order is None:
        order = unique_labels(y_true, y_pred)

    if rename is None:
        label_names = order
    else:
        label_names = [rename[l] for l in order]

    conf_arr = confusion_matrix(y_true, y_pred, order)

    dim = len(conf_arr)
    conf_arr_no_diag = np.copy(conf_arr)
    np.fill_diagonal(conf_arr_no_diag, 0)

    with np.errstate(invalid='ignore'):  # ignore warnings due to NaN
        norm_conf = (conf_arr.astype(np.float) / conf_arr.sum(axis=1)[:, np.newaxis])

        false_negatives_nn = np.sum(conf_arr_no_diag, axis=1)  # false negatives per row
        false_negatives = (false_negatives_nn.astype(np.float) / np.sum(conf_arr, axis=1)).reshape(1, (
            -1))  # false negatives per line / all per line

        false_positives_nn = np.sum(conf_arr_no_diag, axis=0)  # false positives per column
        false_positives = (false_positives_nn.astype(np.float) / np.sum(conf_arr, axis=0)).reshape(1, (
            -1))  # false positives per column / all per column

    fig = plt.figure(figsize=figsize)
    plt.clf()

    reds = plt.cm.Reds
    reds.set_under('white')
    greens = plt.cm.Greens
    greens.set_under('white')
    reds.set_bad('white')  # kind of useless, why would there be masked values in the confmatrix?

    gs = grd.GridSpec(2, 3, height_ratios=[dim, 1], width_ratios=[dim, 1, 0.8], wspace=0.15, hspace=0.2)
    ax = fig.add_subplot(gs[0])
    ax.set_aspect('equal')
    res = ax.imshow(norm_conf, cmap=reds, vmin=0.001, vmax=1)  # interpolation='nearest' does nothing

    if diag_color:  # plot correct values green
        diag = np.diag(np.diag(norm_conf))
        with np.errstate(invalid='ignore'):  # ignore warnings due to NaN
            diag_masked = np.ma.masked_array(diag, diag <= 0)
        res2 = ax.imshow(diag_masked, cmap=greens, vmin=0.001, vmax=1)  # interpolation='nearest' does nothing


    plt.xlabel(labeldict['hypothesis'][lang], size=textsize_expl)
    if hypo_names_above:
        ax.xaxis.set_label_position('top')  # move 'Hypothesis' label to top
    plt.ylabel(labeldict['reference'][lang], size=textsize_expl)

    # numbers (only if not zero)
    for x in range(dim):
        for y in range(dim):
            if norm_conf[x][y] > 0:
                ax.annotate(str(conf_arr[x][y]), xy=(y, x),
                            horizontalalignment='center',
                            verticalalignment='center',
                            color='w' if norm_conf[x][y] > 0.5 else 'k',
                            size=textsize_values)

    # class labels
    # plt.xticks(range(dim), order[:dim], rotation=hypo_names_rotation, size=textsize_labels)
    plt.xticks(range(dim), label_names, rotation=hypo_names_rotation, size=textsize_labels)
    if hypo_names_above:
        ax.xaxis.tick_top()  # move hypothesis class names to top
    # plt.yticks(range(dim), order[:dim], size=textsize_labels)
    plt.yticks(range(dim), label_names, size=textsize_labels)

    remove_chartjunk(ax, ['top', 'right', 'left', 'bottom'])  # removes ticks and border (second arg)

    # bar on the right with false negatives
    if False: # nein, keine False Positive und False Negative Balken
        ax_neg = fig.add_subplot(gs[1])

        res_neg = ax_neg.imshow(false_negatives.T, cmap=reds,
                                interpolation="nearest", vmin=0.001, vmax=1)
        remove_chartjunk(ax_neg, ['top', 'right', 'bottom'])

        for x in range(dim):
            ax_neg.annotate(str(false_negatives_nn[x]), xy=(0, x),
                            horizontalalignment='center',
                            verticalalignment='center',
                            color='w' if false_negatives[0][x] > 0.5 else 'k',
                            size=textsize_values)
        plt.xticks([])
        plt.yticks([])
        # label 'false negatives'
        ax_neg.yaxis.set_label_position("right")
        ax_neg.set_ylabel(labeldict['falseneg'][lang], size=textsize_expl)

        # bar on the bottom with false positives
        ax_pos = fig.add_subplot(gs[3])
        res_pos = ax_pos.imshow(false_positives, cmap=reds,
                                interpolation="nearest", vmin=0.001, vmax=1)
        remove_chartjunk(ax_pos, ['left', 'right', 'bottom'])

        for x in range(dim):
            ax_pos.annotate(str(false_positives_nn[x]), xy=(x, 0),
                            horizontalalignment='center',
                            verticalalignment='center',
                            color='w' if false_positives[0][x] > 0.5 else 'k',
                            size=textsize_values)

        plt.xticks([])
        plt.yticks([])

        # label 'false positives'
        ax_pos.set_xlabel(labeldict['falsepos'][lang], size=textsize_expl)

    if plotlegend:
        ax_clrbr = fig.add_subplot(gs[:, -1])
        cb = fig.colorbar(res, cax=ax_clrbr)

    # bbox_inches = "tight" needed to not cut off xlabels
    plt.savefig(name + ".svg", format="svg", bbox_inches = "tight")
    plt.savefig(name + ".png", format="png", bbox_inches = "tight")
    plt.savefig(name + ".pdf", format="pdf", bbox_inches = "tight")

    plt.show()  # J: show needed here because we clear the figure, fig needs to be cleared for barplot

    plt.clf()
