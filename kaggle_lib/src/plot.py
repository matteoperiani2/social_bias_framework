import seaborn as sns


def plot_bar_with_bar_labels(
    data,
    x=None,
    y=None,
    hue=None,
    bar_label_rotation=0,
    bar_label_padding=3,
    bar_label_type="edge",
    **kwargs,
):
    ax = sns.barplot(data, x=x, y=y, hue=hue, **kwargs)
    add_bar_labels(
        ax,
        rotation=bar_label_rotation,
        padding=bar_label_padding,
        label_type=bar_label_type,
    )
    return ax


def add_bar_labels(ax, rotation=0, padding=3, label_type="edge", **kwargs):
    for i in ax.containers:
        ax.bar_label(
            i, rotation=rotation, padding=padding, label_type=label_type, **kwargs
        )
