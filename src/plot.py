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


# def plot_minority_distribution(min_pred, min_labels, sterotype_labels, sterotype_preds):
#     type = ["Minority", "Stereotype"]
#     name = ["predictions", "labels"]
#     fig, axes = plt.subplots(1, 4, figsize=(15, 4))
#     for i, (data, ax) in enumerate(
#         zip(
#             [min_pred, min_labels, sterotype_labels, sterotype_preds],
#             axes.ravel(),
#             strict=True,
#         )
#     ):
#         _plot_word_bar(data, ax=ax)
#         ax.set_title(f"{type[i//2]} {name[i%2]}")


# def _plot_word_bar(data, ax, n_words=10):
#     if isinstance(data[0], list):
#         topic_words = [y.lower() if y != "" else "''" for x in data for y in x]
#     else:
#         topic_words = [x.lower() if x != "" else "''" for x in data]
#     word_count_dict = dict(Counter(topic_words))
#     popular_words = sorted(word_count_dict, key=word_count_dict.get, reverse=True)
#     popular_words_nonstop = [
#         w for w in popular_words if w not in stopwords.words("english")
#     ]
#     total = sum([word_count_dict[w] for w in reversed(popular_words_nonstop)])
#     ax.barh(
#         range(n_words),
#         [
#             word_count_dict[w] / total
#             for w in reversed(popular_words_nonstop[0:n_words])
#         ],
#     )
#     ax.set_yticks(
#         [x + 0.5 for x in range(n_words)], reversed(popular_words_nonstop[0:n_words])
#     )
#     for i in ax.containers:
#         ax.bar_label(i, padding=2)
