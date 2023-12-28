import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

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


def plot_classification_results(results):
    annotation_type = [
        "Offensive",
        "Intentional",
        "Sex/Lewd content",
        "Group targetted",
        "Speaker in group",
    ]

    split_dfs = {}
    for split, split_res in results.items():
        values = []
        for seed,seed_res in split_res.items():
            for score_name, score_val in seed_res.items():
                values.append({
                    'name': score_name,
                    'score': score_val,
                    'seed': seed,
                })
        split_dfs[split] = pd.DataFrame(values)
    
    _, axs = plt.subplots(1, 2, figsize=(15,4))
    for i,(split, df) in enumerate(split_dfs.items()):
        sns.barplot(df, x='name', y='score', hue='seed', ax=axs[i])
        axs[i].set_title(f'Classification score on {split} set')
        axs[i].set_xlabel(None)
        axs[i].set_ylabel('Score')
    plt.plot() 


def plot_generative_results(results):
    split_dfs = {}
    for split, split_res in results.items():
        values = []
        for seed,seed_res in split_res.items():
            for score_name, score_val in seed_res.items():
                values.append({
                    'name': score_name,
                    'score': score_val,
                    'seed': seed,
                })
        split_dfs[split] = pd.DataFrame(values)
    
    _, axs = plt.subplots(1, 2, figsize=(15,4))
    for i,(split, df) in enumerate(split_dfs.items()):
        sns.barplot(df, x='name', y='score', hue='seed', ax=axs[i])
        axs[i].set_title(f'Generative score on {split} set')
        axs[i].set_xlabel(None)
        axs[i].set_ylabel('Score')
    plt.plot()



def plot_classification_cm(labels, predictions, annotation_type):
    _, axs = plt.subplots(1, 5, figsize=(35, 15))
    for j in range(5):
        lbls = labels[j]
        preds = predictions[j]
        cm = confusion_matrix(lbls[lbls != -1], preds[lbls != -1], normalize='all')
        cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        cm_disp.plot(
            ax=axs[j],
            cmap="Blues",
            values_format="0.2f",
            colorbar=False,
            xticks_rotation=90,
        )
        axs[j].set_title(f"{annotation_type[j]}")
        axs[j].set(xlabel=None)
        axs[j].set(ylabel=None)
    plt.show()

def plot_f1_bar(results):
    data = _prepare_data(results)
    _plot_f1_bar(data, hue="split")


def _prepare_data(results):
    splits = results.keys()

    def prepare_data(results):
        print(pd.DataFrame(results, index=[0]))
        x = (
            pd.DataFrame(results, index=[0])
            .applymap(lambda t: round(t * 100, 2))
            .reset_index()
            .rename(columns={"index": "metric"})
        )
        return x.melt(id_vars=["metric"], value_vars=splits, var_name="split")

    dfs = []
    for _, split_res in results.items():
        split_dfs = []
        for seed, res in split_res.items():
            x = prepare_data(res)
            x["seed"] = seed
            dfs.append(x)
            x = pd.concat(dfs)
        dfs.append(split_dfs)
    return x


def _plot_f1_bar(data, hue: str, multiple_seeds=False):
    errorbar = ("sd", 2) if multiple_seeds else None
    bar_label_type = "center" if multiple_seeds else "edge"

    plt.figure(figsize=(15, 10))
    ax = plot_bar_with_bar_labels(
        data,
        x="metric",
        y="value",
        hue=hue,
        errorbar=errorbar,
        bar_label_rotation=90,
        bar_label_type=bar_label_type,
    )
    ax.tick_params(axis="x", labelrotation=45)

    plt.tight_layout()
    plt.show()


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
