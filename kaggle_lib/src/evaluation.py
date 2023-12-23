from collections import Counter
import evaluate
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from .text_similarity import TextSimilarity
from rouge import Rouge
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.corpus import stopwords

def evaluate_classification(labels, predictions, cls_columns):
    f1 = evaluate.load("f1")
    f1_scores = dict()
    for lbls, preds, cols in zip(labels, predictions, cls_columns, strict=True):
        score = f1.compute(predictions=preds[lbls != -1], references=lbls[lbls != -1])['f1']
        f1_scores[cols] = score

    return f1_scores


def evaluate_generation(data, config):
    rouge = Rouge(metrics=["rouge-l"], stats='f')

    # rouge = evaluate.load("rouge")
    # bleu = evaluate.load("bleu")
    # similarity = TextSimilarity(config["embedding_model"])
    # stop_words = stopwords.words('english')
    # model = api.load('word2vec-google-news-300')

    params = {
        'rouge': rouge,
        'similarity': None
    }

    data = data.map(
        compute_scores,
        load_from_cache_file=False,
        fn_kwargs=params,
        batched=False
    )
    
    return data


def compute_scores(data, rouge, similarity):
    group_scores = {}
    stereotype_score = {}

    if data['group'] != None and data['group_preds'] != '':
        group_scores['rouge'] = [rouge.get_scores(data['group_preds'], lbl)[0]['rouge-l']['f'] for lbl in data['group'] if lbl is not None]
        group_scores['bleu'] = [
            corpus_bleu([[lbl]],
                        [data['group_preds']],
                        weights=(0.5, 0.5),
                        smoothing_function=SmoothingFunction().method1
            ) for lbl in data['group'] if lbl is not None 
        ]
    else:
        group_scores['rouge'] = None

    if data['stereotype'] != None and data['stereotype_preds'] != '':
        stereotype_score['rouge'] = [rouge.get_scores(data['stereotype_preds'], lbl)[0]['rouge-l']['f'] for lbl in data['stereotype'] if lbl is not None]
        stereotype_score['bleu'] = [
            corpus_bleu([[lbl]],
                        [data['stereotype_preds']],
                        weights=(0.5, 0.5),
                        smoothing_function=SmoothingFunction().method1
            ) for lbl in data['stereotype'] if lbl is not None 
        ]

    else:
        stereotype_score['rouge'] = None

    return {
        'group_scores': group_scores,
        'stereotype_scores': stereotype_score
    }


def print_classification_results(
    labels, predictions, results, show_cm=True
):
    annotation_type = [
        "Offensive",
        "Intentional",
        "Sex/Lewd content",
        "Group targetted",
        "Speaker in group",
    ]

    for type, score in zip(annotation_type, results.values(), strict=True):
        print(f"{type}: {score:.3f}")

    if show_cm:
        plt.rcParams["font.size"] = "12"
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


def print_generations_results(
    f1_minorities,
    f1_stereotypes,
    min_labels,
    min_pred,
    sterotype_labels,
    sterotype_preds,
    show_dist=True,
):
    print(f"Minority Rouge-L F1 score: {f1_minorities:.3f}")
    print(f"Stereotype Rouge-L F1 score: {f1_stereotypes:.3f}")
    print()
    if show_dist:
        plt.rcParams["font.size"] = "8"
        plot_minority_distribution(
            min_pred, min_labels, sterotype_labels, sterotype_preds
        )
        plt.show()


def plot_minority_distribution(min_pred, min_labels, sterotype_labels, sterotype_preds):
    type = ["Minority", "Stereotype"]
    name = ["predictions", "labels"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for i, (data, ax) in enumerate(
        zip(
            [min_pred, min_labels, sterotype_labels, sterotype_preds],
            axes.ravel(),
            strict=True,
        )
    ):
        _plot_word_bar(data, ax=ax)
        ax.set_title(f"{type[i//2]} {name[i%2]}")


def _plot_word_bar(data, ax, n_words=10):
    if isinstance(data[0], list):
        topic_words = [y.lower() if y != "" else "''" for x in data for y in x]
    else:
        topic_words = [x.lower() if x != "" else "''" for x in data]
    word_count_dict = dict(Counter(topic_words))
    popular_words = sorted(word_count_dict, key=word_count_dict.get, reverse=True)
    popular_words_nonstop = [
        w for w in popular_words if w not in stopwords.words("english")
    ]
    total = sum([word_count_dict[w] for w in reversed(popular_words_nonstop)])
    ax.barh(
        range(n_words),
        [
            word_count_dict[w] / total
            for w in reversed(popular_words_nonstop[0:n_words])
        ],
    )
    ax.set_yticks(
        [x + 0.5 for x in range(n_words)], reversed(popular_words_nonstop[0:n_words])
    )
    for i in ax.containers:
        ax.bar_label(i, padding=2)
