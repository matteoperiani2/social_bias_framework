import numpy as np
from src.plot import plot_classification_cm
from .text_similarity import TextSimilarity

from sklearn.metrics import f1_score
from rouge import Rouge
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity

# import nltk
# from nltk.tokenize import word_tokenize
# from gensim.models import KeyedVectors
# from nltk.corpus import stopwords
# nltk.download('stopwords')
# nltk.download('punkt')

def evaluate_classification(labels, predictions, cls_columns):
    f1_scores = dict()
    for lbls, preds, cols in zip(labels, predictions, cls_columns, strict=True):
        score = f1_score(lbls[lbls != -1], preds[lbls != -1], average='binary')
        f1_scores[cols] = score

    return f1_scores


def evaluate_generation(data, config):
    rouge = Rouge(metrics=["rouge-l"], stats='f')
    # stop_words = set(stopwords.words('english'))    
    # word_vectors = KeyedVectors.load_word2vec_format(config.wmd_model, binary=True)
    similarity = TextSimilarity(config.embedding_model)

    # all_groups_or_minorities = set()
    # for cols in ['group', 'stereotype', 'group_preds', 'stereotype_preds']:
    #     all_groups_or_minorities.update(*[v for v in data[cols] if v is not None and v != ''])

    # emeddings = dict(zip(all_groups_or_minorities, similarity.generate_embeddings(all_groups_or_minorities)))

    params = {
        'rouge': rouge,
        'emeddings': None
    }

    data = data.map(
        compute_generative_scores,
        load_from_cache_file=False,
        fn_kwargs=params,
        batched=False
    )
    
    return data


def compute_generative_scores(data, rouge, emeddings):
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
        # group_scores['similarity'] = [
        #     cosine_similarity(emeddings[data['group_preds']], emeddings[lbl])
        #     for lbl in data['group'] if lbl is not None 
        # ]
        # group_scores['wmd'] = [
        #     word_vectors.wmdistance(
        #         [token for token in data['group_preds'].lower().split() if token not in stop_words],
        #         [token for token in lbl.lower().split() if token not in stop_words]
        #     ) for lbl in data['group'] if lbl is not None 
        # ]
    else:
        group_scores['rouge'] = None
        group_scores['bleu'] = None
        # stereotype_score['similarity'] = None
        # group_scores['wmd'] = None

    if data['stereotype'] != None and data['stereotype_preds'] != '':
        stereotype_score['rouge'] = [rouge.get_scores(data['stereotype_preds'], lbl)[0]['rouge-l']['f'] for lbl in data['stereotype'] if lbl is not None]
        stereotype_score['bleu'] = [
            corpus_bleu([[lbl]],
                        [data['stereotype_preds']],
                        weights=(0.5, 0.5),
                        smoothing_function=SmoothingFunction().method1
            ) for lbl in data['stereotype'] if lbl is not None 
        ]
        # stereotype_score['similarity'] = [
        #     cosine_similarity(emeddings[data['stereotype_preds']], emeddings[lbl])
        #     for lbl in data['stereotype'] if lbl is not None 
        # ]
        # stereotype_score['wmd'] = [
        #     word_vectors.wmdistance(
        #         [token for token in data['stereotype_preds'].lower().split() if token not in stop_words],
        #         [token for token in lbl.lower().split() if token not in stop_words]
        #     ) for lbl in data['stereotype'] if lbl is not None 
        # ]
    else:
        stereotype_score['rouge'] = None
        stereotype_score['bleu'] = None
        # stereotype_score['similarity'] = None
        # stereotype_score['wmd'] = None

    return {
        'group_scores': group_scores,
        'stereotype_scores': stereotype_score
    }


def aggregate_generation_results(group_scores, stereotype_scores):
    group_rouge_scores = [max(scores['rouge']) for scores in group_scores if scores['rouge'] != None]
    group_bleu_score = [max(scores['bleu']) for scores in group_scores if scores['bleu'] != None]
    # group_sim_score = [max(scores['similarity']) for scores in group_scores if scores['similarity'] != None]
    # group_wmd_score = [min(scores['wmd']) for scores in group_scores if scores['bleu'] != None]

    stereotype_rouge_score = [max(scores['rouge']) for scores in stereotype_scores if scores['rouge'] != None]
    stereotype_bleu_score = [max(scores['bleu']) for scores in stereotype_scores if scores['bleu'] != None]
    # stereotype__sim_score = [max(scores['similarity']) for scores in stereotype_scores if scores['similarity'] != None]
    # stereotype_wmd_score = [min(scores['wmd']) for scores in stereotype_scores if scores['bleu'] != None]

    return {
        'group_rouge': np.mean(group_rouge_scores),
        'group_bleu': np.mean(group_bleu_score),
        'stereotype_rouge': np.mean(stereotype_rouge_score),
        'stereotype_bleu': np.mean(stereotype_bleu_score),
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
        plot_classification_cm(labels, predictions, annotation_type)


def print_generations_results(results):
    for score_name, score in results.items():
        s_class, s_type = score_name.split('_')
        print(f"{s_class.title()} {s_type.title()} score:{score:.3f}")