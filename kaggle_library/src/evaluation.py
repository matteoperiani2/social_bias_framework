import numpy as np
from sklearn.metrics import f1_score
from rouge import Rouge
import models.gpt2 as gpt2
import models.bart as bart
from .text_similarity import TextSimilarity


def generate_model_predictions(model, tokenizer, dataloader, split, gen_cfg, config):
    if config.model['name'] == 'gpt2':
        return gpt2.generate_predictions(model, tokenizer, dataloader, split, gen_cfg, config)
    elif config.model['name'] == 'bart':
        return bart.generate_predictions(model, tokenizer, dataloader, split, gen_cfg, config)
    else:
        raise ValueError("Invalid name. Possible values are [gpt2, bart]")


def evaluate_classification(labels, predictions):
    f1_scores = []
    for lbls, preds in zip(labels, predictions):
        mask_not_present_labels = lbls != 2
        preds = preds[mask_not_present_labels]
        f1 = f1_score(lbls[mask_not_present_labels],
                      preds[mask_not_present_labels],
                      average="binary")
        f1_scores.append(f1)

    return f1_scores


def evaluate_generation(labels, predictions, config):  
    rouge_metric = Rouge()
    text_similarity = TextSimilarity(config['embedding_model'])
    rouge_scores = []
    similarity_scores = []
    for lbls, pred in zip(labels, predictions):
        # if post is offensive or it target a group
        if len(lbls) > 0:
            if pred != '':
                r_scores = []
                sim_scores = []
                for lbl in lbls:
                    r_score = rouge_metric.get_scores(pred, lbl)[0]["rouge-l"]["f"]
                    s_score = text_similarity.similarity(pred, lbl)
                    r_scores.append(r_score)
                    sim_scores.append(s_score)
                rouge_scores.append(np.nan_to_num(r_scores))
                similarity_scores.append(sim_scores)
            else:
                rouge_scores.append(0.)
                similarity_scores.append(0.)

    return {
        'rouges': rouge_scores,
        'similarities': similarity_scores
    }