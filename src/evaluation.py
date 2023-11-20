import numpy as np
from rouge import Rouge
from sklearn.metrics import f1_score

from .config import CONFIG

rouge_metric = Rouge()

def evaluate_predictions(tokenizer, predictions, labels):
    class_preds, minority_preds, stereotype_preds = get_predictions(tokenizer, predictions)
    class_labels = np.asarray(labels["class_labels"]).transpose()
    class_preds = np.asarray(class_preds).transpose()

    f1_classifiaction = compute_f1_classification(tokenizer, class_labels, class_preds)
    f1_rouge_minority = compute_f1_generation(labels["minority_labels"], minority_preds)
    f1_rouge_stereotype = compute_f1_generation(labels["stereotype_labels"], stereotype_preds)

    return {
        "class_f1": f1_classifiaction,
        "rouge_minor": f1_rouge_minority,
        "rouge_strtp": f1_rouge_stereotype
    }


def eval_classifications(tokenizer, predictions, labels):
    preds, _, _ = get_predictions(tokenizer, predictions)
    lbls = np.asarray(labels["class_labels"]).transpose()
    preds = np.asarray(preds).transpose()

    f1 = compute_f1_classification(tokenizer, lbls, preds)

    return {
        'predictions': preds,
        'labels': lbls,
        'f1': f1
    }


def get_predictions(tokenizer, predictions):
    class_preds = []
    minority_preds = []
    stereotype_preds = []

    for pred in predictions:
        sep_idx = np.where(pred == tokenizer.sep_token_id)[0]
        eos_idx = np.where(pred == tokenizer.eos_token_id)[0]

        # --- get classification tokens --- 
        if len(eos_idx) > 0:
            # concatenate first 4 tokens with the token generated before the eos
            class_preds.append(np.concatenate((pred[:4], [pred[eos_idx[0]-1]])))
        else:
            # concatenate first 4 tokens with the last generated token
            class_preds.append(np.concatenate((pred[:4], [pred[-1]])))
    
        # --- get minority and stereotype tokens ---
        if len(sep_idx) > 0:
            ## MINORITY
            if len(sep_idx) > 1: 
                # select as minority tokens, those tokens that are between first 2 sep
                minority_preds.append(pred[sep_idx[0]+1:sep_idx[1]])

                ## STEREOTYPE
                if len(sep_idx) > 2:
                    # select as stereotype tokens, those tokens that are between the 2nd and the 3rd sep
                    stereotype_preds.append(pred[sep_idx[1]+1:sep_idx[2]])
                else:
                    # select as stereotype tokens, those tokens that are after the 2nd sep
                    if len(eos_idx) > 0:
                        stereotype_preds.append(pred[sep_idx[1]+1:eos_idx[0]-1])
                    else:
                        stereotype_preds.append(pred[sep_idx[1]+1:len(pred)-1])
            else:
                # select as minority tokens, those tokens that are after the sep
                # for stereotypes no tokens are selected
                if len(eos_idx) > 0:
                    minority_preds.append(pred[sep_idx[0]+1:eos_idx[0]-1])
                else:
                    minority_preds.append(pred[sep_idx[0]+1:len(pred)-1])
                stereotype_preds.append([])
        else:
            # in the case the output is very bad, both minority and stereotypes are discarded
            minority_preds.append([])
            stereotype_preds.append([])


    minority_preds = tokenizer.batch_decode(minority_preds)
    stereotype_preds = tokenizer.batch_decode(stereotype_preds)

    return  class_preds, minority_preds, stereotype_preds


def compute_f1_classification(tokenizer, labels, predictions):
    f1_scores = []
    for lbls, preds in zip(labels, predictions):
        good_idx = [idx for idx,lbl in enumerate(lbls) if lbl != tokenizer.pad_token_id]
        f1 = f1_score([lbls[index] for index in good_idx],
                      [preds[index] for index in good_idx],
                       zero_division=0,
                       average="macro")
        f1_scores.append(np.nan_to_num(f1))
    return f1_scores


def compute_f1_generation(labels, predictions):
    rouge_score = []
    for lbls, preds in zip(labels, predictions):
        if len(lbls) > 0:
            if preds != '':
                lbl_score = []
                for lbl in lbls:
                    r_score = rouge_metric.get_scores(preds, lbl)[0]["rouge-l"]["f"]
                    lbl_score.append(r_score)

                rouge_score.append(np.nan_to_num(np.max(lbl_score)))
            else:
                rouge_score.append(0.)
            
    return rouge_score