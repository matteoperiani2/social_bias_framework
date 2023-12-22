import evaluate
# from nltk.corpus import stopwords
# from nltk import download
# import gensim.downloader as api
import src.models.gpt2 as gpt2
import src.models.bart as bart
from .text_similarity import TextSimilarity
# download('stopwords')


def generate_model_predictions(model, tokenizer, dataloader, split, gen_cfg, config):
    if config.model['name'] == 'gpt2':
        return gpt2.generate_predictions(model, tokenizer, dataloader, split, gen_cfg, config)
    elif config.model['name'] == 'bart':
        return bart.generate_predictions(model, tokenizer, dataloader, split, gen_cfg, config)
    else:
        raise ValueError("Invalid name. Possible values are [gpt2, bart]")


def evaluate_classification(labels, predictions):
    f1 = evaluate.load('f1')
    f1_scores = []
    for lbls, preds in zip(labels, predictions):
        score = f1(lbls[lbls != 2], preds[lbls != 2])
        f1_scores.append(score)

    return f1_scores


def evaluate_generation(labels, predictions, config):
    rouge = evaluate.load('rouge')
    bleu = evaluate.load("bleu")
    text_similarity = TextSimilarity(config['embedding_model'])
    # stop_words = stopwords.words('english')
    # model = api.load('word2vec-google-news-300')

    rouge_scores = []
    bleu_scores = []
    similarity_scores = []
    for lbls, pred in zip(labels, predictions):
        # if post is offensive or it target a group
        if len(lbls) > 0:
            if pred != '':
                r_scores = rouge.compute(pred, lbls)['rougeL']
                b_scores = bleu.compute(pred, lbls)['bleu']
                s_scores = [text_similarity.similarity(pred, lbl) for lbl in lbls]

                # pred_split = pred.lower().split()
                # pred_split = [p for p in pred_split if p not in stop_words]
                # for lbl in lbls:
                #     lbl_split = lbl.lower().split()
                #     lbl_split = [l for l in lbl_split if l not in stop_words]
                #     wmd_score = model.wmdistance(pred_split, lbl_split)

                rouge_scores.append(r_scores)
                bleu_scores.append(b_scores)
                similarity_scores.append(s_scores)
            else:
                rouge_scores.append(0.)
                bleu_scores.append(0.)
                similarity_scores.append(0.)

    return {
        'rouges': rouge_scores,
        'similarities': similarity_scores
    }