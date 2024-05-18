from transformers import AdamW
import textdistance as td

def eval_anls(gold_labels, predictions, tau=0.5, rank=0):
    dis = td.levenshtein.distance(gold_labels.lower(), predictions.lower())
    max_len = max(len(gold_labels), len(predictions))
    if max_len == 0:
        s = 0
    else:
        nl = dis / max_len
        s = 1-nl if nl < tau else 0
    return s

def eval_accuracy(gold_labels, predictions):
    if gold_labels.lower() in predictions.lower():
        return 1.0
    else:
        return 0

def eval_accanls(gold_labels, predictions):
    if gold_labels.lower() in predictions.lower():
        return 1.0
    else:
        return eval_anls(gold_labels, predictions)