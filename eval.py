import textdistance as td

def eval_anls(gold_labels, predictions, tau=0.5, rank=0):
    max_s = 0
    for gold_label in gold_labels:
        dis = td.levenshtein.distance(gold_label.lower(), predictions.lower())
        max_len = max(len(gold_label), len(predictions))
        if max_len == 0:
            s = 0
        else:
            nl = dis / max_len
            s = 1-nl if nl < tau else 0
        if s > max_s:
            max_s = s
    return max_s

def eval_accuracy(gold_labels, predictions):
    max_s = 0
    for gold_label in gold_labels:
        if gold_label.lower() in predictions.lower():
            max_s = 1.0
            break
    return max_s    

def eval_accanls(gold_labels, predictions):
    if eval_accuracy(gold_labels, predictions) > 0:
        return eval_accuracy(gold_labels, predictions)
    else:
        return eval_anls(gold_labels, predictions)
