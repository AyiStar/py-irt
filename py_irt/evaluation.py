import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


def eval_scores(preds: list[float], responses: list[float | int]):
    
    preds_bin = [int(x > 0.5) for x in preds]
    truth = [int(x) for x in responses]
    acc = accuracy_score(truth, preds_bin)
    try:
        auc = roc_auc_score(truth, preds)
    except ValueError:
        auc = -1.0
    
    return {
        'acc': acc,
        'auc': auc
    }


def eval_scores_per_knowledge(preds: list[float], responses: list[float | int], knowledges: list[int]):
    acc_dict = {}
    auc_dict = {}
    
    preds_dict = {}
    responses_dict = {}
    
    for pred, response, knowledge in zip(preds, responses, knowledges):
        preds_dict.setdefault(knowledge, [])
        responses_dict.setdefault(knowledge, [])
        preds_dict[knowledge].append(pred)
        responses_dict[knowledge].append(response)
    
    for knowledge in preds_dict.keys():
        scores = eval_scores(preds=preds_dict[knowledge], responses=responses_dict[knowledge])
        acc_dict[knowledge] = scores['acc']
        auc_dict[knowledge] = scores['auc']
    
    acc_mean = np.mean([x for x in acc_dict.values()])
    auc_mean = np.mean([x for x in auc_dict.values() if x >= 0])
    return {
        'acc/knowledge': acc_mean,
        'auc/knowledge': auc_mean,
    }