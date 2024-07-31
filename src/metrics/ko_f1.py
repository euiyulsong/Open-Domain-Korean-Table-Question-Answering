from .preprocess import normalize_answer
import numpy as np
from collections import Counter
from konlpy.tag import Mecab

def _f1_score(prediction, ground_truth, tokenizer):
    prediction_tokens = tokenizer.morphs(normalize_answer(prediction))
    ground_truth_tokens = tokenizer.morphs(normalize_answer(ground_truth))
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def f1_score(prediction, ground_truth):
    if isinstance(prediction, str):
        prediction = [prediction]
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]
    y_pred = np.asarray(prediction).flatten()
    y_true = np.asarray(ground_truth).flatten()
    tokenizer = Mecab()
    return sum([_f1_score(prediction, label, tokenizer) for prediction, label in zip(y_pred, y_true)]) / len(y_true)