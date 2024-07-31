from .preprocess import normalize_answer
import numpy as np

def exact_match_score(prediction, ground_truth):
    if isinstance(prediction, str):
        prediction = [prediction]
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]
    prediction = [normalize_answer(i) for i in prediction]
    ground_truth = [normalize_answer(i) for i in ground_truth]
    y_pred = np.asarray(prediction).flatten()
    y_true = np.asarray(ground_truth).flatten()
    return sum(y_pred == y_true) / len(y_true)
