from sklearn.metrics import f1_score, roc_curve, auc, accuracy_score

performance_metrics = {
            'accuracy': lambda actual, pred: accuracy_score(actual, pred, normalize=True),
            'f1': lambda actual, pred: f1_score(actual, pred, average='micro'),
            'auc': lambda actual, pred: auc(roc_curve(actual, pred, pos_label=1)[0],
                                            roc_curve(actual, pred, pos_label=1)[1])
}


def evaluate_performance_metric(performance_metric, pred_y, actual_y):
    """
    evaluates predictions based on the performance metric input
    :param performance_metric:
    :param pred_y:
    :param actual_y:
    :return:
    """
    res = performance_metrics[performance_metric](actual_y, pred_y)
    return res
