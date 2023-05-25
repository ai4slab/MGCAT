from sklearn import metrics


def calc_metrics(y_true, y_score):
    auc = metrics.roc_auc_score(y_true, y_score)
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_score)
    au_prc = metrics.auc(recall, precision)
    y_pred = [0 if i < 0.5 else 1 for i in y_score]
    acc = metrics.accuracy_score(y_true, y_pred)
    pre = metrics.precision_score(y_true, y_pred)
    rec = metrics.recall_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred)
    print(auc, au_prc, acc, pre, rec, f1)
    return auc, au_prc, acc, pre, rec, f1
