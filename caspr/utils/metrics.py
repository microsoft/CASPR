"""Get classification metrics for CASPR models."""

# coding: utf-8
import logging

from sklearn.metrics import auc, classification_report, confusion_matrix, precision_recall_curve, roc_auc_score

logger = logging.getLogger(__name__)

def check_topk_values_if_churn(k, preds, y):
    """Check how many of top k churn predictions actually churned."""

    pred_arr = preds.cpu()
    pred_arr = pred_arr.detach().numpy()
    topk = pred_arr.argsort()[-k:][::-1]
    count = 0
    for ind in topk:
        if y[ind] == 1:
            count += 1
    return count


def pr_auc_score(y_true, y_score):
    """Get pr_auc score."""

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    return pr_auc


def get_metrics(y_true, y_score, threshold=0.5, digits=3):
    """Get classification report, confusion matrix, roc_auc score, and pr_auc score."""

    y_pred = y_score > threshold

    report = classification_report(y_true, y_pred, digits=digits)
    report_dict = convert_classification_report_to_dict(report)
    logger.info(report)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    report_dict.update({'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn})
    logger.info("tp: {}, fp: {}, tn: {}, fn: {}".format(tp, fp, tn, fn))

    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc = pr_auc_score(y_true, y_score)
    report_dict.update({'roc_auc_score': roc_auc, 'pr_auc_score': pr_auc})
    logger.info("roc_auc_score = {:.4f}, pr_auc_score = {:.4f}".format(roc_auc, pr_auc))

    return report_dict


def convert_classification_report_to_dict(report):
    """Convert classification report to Dict format."""

    rows = [row.split() for row in report.split('\n') if row]
    headers = rows[0]
    report_dict = {}
    for row in rows[1:]:
        if row[1] == 'avg':
            label, scores = ' '.join(row[:2]), row[2:]
        else:
            label, scores = row[0], row[1:]

        if label == 'accuracy':
            report_dict[label] = float(scores[-2])
        else:
            report_dict[label] = dict(zip(headers, [float(score) for score in scores[:-1]] + [int(scores[-1])]))
    return report_dict
