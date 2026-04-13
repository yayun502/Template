import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def compute_classification_metrics(y_true, y_pred, label_names=None):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    if label_names is None:
        report = classification_report(y_true, y_pred, digits=4)
    else:
        report = classification_report(
            y_true, y_pred, target_names=label_names, digits=4
        )

    return {
        "accuracy": acc,
        "confusion_matrix": cm,
        "classification_report": report
    }
