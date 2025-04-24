from sklearn.metrics import precision_score, recall_score, f1_score, classification_report


def compute_metrics(y_true, y_pred, average="macro"):
    """
    Compute precision, recall, and F1-score.

    Args:
        y_true (list or array): Ground truth labels
        y_pred (list or array): Predicted labels
        average (str): "macro", "micro", or "weighted"

    Returns:
        dict: Dictionary with precision, recall, and F1
    """
    return {
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0)
    }


def print_classification_report(y_true, y_pred, target_names=None):
    """
    Print a full classification report.

    Args:
        y_true (list or array): Ground truth labels
        y_pred (list or array): Predicted labels
        target_names (list): Optional label names
    """
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
