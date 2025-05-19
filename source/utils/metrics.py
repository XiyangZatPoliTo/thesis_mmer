from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from torch import nn


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


import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

@torch.no_grad()
def calculate_metrics(model_or_func,
                      dataloader,
                      device,
                      # audio_model,
                      # video_model,
                      mode="acc"):
    """
    兼容 nn.Module和lambda函数两种形式：
    model_or_func(audio, video) -> logits
    """
    is_module = isinstance(model_or_func, nn.Module)
    if is_module:
        model_or_func.eval()

    # audio_model.eval()
    # video_model.eval()
    all_preds = []
    all_labels = []

    for audio, video, labels in dataloader:
        audio, video, labels = audio.to(device), video.to(device), labels.to(device)
        # audio_feat = audio_model(audio)
        # video_feat = video_model(video)

        outputs = model_or_func(audio, video)

        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    if mode == "acc":
        return accuracy_score(all_labels, all_preds), None
    elif mode == "f1":
        return f1_score(all_labels, all_preds, average='weighted'), None
    elif mode == "all":
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        return acc, {"precision": prec, "recall": recall, "f1": f1}
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def print_classification_report(y_true, y_pred, target_names=None):
    """
    Print a full classification report.

    Args:
        y_true (list or array): Ground truth labels
        y_pred (list or array): Predicted labels
        target_names (list): Optional label names
    """
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
