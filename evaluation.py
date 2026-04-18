import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)


def evaluate_model(y_true, y_pred, y_pred_prob):
    """Вычисляет основные метрики классификации"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
    }

    # ROC-AUC
    try:
        if len(np.unique(y_true)) > 1 and len(np.unique(y_pred_prob)) > 1:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_prob)
        else:
            metrics['roc_auc'] = 0.5
    except:
        metrics['roc_auc'] = 0.5

    return metrics


def plot_predictions(y_true, y_pred_prob, model_name="Model", ax=None):
    """Строит гистограмму предсказаний"""
    if ax is None:
        ax = plt.gca()

    mask_0 = y_true == 0
    mask_1 = y_true == 1

    if np.any(mask_0):
        ax.hist(y_pred_prob[mask_0], bins=15, alpha=0.7,
                label='Норма (0)', color='green', density=True)
    if np.any(mask_1):
        ax.hist(y_pred_prob[mask_1], bins=15, alpha=0.7,
                label='Усталость (1)', color='red', density=True)

    ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Вероятность усталости')
    ax.set_ylabel('Плотность')
    ax.set_title(f'{model_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_confusion_matrix(y_true, y_pred, model_name="Model", ax=None):
    """Строит матрицу ошибок"""
    if ax is None:
        ax = plt.gca()

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Норма', 'Усталость'])
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(f'Матрица ошибок - {model_name}')


def print_metrics(metrics_dict):
    """Красивый вывод метрик"""
    for name, value in metrics_dict.items():
        print(f"      {name}: {value:.4f}")