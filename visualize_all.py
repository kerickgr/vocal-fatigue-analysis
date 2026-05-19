"""
visualize_all.py — Полный скрипт визуализации для НИР
======================================================
Запускается ПОСЛЕ main.py (нужны: experiment_results.csv, model_comparison.png)
Генерирует 6 публикационных рисунков:

  fig1_dataset_overview.png     — обзор датасета и 4 паттерна усталости
  fig2_feature_dynamics.png     — динамика акустических признаков по времени
  fig3_titze_calibration.png    — калибровка TitzeBaseline + физический скор
  fig4_model_comparison.png     — вероятности + матрицы ошибок (4 модели)
  fig5_roc_curves.png           — ROC-кривые всех моделей на одном графике
  fig6_feature_importance.png   — важность признаков RF + интерпретация
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, Rectangle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay

# ── импорт модулей проекта ──────────────────────────────────────────────────
from data_simulation import create_synthetic_vocal_dataset, extract_features, balance_dataset
from models import TitzeBaseline, BaseLSTM, PhysicsGuidedLSTM, InterpretableModel

# ── глобальный стиль ───────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':        11,
    'axes.titlesize':   13,
    'axes.labelsize':   11,
    'xtick.labelsize':  10,
    'ytick.labelsize':  10,
    'legend.fontsize':  10,
    'figure.dpi':       150,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'grid.linestyle':   '--',
})

COLORS = {
    'normal':  '#2196F3',   # синий
    'fatigue': '#F44336',   # красный
    'titze':   '#9C27B0',   # фиолетовый
    'rf':      '#4CAF50',   # зелёный
    'lstm':    '#FF9800',   # оранжевый
    'pinn':    '#F44336',   # красный
}

# ══════════════════════════════════════════════════════════════════════════════
# ШАГ 0: генерация данных и обучение моделей
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Подготовка данных и обучение моделей для визуализации...")
print("=" * 60)

# --- данные ---
data_raw, labels_raw, _ = create_synthetic_vocal_dataset(
    n_singers=300, time_steps=50, fatigue_prob=0.35, seed=42, difficulty='hard'
)
X_all, y_all = extract_features(data_raw, labels_raw)
X_bal, y_bal = balance_dataset(X_all, y_all, method='smote')

X_temp, X_test, y_temp, y_test = train_test_split(
    X_bal, y_bal, test_size=0.15, random_state=42, stratify=y_bal
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15, random_state=42, stratify=y_temp
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_val_s   = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
X_test_s  = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
cw_dict = {i: cw[i] for i in range(len(cw))}

input_shape = (X_train_s.shape[1], X_train_s.shape[2])

# --- модели ---
print("\n[1/4] TitzeBaseline...")
titze = TitzeBaseline(physics_params={
    'tissue_damping': 0.1, 'muscle_tension': 0.8,
    'vocal_fold_length': 1.2, 'subglottal_pressure': 0.5,
})
titze.fit(X_train_s, y_train)

print("[2/4] Random Forest...")
rf = InterpretableModel(model_type='random_forest', class_weight='balanced')
rf.train(X_train_s, y_train)

print("[3/4] LSTM...")
lstm = BaseLSTM(input_shape=input_shape, lstm_units=64, class_weights=cw_dict)
lstm.train(X_train_s, y_train, X_val=X_val_s, y_val=y_val, epochs=200, batch_size=32, verbose=0)

print("[4/4] PINN...")
pinn = PhysicsGuidedLSTM(
    input_shape=input_shape,
    physics_params={'tissue_damping': 0.10, 'muscle_tension': 0.80,
                    'vocal_fold_length': 1.2, 'subglottal_pressure': 0.5},
    class_weights=cw_dict
)
pinn.train(X_train_s, y_train, X_val=X_val_s, y_val=y_val, epochs=200, batch_size=32, verbose=0)

# --- предсказания ---
prob_titze = titze.predict_proba(X_test_s)
pred_titze = titze.predict(X_test_s)

prob_rf    = rf.predict_proba(X_test_s)
pred_rf    = rf.predict(X_test_s)

prob_lstm  = lstm.predict(X_test_s)
pred_lstm  = (prob_lstm > 0.5).astype(int)

prob_pinn  = pinn.predict(X_test_s)
pred_pinn  = (prob_pinn > 0.5).astype(int)

models_data = {
    'TitzeBaseline': (pred_titze, prob_titze, COLORS['titze']),
    'Random Forest': (pred_rf,    prob_rf,    COLORS['rf']),
    'LSTM':          (pred_lstm,  prob_lstm,  COLORS['lstm']),
    'PINN/PhysicsGuidedLSTM': (pred_pinn,  prob_pinn,  COLORS['pinn']),
}

print("\nМодели готовы. Строим визуализации...\n")


# ══════════════════════════════════════════════════════════════════════════════
# РИС. 1 — Обзор датасета: 4 паттерна усталости + распределение
# ══════════════════════════════════════════════════════════════════════════════
print("[Рис.1] Обзор датасета...")

pattern_names = [
    'Тип 0: Явные признаки\n(монотонная деградация)',
    'Тип 1: Компенсаторный\n(рост F0 → затем падение)',
    'Тип 2: Резкое ухудшение\n(порог на 70% ряда)',
    'Тип 3: Смешанный\n(комбинация + шум)',
]

fig1, axes = plt.subplots(2, 4, figsize=(18, 8))
fig1.suptitle(
    'Рисунок 1 — Обзор синтетического датасета:\nчетыре паттерна накопления вокальной усталости',
    fontsize=14, fontweight='bold', y=1.01
)

feat_labels = ['F0 (Гц)', 'Jitter (%)', 'Shimmer (%)', 'HNR (dB)']
feat_colors = ['#1565C0', '#AD1457', '#2E7D32', '#E65100']

# Верхний ряд: паттерны F0 для каждого типа (норма vs усталость)
for pid in range(4):
    ax = axes[0, pid]
    # Берём по 3 вокалиста каждого типа (номера: pid, pid+4, pid+8)
    for label_val, linestyle, alpha in [(0, '--', 0.6), (1, '-', 0.9)]:
        candidates = [i for i in range(len(labels_raw))
                      if labels_raw[i] == label_val and i % 4 == pid][:3]
        for idx in candidates:
            series = data_raw[idx][:, 0]   # F0
            color  = COLORS['normal'] if label_val == 0 else COLORS['fatigue']
            ax.plot(series, color=color, linestyle=linestyle, alpha=alpha, linewidth=1.5)
    ax.set_title(pattern_names[pid], fontsize=10)
    ax.set_xlabel('Временной шаг')
    ax.set_ylabel('F0 (Гц)' if pid == 0 else '')
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0],[0], color=COLORS['normal'],  linestyle='--', label='Норма'),
        Line2D([0],[0], color=COLORS['fatigue'], linestyle='-',  label='Усталость'),
    ], fontsize=9)

# Нижний ряд: динамика 4 признаков — средние по всему датасету
t = np.arange(50)
for fi, (fname, fcolor) in enumerate(zip(feat_labels, feat_colors)):
    ax = axes[1, fi]
    norm_idx    = np.where(y_all == 0)[0]
    fatigue_idx = np.where(y_all == 1)[0]

    norm_mean    = np.mean(X_all[norm_idx, :, fi],    axis=0)
    fatigue_mean = np.mean(X_all[fatigue_idx, :, fi], axis=0)
    norm_std     = np.std(X_all[norm_idx, :, fi],     axis=0)
    fatigue_std  = np.std(X_all[fatigue_idx, :, fi],  axis=0)

    ax.plot(t, norm_mean,    color=COLORS['normal'],  linewidth=2, label='Норма')
    ax.plot(t, fatigue_mean, color=COLORS['fatigue'], linewidth=2, label='Усталость')
    ax.fill_between(t, norm_mean - norm_std,    norm_mean + norm_std,
                    color=COLORS['normal'],  alpha=0.15)
    ax.fill_between(t, fatigue_mean - fatigue_std, fatigue_mean + fatigue_std,
                    color=COLORS['fatigue'], alpha=0.15)
    ax.set_title(f'Динамика: {fname}', fontsize=10)
    ax.set_xlabel('Временной шаг')
    ax.set_ylabel(fname if fi == 0 else '')
    ax.legend(fontsize=9)

plt.tight_layout()
fig1.savefig('fig1_dataset_overview.png', dpi=150, bbox_inches='tight')
plt.close(fig1)
print("   → fig1_dataset_overview.png")


# ══════════════════════════════════════════════════════════════════════════════
# РИС. 2 — Акустические признаки: boxplot норма vs усталость
# ══════════════════════════════════════════════════════════════════════════════
print("[Рис.2] Распределение признаков...")

feat_names_full = [
    'F0 (Гц)', 'Jitter (%)', 'Shimmer (%)', 'HNR (dB)',
    'MFCC-1', 'MFCC-2', 'MFCC-3', 'MFCC-4'
]

fig2, axes2 = plt.subplots(2, 4, figsize=(18, 7))
fig2.suptitle(
    'Рисунок 2 — Распределение акустических признаков: норма vs усталость\n'
    '(средние значения по временному ряду, тестовая выборка)',
    fontsize=13, fontweight='bold', y=1.01
)

# Используем исходные (немасштабированные) данные для читаемого violin plot
norm_idx    = np.where(y_test == 0)[0]
fatigue_idx = np.where(y_test == 1)[0]

for fi in range(8):
    ax = axes2.flat[fi]
    feat_idx = fi
    vals_norm    = np.mean(X_test[norm_idx,    :, feat_idx], axis=1)
    vals_fatigue = np.mean(X_test[fatigue_idx, :, feat_idx], axis=1)

    parts = ax.violinplot(
        [vals_norm, vals_fatigue],
        positions=[1, 2], showmedians=True, showextrema=True
    )
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(COLORS['normal'] if i == 0 else COLORS['fatigue'])
        pc.set_alpha(0.6)
    parts['cmedians'].set_colors(['white', 'white'])
    parts['cmedians'].set_linewidth(2)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Норма', 'Усталость'])
    ax.set_title(feat_names_full[fi], fontsize=10)
    if fi % 4 == 0:
        ax.set_ylabel('Норм. значение')

plt.tight_layout()
fig2.savefig('fig2_feature_dynamics.png', dpi=150, bbox_inches='tight')
plt.close(fig2)
print("   → fig2_feature_dynamics.png")


# ══════════════════════════════════════════════════════════════════════════════
# РИС. 3 — TitzeBaseline: физический скор + калибровка порога
# ══════════════════════════════════════════════════════════════════════════════
print("[Рис.3] Titze Baseline анализ...")

fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5))
fig3.suptitle(
    'Рисунок 3 — Анализ порогового классификатора TitzeBaseline\n'
    '(модель Titze 2006: пороговое давление фонации как индикатор усталости)',
    fontsize=13, fontweight='bold'
)

scores = titze.predict_proba(X_test_s)

# Панель A: распределение физического скора
ax = axes3[0]
ax.hist(scores[y_test == 0], bins=20, alpha=0.7, color=COLORS['normal'],
        label='Норма', density=True, edgecolor='white')
ax.hist(scores[y_test == 1], bins=20, alpha=0.7, color=COLORS['fatigue'],
        label='Усталость', density=True, edgecolor='white')
ax.axvline(titze.threshold_, color='black', linestyle='--', linewidth=2,
           label=f'Порог = {titze.threshold_:.2f}')
ax.set_xlabel('Физический скор усталости S')
ax.set_ylabel('Плотность')
ax.set_title('A. Распределение физического скора')
ax.legend()

# Панель B: калибровочная кривая (F1 vs порог)
ax = axes3[1]
thresholds = np.linspace(0.05, 0.95, 91)
f1_scores  = []
for thr in thresholds:
    pred = (scores >= thr).astype(int)
    tp = np.sum((pred == 1) & (y_test == 1))
    fp = np.sum((pred == 1) & (y_test == 0))
    fn = np.sum((pred == 0) & (y_test == 1))
    p  = tp / (tp + fp + 1e-9)
    r  = tp / (tp + fn + 1e-9)
    f1 = 2 * p * r / (p + r + 1e-9)
    f1_scores.append(f1)
ax.plot(thresholds, f1_scores, color=COLORS['titze'], linewidth=2)
ax.axvline(titze.threshold_, color='black', linestyle='--', linewidth=2,
           label=f'Оптимальный порог = {titze.threshold_:.2f}')
ax.set_xlabel('Порог классификации')
ax.set_ylabel('F1-score')
ax.set_title('B. Калибровочная кривая (F1 vs порог)')
ax.legend()

# Панель C: вклад компонентов физического скора
ax = axes3[2]
comp_labels = ['norm(Pth)\n×0.35', 'norm(jitter)\n×0.30',
               'norm(shimmer)\n×0.25', 'norm(−ΔHNR)\n×0.10']
weights = [0.35, 0.30, 0.25, 0.10]
comp_colors = ['#7B1FA2', '#C62828', '#2E7D32', '#E65100']

f0_mean     = np.mean(X_test_s[:, :, 0], axis=1) + 1e-6
xi, c_val   = 0.8 + 0.1, 0.5 + 0.3
pth_raw     = (0.1 * c_val * 0.5) / (2.0 * 1.2 * f0_mean * xi)
jitter_raw  = np.mean(X_test_s[:, :, 1], axis=1)
shimmer_raw = np.mean(X_test_s[:, :, 2], axis=1)
hnr_trend   = np.mean(np.diff(X_test_s[:, :, 3], axis=1), axis=1)

def norm01(v):
    return (v - v.min()) / (v.max() - v.min() + 1e-9)

components_fatigue = [
    np.mean(norm01(pth_raw)[y_test == 1]),
    np.mean(norm01(jitter_raw)[y_test == 1]),
    np.mean(norm01(shimmer_raw)[y_test == 1]),
    np.mean(norm01(-hnr_trend)[y_test == 1]),
]
components_normal = [
    np.mean(norm01(pth_raw)[y_test == 0]),
    np.mean(norm01(jitter_raw)[y_test == 0]),
    np.mean(norm01(shimmer_raw)[y_test == 0]),
    np.mean(norm01(-hnr_trend)[y_test == 0]),
]

x_pos  = np.arange(len(comp_labels))
width  = 0.35
bars_n = ax.bar(x_pos - width/2, [w * v for w, v in zip(weights, components_normal)],
                width, label='Норма', color=COLORS['normal'], alpha=0.8, edgecolor='white')
bars_f = ax.bar(x_pos + width/2, [w * v for w, v in zip(weights, components_fatigue)],
                width, label='Усталость', color=COLORS['fatigue'], alpha=0.8, edgecolor='white')
ax.set_xticks(x_pos)
ax.set_xticklabels(comp_labels, fontsize=9)
ax.set_ylabel('Взвешенный вклад в скор S')
ax.set_title('C. Вклад компонентов формулы Titze')
ax.legend()

plt.tight_layout()
fig3.savefig('fig3_titze_calibration.png', dpi=150, bbox_inches='tight')
plt.close(fig3)
print("   → fig3_titze_calibration.png")


# ══════════════════════════════════════════════════════════════════════════════
# РИС. 4 — Сравнение моделей: вероятности + матрицы ошибок
# ══════════════════════════════════════════════════════════════════════════════
print("[Рис.4] Вероятности + матрицы ошибок...")

model_list = list(models_data.items())
n_m = len(model_list)

fig4, axes4 = plt.subplots(2, n_m, figsize=(5 * n_m, 9))
fig4.suptitle(
    'Рисунок 4 — Сравнительный анализ четырёх моделей\n'
    'Верхний ряд: распределение предсказанных вероятностей. '
    'Нижний ряд: матрицы ошибок',
    fontsize=13, fontweight='bold'
)

for i, (name, (preds, probs, color)) in enumerate(model_list):
    # Гистограмма вероятностей
    ax = axes4[0, i]
    ax.hist(probs[y_test == 0], bins=15, alpha=0.7, color=COLORS['normal'],
            label='Норма', density=True, edgecolor='white')
    ax.hist(probs[y_test == 1], bins=15, alpha=0.7, color=COLORS['fatigue'],
            label='Усталость', density=True, edgecolor='white')
    ax.axvline(0.5, color='black', linestyle='--', alpha=0.7, linewidth=1.5)

    from sklearn.metrics import accuracy_score, recall_score, f1_score
    acc = accuracy_score(y_test, preds)
    rec = recall_score(y_test, preds, zero_division=0)
    f1  = f1_score(y_test, preds, zero_division=0)
    ax.set_title(f'{name}\nAcc={acc:.3f}  Recall={rec:.3f}  F1={f1:.3f}', fontsize=10)
    ax.set_xlabel('P(усталость)')
    ax.set_ylabel('Плотность' if i == 0 else '')
    ax.legend(fontsize=9)

    # Матрица ошибок
    ax2 = axes4[1, i]
    cm  = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Норма', 'Усталость'])
    disp.plot(ax=ax2, cmap='Blues', values_format='d', colorbar=False)
    ax2.set_title(f'Матрица ошибок — {name}', fontsize=10)
    # Аннотации ошибок
    tn, fp, fn, tp = cm.ravel()
    ax2.text(0.5, -0.18, f'TN={tn}  FP={fp}  FN={fn}  TP={tp}',
             ha='center', transform=ax2.transAxes, fontsize=9, color='#555555')

plt.tight_layout()
fig4.savefig('fig4_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close(fig4)
print("   → fig4_model_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# РИС. 5 — ROC-кривые всех моделей
# ══════════════════════════════════════════════════════════════════════════════
print("[Рис.5] ROC-кривые...")

fig5, axes5 = plt.subplots(1, 2, figsize=(14, 6))
fig5.suptitle(
    'Рисунок 5 — ROC-кривые и сравнение AUC\n'
    '(нарастание сложности: пороговое правило → гибридная нейросеть)',
    fontsize=13, fontweight='bold'
)

ax_roc  = axes5[0]
ax_bar  = axes5[1]

auc_vals = {}
for name, (preds, probs, color) in models_data.items():
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_test, probs)
        fpr, tpr, _ = roc_curve(y_test, probs)
        ax_roc.plot(fpr, tpr, color=color, linewidth=2.5,
                    label=f'{name}  (AUC = {auc:.3f})')
        auc_vals[name] = auc
    except Exception as e:
        print(f"   ROC для {name}: {e}")

ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=1, label='Случайный классификатор')
ax_roc.set_xlabel('False Positive Rate (1 − Specificity)')
ax_roc.set_ylabel('True Positive Rate (Recall / Sensitivity)')
ax_roc.set_title('A. ROC-кривые')
ax_roc.legend(loc='lower right', fontsize=9)
ax_roc.fill_between([0, 1], [0, 1], alpha=0.04, color='grey')

# Столбчатая диаграмма AUC
names  = list(auc_vals.keys())
aucs   = [auc_vals[n] for n in names]
colors = [models_data[n][2] for n in names]
bars   = ax_bar.barh(names, aucs, color=colors, edgecolor='white', height=0.5)
ax_bar.set_xlim(0.5, 1.05)
ax_bar.axvline(1.0, color='black', linestyle='--', alpha=0.4)
ax_bar.set_xlabel('ROC-AUC')
ax_bar.set_title('B. Сравнение AUC по моделям')
for bar, auc in zip(bars, aucs):
    ax_bar.text(auc + 0.005, bar.get_y() + bar.get_height()/2,
                f'{auc:.3f}', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
fig5.savefig('fig5_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close(fig5)
print("   → fig5_roc_curves.png")


# ══════════════════════════════════════════════════════════════════════════════
# РИС. 6 — Важность признаков Random Forest
# ══════════════════════════════════════════════════════════════════════════════
print("[Рис.6] Важность признаков...")

fig6 = plt.figure(figsize=(18, 8))
fig6.suptitle(
    'Рисунок 6 — Анализ важности признаков (Random Forest)\n'
    'Топ-20 признаков + суммарный вклад групп акустических параметров',
    fontsize=13, fontweight='bold'
)
gs6 = gridspec.GridSpec(1, 2, figure=fig6, width_ratios=[1.6, 1], wspace=0.35)
ax_a = fig6.add_subplot(gs6[0])
ax_b = fig6.add_subplot(gs6[1])

if hasattr(rf, 'feature_importance') and rf.feature_importance is not None:
    importances = rf.feature_importance
else:
    importances = rf.model.feature_importances_

# Имена признаков (14 статистик × 20 каналов)
stat_names = [
    'mean', 'std', 'p10', 'p25', 'p50', 'p75', 'p90', 'range',
    'abs_diff_mean', 'diff_std', 'max_abs_diff', 'pos_frac', 'neg_frac', 'slope'
]
channel_names = ['F0', 'Jitter', 'Shimmer', 'HNR'] + [f'MFCC-{i}' for i in range(1, 17)]
feat_names_all = [f'{ch}_{st}' for ch in channel_names for st in stat_names]

n_feats = min(len(importances), len(feat_names_all))
imp_series = pd.Series(importances[:n_feats], index=feat_names_all[:n_feats])
top20 = imp_series.sort_values(ascending=False).head(20)

channel_color_map = {
    'F0':     '#1565C0',
    'Jitter': '#AD1457',
    'Shimmer':'#2E7D32',
    'HNR':    '#E65100',
    'MFCC':   '#6A1B9A',
}

def feat_color(fname):
    for key in channel_color_map:
        if fname.startswith(key):
            return channel_color_map[key]
    return '#607D8B'

bar_colors = [feat_color(n) for n in top20.index]

# ── Панель A: горизонтальный bar chart топ-20 ─────────────────────────────
bars_a = ax_a.barh(range(len(top20)), top20.values,
                   color=bar_colors, edgecolor='white', height=0.7)
ax_a.set_yticks(range(len(top20)))
ax_a.set_yticklabels(top20.index, fontsize=9.5)
ax_a.invert_yaxis()
ax_a.set_xlabel('Важность признака', fontsize=11)
ax_a.set_title('A. Топ-20 признаков по важности', fontsize=11, pad=10)
ax_a.grid(axis='x', alpha=0.3, linestyle='--')
ax_a.set_axisbelow(True)

# Подписи значений справа от баров
for bar in bars_a:
    ax_a.text(bar.get_width() + top20.values.max() * 0.01,
              bar.get_y() + bar.get_height() / 2,
              f'{bar.get_width():.4f}', va='center', fontsize=8, color='#333333')

# Цветная легенда параметров
from matplotlib.patches import Patch
legend_els = [Patch(facecolor=v, label=k, linewidth=0)
              for k, v in channel_color_map.items()]
ax_a.legend(handles=legend_els, title='Акустический\nпараметр',
            loc='lower right', fontsize=9, title_fontsize=9,
            framealpha=0.9, edgecolor='#cccccc')

# ── Панель B: горизонтальный grouped bar — суммарная важность по группам ──
group_importance = {}
for ch in channel_names:
    key = ch if not ch.startswith('MFCC') else 'MFCC'
    mask = imp_series.index.str.startswith(ch)
    group_importance[key] = group_importance.get(key, 0) + imp_series[mask].sum()

# Сортируем по убыванию
groups  = sorted(group_importance.items(), key=lambda x: -x[1])
g_names = [g[0] for g in groups]
g_vals  = [g[1] for g in groups]
g_cols  = [channel_color_map[g[0]] for g in groups]
total   = sum(g_vals)

y_pos = np.arange(len(g_names))
bars_b = ax_b.barh(y_pos, g_vals, color=g_cols, edgecolor='white', height=0.55)
ax_b.set_yticks(y_pos)
ax_b.set_yticklabels(g_names, fontsize=11, fontweight='bold')
ax_b.invert_yaxis()
ax_b.set_xlabel('Суммарная важность', fontsize=11)
ax_b.set_title('B. Вклад групп параметров (%)', fontsize=11, pad=10)
ax_b.grid(axis='x', alpha=0.3, linestyle='--')
ax_b.set_axisbelow(True)

# Подпись: абсолютное значение + процент
for bar, val in zip(bars_b, g_vals):
    pct = val / total * 100
    ax_b.text(bar.get_width() + total * 0.01,
              bar.get_y() + bar.get_height() / 2,
              f'{pct:.1f}%  ({val:.3f})',
              va='center', fontsize=10, fontweight='bold', color='#222222')

ax_b.set_xlim(0, max(g_vals) * 1.45)

fig6.savefig('fig6_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close(fig6)
print("   → fig6_feature_importance.png")


# ══════════════════════════════════════════════════════════════════════════════
# ИТОГ
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Все визуализации построены:")
for fname in [
    'fig1_dataset_overview.png',
    'fig2_feature_dynamics.png',
    'fig3_titze_calibration.png',
    'fig4_model_comparison.png',
    'fig5_roc_curves.png',
    'fig6_feature_importance.png',
]:
    import os
    size_kb = os.path.getsize(fname) // 1024 if os.path.exists(fname) else 0
    print(f"   {fname}  ({size_kb} KB)")
print("=" * 60)
