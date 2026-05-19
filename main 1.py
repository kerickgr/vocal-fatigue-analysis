"""
Гибридный анализ временных рядов акустических параметров вокала
для прогнозирования усталости голосового аппарата

Порядок моделей (нарастание сложности):
  1. Titze Threshold Baseline  — пороговое правило на основе модели Titze 2006
  2. Random Forest             — интерпретируемый ансамблевый метод
  3. LSTM                      — неинтерпретируемая нейросеть (рекуррентная)
  4. PhysicsGuidedLSTM       — гибридная модель (LSTM + физиологические параметры)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import os

from data_simulation import create_synthetic_vocal_dataset, extract_features, balance_dataset
from models import TitzeBaseline, BaseLSTM, PhysicsGuidedLSTM, InterpretableModel
from evaluation import evaluate_model, plot_predictions, print_metrics, plot_confusion_matrix


def main():
    print("=" * 65)
    print("Гибридный анализ временных рядов для прогнозирования усталости голоса")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. ПОДГОТОВКА ДАННЫХ
    # ------------------------------------------------------------------
    print("\n[1/5] Генерация датасета...")

    n_singers   = 300
    time_steps  = 50
    difficulty  = 'hard'

    data, labels, metadata = create_synthetic_vocal_dataset(
        n_singers=n_singers,
        time_steps=time_steps,
        fatigue_prob=0.35,
        seed=42,
        difficulty=difficulty
    )

    X, y = extract_features(data, labels)
    print(f"   Исходные данные: {X.shape}")
    print(f"   Распределение классов: норма={np.sum(y==0)}, усталость={np.sum(y==1)}")

    # Балансировка SMOTE
    print("\n   Балансировка данных (SMOTE)...")
    X_balanced, y_balanced = balance_dataset(X, y, method='smote')
    print(f"   После балансировки: {X_balanced.shape}")
    print(f"   Новое распределение: норма={np.sum(y_balanced==0)}, усталость={np.sum(y_balanced==1)}")

    # Разделение train / val / test  (70 / 15 / 15)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.15, random_state=42, stratify=y_balanced
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15, random_state=42, stratify=y_temp
    )

    print(f"\n   Train : {X_train.shape[0]}  (норма={np.sum(y_train==0)}, усталость={np.sum(y_train==1)})")
    print(f"   Val   : {X_val.shape[0]}  (норма={np.sum(y_val==0)},  усталость={np.sum(y_val==1)})")
    print(f"   Test  : {X_test.shape[0]}  (норма={np.sum(y_test==0)},  усталость={np.sum(y_test==1)})")

    # Веса классов
    class_weights_arr = class_weight.compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train
    )
    class_weight_dict = {i: class_weights_arr[i] for i in range(len(class_weights_arr))}
    print(f"   Веса классов: {class_weight_dict}")

    # Масштабирование
    scaler = StandardScaler()
    orig_shape = X_train.shape

    X_train_s = scaler.fit_transform(X_train.reshape(-1, orig_shape[-1])).reshape(X_train.shape)
    X_val_s   = scaler.transform(X_val.reshape(-1, orig_shape[-1])).reshape(X_val.shape)
    X_test_s  = scaler.transform(X_test.reshape(-1, orig_shape[-1])).reshape(X_test.shape)

    print("\n[2/5] Данные подготовлены и масштабированы.")

    # ------------------------------------------------------------------
    # 2. ИНИЦИАЛИЗАЦИЯ МОДЕЛЕЙ  (нарастание сложности)
    # ------------------------------------------------------------------
    input_shape = (X_train_s.shape[1], X_train_s.shape[2])

    titze_model = TitzeBaseline(
        physics_params={
            'tissue_damping':      0.1,
            'muscle_tension':      0.8,
            'vocal_fold_length':   1.2,
            'subglottal_pressure': 0.5,
        }
    )

    rf_model = InterpretableModel(
        model_type='random_forest',
        class_weight='balanced'
    )

    lstm_model = BaseLSTM(
        input_shape=input_shape,
        lstm_units=64,
        class_weights=class_weight_dict
    )

    pinn_model = PhysicsGuidedLSTM(
        input_shape=input_shape,
        physics_params={
            'tissue_damping':      0.10 + 0.05 * np.random.random(),
            'muscle_tension':      0.80 + 0.10 * np.random.random(),
            'vocal_fold_length':   1.2,
            'subglottal_pressure': 0.5,
        },
        class_weights=class_weight_dict
    )

    # Упорядоченный словарь: от простого к сложному
    models = {
        'Titze Baseline':        titze_model,
        'Random Forest':         rf_model,
        'LSTM':                  lstm_model,
        'PhysicsGuidedLSTM':     pinn_model,
    }

    # ------------------------------------------------------------------
    # 3. ОБУЧЕНИЕ
    # ------------------------------------------------------------------
    print("\n[3/5] Обучение моделей (нарастание сложности)...")
    history = {}

    for name, model in models.items():
        print(f"\n   [{name}]")
        if name == 'Titze Baseline':
            # Пороговое правило — обучения не требует
            model.fit(X_train_s, y_train)
            print("   ✓ Правило Titze инициализировано (обучение не требуется)")

        elif name == 'Random Forest':
            model.train(X_train_s, y_train)
            print("   ✓ Random Forest обучен")

        else:
            hist = model.train(
                X_train_s, y_train,
                X_val=X_val_s, y_val=y_val,
                epochs=200,
                batch_size=32,
                verbose=0
            )
            history[name] = hist
            if hist and 'val_recall' in hist:
                print(f"   ✓ {name} обучена  "
                      f"(val_recall={hist['val_recall'][-1]:.3f}, "
                      f"val_precision={hist['val_precision'][-1]:.3f})")
            else:
                print(f"   ✓ {name} обучена")

    # ------------------------------------------------------------------
    # 4. ОЦЕНКА
    # ------------------------------------------------------------------
    print("\n[4/5] Оценка моделей на тестовых данных...")
    results = {}
    n_models = len(models)

    fig, axes = plt.subplots(2, n_models, figsize=(5 * n_models, 10))
    fig.suptitle("Сравнительный анализ моделей (нарастание сложности)", fontsize=14, y=1.01)

    for i, (name, model) in enumerate(models.items()):
        print(f"\n   [{name}]")

        if name == 'Titze Baseline':
            y_pred      = model.predict(X_test_s)
            y_pred_prob = model.predict_proba(X_test_s)

        elif name == 'Random Forest':
            y_pred      = model.predict(X_test_s)
            y_pred_prob = model.predict_proba(X_test_s)

        else:
            y_pred_prob = model.predict(X_test_s)
            y_pred      = (y_pred_prob > 0.5).astype(int)
            if name == 'PhysicsGuidedLSTM':
                print(f"      Вероятности — мин: {np.min(y_pred_prob):.3f}, "
                      f"макс: {np.max(y_pred_prob):.3f}, "
                      f"среднее: {np.mean(y_pred_prob):.3f}")

        metrics = evaluate_model(y_test, y_pred, y_pred_prob)
        results[name] = metrics
        print_metrics(metrics)

        plot_predictions(y_test, y_pred_prob, model_name=name, ax=axes[0, i])
        plot_confusion_matrix(y_test, y_pred,  model_name=name, ax=axes[1, i])

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    print("\n   График сохранён: model_comparison.png")

    # ------------------------------------------------------------------
    # 5. ИТОГОВЫЕ МЕТРИКИ
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("[5/5] ИТОГОВЫЕ МЕТРИКИ (нарастание сложности):")
    print("=" * 65)

    results_df = pd.DataFrame(results).T
    print("\n", results_df.round(3))
    results_df.to_csv('experiment_results.csv')

    # Краткий вывод: достаточна ли точность
    pinn_acc = results.get('PhysicsGuidedLSTM', {}).get('accuracy', 0)
    pinn_rec = results.get('PhysicsGuidedLSTM', {}).get('recall', 0)
    print("\n--- Интерпретация результатов ---")
    if pinn_acc >= 0.95 and pinn_rec == 1.0:
        print("   ✓ Гибридная модель PINN достигла высокой точности на синтетических данных.")
        print("   ✓ Recall = 1.0: все случаи усталости обнаружены (нет ложноотрицательных).")
        print("   → Прототип подтверждён. Следующий шаг: апробация на реальных данных вокалистов.")
    else:
        print("   ⚠ Точность ниже ожидаемой. Требуется анализ данных и архитектуры.")

    print("\nРезультаты сохранены в 'experiment_results.csv'")
    print("Графики сохранены в 'model_comparison.png'")


if __name__ == "__main__":
    main()
