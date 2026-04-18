"""
Обучение моделей на реальных данных вокалистов
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import os

from models import BaseLSTM, PhysicsInformedNN, InterpretableModel
from evaluation import evaluate_model, plot_predictions, print_metrics, plot_confusion_matrix


def main():
    print("=" * 60)
    print("Гибридный анализ на РЕАЛЬНЫХ данных вокалистов")
    print("=" * 60)

    # 1. ЗАГРУЗКА ДАННЫХ
    print("\n[1/5] Загрузка данных...")

    # Проверяем наличие сохраненных .npy файлов
    if os.path.exists('X_vocal_data.npy') and os.path.exists('y_vocal_data.npy'):
        X = np.load('X_vocal_data.npy')
        y = np.load('y_vocal_data.npy')
        print(f"   Загружены сохраненные данные: {X.shape}")
    else:
        # Если нет, используем синтетические для демонстрации
        print("   Реальные данные не найдены, использую синтетические")
        from data_simulation import create_synthetic_vocal_dataset
        data, labels, _ = create_synthetic_vocal_dataset(
            n_singers=50, time_steps=50, fatigue_prob=0.4
        )
        X, y = data, labels

    print(f"   Данные: {X.shape}")
    print(f"   Распределение классов: {np.bincount(y)}")

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Масштабирование
    scaler = StandardScaler()
    original_shape = X_train.shape
    X_train_reshaped = X_train.reshape(-1, original_shape[-1])
    X_test_reshaped = X_test.reshape(-1, original_shape[-1])

    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_test_scaled = scaler.transform(X_test_reshaped)

    X_train = X_train_scaled.reshape(X_train.shape)
    X_test = X_test_scaled.reshape(X_test.shape)

    print("\n[2/5] Данные подготовлены")

    # 2. МОДЕЛИ
    input_shape = (X_train.shape[1], X_train.shape[2])

    # Веса классов для балансировки
    if len(np.unique(y_train)) > 1:
        class_weights = class_weight.compute_class_weight(
            'balanced', classes=np.unique(y_train), y=y_train
        )
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    else:
        class_weight_dict = {0: 1.0, 1: 1.0}

    models = {
        'LSTM': BaseLSTM(input_shape=input_shape, class_weights=class_weight_dict),
        'PINN': PhysicsInformedNN(
            input_shape=input_shape,
            physics_params={'tissue_damping': 0.1, 'muscle_tension': 0.8},
            class_weights=class_weight_dict
        ),
        'Random Forest': InterpretableModel()
    }

    # 3. ОБУЧЕНИЕ
    print("\n[3/5] Обучение моделей...")
    for name, model in models.items():
        print(f"   {name}...")
        if name != 'Random Forest':
            model.train(X_train, y_train, epochs=100, batch_size=16, verbose=0)
        else:
            model.train(X_train, y_train)
        print(f"   ✓ {name} готова")

    # 4. ОЦЕНКА
    print("\n[4/5] Оценка моделей...")
    results = {}

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for i, (name, model) in enumerate(models.items()):
        if name != 'Random Forest':
            y_pred_prob = model.predict(X_test)
            y_pred = (y_pred_prob > 0.5).astype(int)
        else:
            y_pred = model.predict(X_test)
            y_pred_prob = model.predict_proba(X_test)

        metrics = evaluate_model(y_test, y_pred, y_pred_prob)
        results[name] = metrics

        plot_predictions(y_test, y_pred_prob, model_name=name, ax=axes[0, i])
        plot_confusion_matrix(y_test, y_pred, model_name=name, ax=axes[1, i])

    plt.tight_layout()
    plt.savefig('real_data_results.png', dpi=150)
    plt.show()

    # 5. РЕЗУЛЬТАТЫ
    print("\n" + "=" * 60)
    print("[5/5] РЕЗУЛЬТАТЫ:")
    print("=" * 60)

    results_df = pd.DataFrame(results).T
    print("\n", results_df.round(3))
    results_df.to_csv('real_data_results.csv')


if __name__ == "__main__":
    main()