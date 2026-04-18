import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import os

from data_simulation import create_synthetic_vocal_dataset, extract_features, balance_dataset
from models import BaseLSTM, PhysicsInformedNN, InterpretableModel
from evaluation import evaluate_model, plot_predictions, print_metrics, plot_confusion_matrix


def main():
    print("=" * 60)
    print("Гибридный анализ временных рядов для прогнозирования усталости голоса")
    print("=" * 60)

    # 1. ПОДГОТОВКА ДАННЫХ
    print("\n[1/5] Генерация датасета...")

    difficulty = 'hard'
    print(f"   Уровень сложности: {difficulty}")

    # Генерируем больше данных для лучшего обучения
    n_singers = 300
    time_steps = 50
    data, labels, metadata = create_synthetic_vocal_dataset(
        n_singers=n_singers,
        time_steps=time_steps,
        fatigue_prob=0.35,  # Увеличиваем долю усталости
        seed=42,
        difficulty=difficulty
    )

    X, y = extract_features(data, labels)
    print(f"   Исходные данные: {X.shape}")
    print(f"   Распределение классов: {np.bincount(y)}")

    # Балансировка данных
    print("\n   Балансировка данных...")
    X_balanced, y_balanced = balance_dataset(X, y, method='smote')
    print(f"   После балансировки: {X_balanced.shape}")
    print(f"   Новое распределение: {np.bincount(y_balanced)}")

    # Разделение на train/test/validation
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.15, random_state=42, stratify=y_balanced
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15, random_state=42, stratify=y_temp
    )

    print(f"\n   Train: {X_train.shape[0]} (норма: {np.sum(y_train == 0)}, усталость: {np.sum(y_train == 1)})")
    print(f"   Val: {X_val.shape[0]} (норма: {np.sum(y_val == 0)}, усталость: {np.sum(y_val == 1)})")
    print(f"   Test: {X_test.shape[0]} (норма: {np.sum(y_test == 0)}, усталость: {np.sum(y_test == 1)})")

    # Вычисляем веса классов
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    print(f"   Веса классов: {class_weight_dict}")

    # Масштабирование
    scaler = StandardScaler()
    original_shape = X_train.shape
    X_train_reshaped = X_train.reshape(-1, original_shape[-1])
    X_val_reshaped = X_val.reshape(-1, original_shape[-1])
    X_test_reshaped = X_test.reshape(-1, original_shape[-1])

    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_val_scaled = scaler.transform(X_val_reshaped)
    X_test_scaled = scaler.transform(X_test_reshaped)

    X_train = X_train_scaled.reshape(X_train.shape)
    X_val = X_val_scaled.reshape(X_val.shape)
    X_test = X_test_scaled.reshape(X_test.shape)

    print("\n[2/5] Данные подготовлены и масштабированы.")

    # 2. ИНИЦИАЛИЗАЦИЯ МОДЕЛЕЙ
    input_shape = (X_train.shape[1], X_train.shape[2])

    lstm_model = BaseLSTM(
        input_shape=input_shape,
        lstm_units=64,
        class_weights=class_weight_dict
    )

    pinn_model = PhysicsInformedNN(
        input_shape=input_shape,
        physics_params={
            'tissue_damping': 0.1 + 0.05 * np.random.random(),
            'muscle_tension': 0.8 + 0.1 * np.random.random(),
            'vocal_fold_length': 1.2,
            'subglottal_pressure': 0.5
        },
        class_weights=class_weight_dict
    )

    interpretable_model = InterpretableModel(
        model_type='random_forest',
        class_weight='balanced'
    )

    models = {
        'LSTM': lstm_model,
        'Physics-Informed NN': pinn_model,
        'Interpretable Baseline (RF)': interpretable_model
    }

    # 3. ОБУЧЕНИЕ
    print("\n[3/5] Обучение моделей...")
    history = {}

    for name, model in models.items():
        print(f"\n   Обучение модели: {name}...")

        if 'LSTM' in name or 'Physics' in name:
            hist = model.train(
                X_train, y_train,
                X_val=X_val, y_val=y_val,
                epochs=200,
                batch_size=32,  # Увеличим batch size
                verbose=0
            )
            history[name] = hist
            print(f"   ✓ {name} обучена")

            # Показываем финальные метрики на валидации
            if hist and 'val_precision' in hist:
                print(f"      Val precision: {hist['val_precision'][-1]:.4f}")
                print(f"      Val recall: {hist['val_recall'][-1]:.4f}")
        else:
            model.train(X_train, y_train)
            print(f"   ✓ {name} обучена")

    # 4. ОЦЕНКА
    print("\n[4/5] Оценка моделей на тестовых данных...")
    results = {}

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for i, (name, model) in enumerate(models.items()):
        print(f"\n   Оценка {name}:")

        if 'LSTM' in name or 'Physics' in name:
            y_pred_prob = model.predict(X_test)
            y_pred = (y_pred_prob > 0.5).astype(int)

            # Дополнительная информация для Physics-Informed модели
            if name == 'Physics-Informed NN':
                print(f"      Распределение вероятностей:")
                print(f"         Мин: {np.min(y_pred_prob):.3f}")
                print(f"         Макс: {np.max(y_pred_prob):.3f}")
                print(f"         Среднее: {np.mean(y_pred_prob):.3f}")
                print(f"         Предсказания >0.5: {np.sum(y_pred_prob > 0.5)} из {len(y_pred_prob)}")
        else:
            y_pred = model.predict(X_test)
            y_pred_prob = model.predict_proba(X_test)

        metrics = evaluate_model(y_test, y_pred, y_pred_prob)
        results[name] = metrics
        print_metrics(metrics)

        plot_predictions(y_test, y_pred_prob, model_name=name, ax=axes[0, i])
        plot_confusion_matrix(y_test, y_pred, model_name=name, ax=axes[1, i])

    plt.tight_layout()
    plt.savefig('model_comparison_balanced.png', dpi=150)
    plt.show()

    # 5. РЕЗУЛЬТАТЫ
    print("\n" + "=" * 60)
    print("[5/5] ИТОГОВЫЕ МЕТРИКИ:")
    print("=" * 60)

    results_df = pd.DataFrame(results).T
    print("\n", results_df.round(3))

    results_df.to_csv('experiment_results_final.csv')
    print("\nРезультаты сохранены в 'experiment_results_final.csv'")
    print("Графики сохранены в 'model_comparison_balanced.png'")


if __name__ == "__main__":
    main()