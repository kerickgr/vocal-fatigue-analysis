# data_simulation.py (сбалансированная версия)

import numpy as np
import random
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek


def create_synthetic_vocal_dataset(n_singers=200, time_steps=50, fatigue_prob=0.4, seed=42, difficulty='hard'):
    """
    Создает синтетический датасет с балансировкой классов
    """
    np.random.seed(seed)
    random.seed(seed)

    data = []
    labels = []

    n_features = 20

    for singer_id in range(n_singers):
        # Индивидуальные особенности
        base_f0 = np.random.uniform(120, 250)
        base_jitter = np.random.uniform(0.5, 1.5)
        base_shimmer = np.random.uniform(2.0, 4.0)
        base_hnr = np.random.uniform(18, 25)

        # Увеличиваем вероятность усталости для баланса
        adjusted_fatigue_prob = min(0.5, fatigue_prob * 1.5)
        fatigue_level = np.random.beta(2, 3) if np.random.random() < adjusted_fatigue_prob else np.random.beta(5, 2)

        # Метка с более четкой границей
        is_fatigued = fatigue_level > 0.55

        singer_data = []
        for t in range(time_steps):
            fatigue_progress = t / time_steps

            if difficulty == 'hard':
                # Индивидуальные паттерны усталости
                if singer_id % 4 == 0:
                    # Тип 1: Явные признаки усталости
                    f0_factor = 1 - 0.25 * fatigue_level * fatigue_progress
                    jitter_factor = 1 + 3.0 * fatigue_level * fatigue_progress
                    shimmer_factor = 1 + 2.5 * fatigue_level * fatigue_progress
                    hnr_factor = 1 - 0.3 * fatigue_level * fatigue_progress
                elif singer_id % 4 == 1:
                    # Тип 2: Компенсаторные механизмы
                    f0_factor = 1 + 0.15 * fatigue_level * fatigue_progress
                    jitter_factor = 1 + 2.0 * fatigue_level * fatigue_progress
                    shimmer_factor = 1 + 2.2 * fatigue_level * fatigue_progress
                    hnr_factor = 1 - 0.2 * fatigue_level * fatigue_progress
                elif singer_id % 4 == 2:
                    # Тип 3: Резкое ухудшение
                    if t > time_steps * 0.7:
                        f0_factor = 1 - 0.4 * fatigue_level
                        jitter_factor = 1 + 4.0 * fatigue_level
                        shimmer_factor = 1 + 3.5 * fatigue_level
                        hnr_factor = 1 - 0.5 * fatigue_level
                    else:
                        f0_factor = 1 - 0.05 * fatigue_level * fatigue_progress
                        jitter_factor = 1 + 1.0 * fatigue_level * fatigue_progress
                        shimmer_factor = 1 + 1.0 * fatigue_level * fatigue_progress
                        hnr_factor = 1 - 0.1 * fatigue_level * fatigue_progress
                else:
                    # Тип 4: Смешанный тип
                    f0_factor = 1 - 0.1 * fatigue_level * fatigue_progress + 0.1 * np.sin(2 * np.pi * t / 30)
                    jitter_factor = 1 + 1.8 * fatigue_level * fatigue_progress + 0.2 * np.random.random()
                    shimmer_factor = 1 + 1.6 * fatigue_level * fatigue_progress + 0.2 * np.random.random()
                    hnr_factor = 1 - 0.15 * fatigue_level * fatigue_progress + 0.1 * np.random.random()
            else:
                # Более простые зависимости для других уровней сложности
                f0_factor = 1 - 0.15 * fatigue_level * fatigue_progress
                jitter_factor = 1 + 2.0 * fatigue_level * fatigue_progress
                shimmer_factor = 1 + 1.8 * fatigue_level * fatigue_progress
                hnr_factor = 1 - 0.2 * fatigue_level * fatigue_progress

            # Добавляем шум
            f0 = base_f0 * (f0_factor + np.random.normal(0, 0.06))
            jitter = base_jitter * (jitter_factor + np.random.normal(0, 0.12))
            shimmer = base_shimmer * (shimmer_factor + np.random.normal(0, 0.1))
            hnr = base_hnr * (hnr_factor + np.random.normal(0, 0.07))

            # MFCC
            mfccs = np.random.normal(
                loc=0.7 * fatigue_level * fatigue_progress if is_fatigued else 0,
                scale=1.0 + 0.3 * fatigue_level,
                size=n_features - 4
            )

            feature_vector = np.hstack([f0, jitter, shimmer, hnr, mfccs])
            singer_data.append(feature_vector)

        data.append(np.array(singer_data))
        labels.append(1 if is_fatigued else 0)

    X = np.array(data)
    y = np.array(labels)

    print(f"Сгенерировано: норма={np.sum(y == 0)}, усталость={np.sum(y == 1)}")

    return X, y, {"n_singers": n_singers, "time_steps": time_steps, "difficulty": difficulty}


def balance_dataset(X, y, method='smote'):
    """
    Балансировка датасета с помощью SMOTE
    """
    from sklearn.utils import resample

    n_samples = X.shape[0]
    n_timesteps = X.shape[1]
    n_features = X.shape[2]

    # Преобразуем 3D в 2D для SMOTE
    X_2d = X.reshape(n_samples, -1)

    if method == 'smote':
        try:
            smote = SMOTE(random_state=42, sampling_strategy='auto')
            X_resampled, y_resampled = smote.fit_resample(X_2d, y)
            print(f"SMOTE балансировка: {np.bincount(y)} -> {np.bincount(y_resampled)}")
        except:
            # Если SMOTE не работает, используем простой oversampling
            print("SMOTE не сработал, используем oversampling")
            X_resampled, y_resampled = simple_oversample(X_2d, y)
    else:
        # Простой oversampling для меньшинства
        X_resampled, y_resampled = simple_oversample(X_2d, y)

    # Возвращаем в 3D
    X_resampled = X_resampled.reshape(-1, n_timesteps, n_features)

    return X_resampled, y_resampled


def simple_oversample(X, y):
    """Простой oversampling для класса меньшинства"""
    from sklearn.utils import resample

    X_normal = X[y == 0]
    X_fatigue = X[y == 1]

    # Увеличиваем количество записей с усталостью
    n_samples_needed = len(X_normal) - len(X_fatigue)
    if n_samples_needed > 0:
        X_fatigue_upsampled = resample(
            X_fatigue,
            replace=True,
            n_samples=len(X_normal),
            random_state=42
        )
        X_balanced = np.vstack([X_normal, X_fatigue_upsampled])
        y_balanced = np.hstack([np.zeros(len(X_normal)), np.ones(len(X_normal))])
    else:
        X_balanced = X
        y_balanced = y

    return X_balanced, y_balanced


def extract_features(data, labels=None):
    """Извлечение признаков"""
    X = np.array(data)
    if labels is not None:
        y = np.array(labels)
        return X, y
    return X