"""
Простой загрузчик для готовых записей вокалистов
"""

import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import argparse


def extract_features_from_file(audio_path, duration=3.0):
    """
    Извлечение признаков из одного аудиофайла
    """
    try:
        # Загружаем аудио
        y, sr = librosa.load(audio_path, sr=22050, duration=duration)

        # 1. F0 (основная частота)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=65, fmax=1047,
            sr=sr, hop_length=512
        )
        f0 = np.nan_to_num(f0)
        f0_mean = np.mean(f0[f0 > 0]) if np.any(f0 > 0) else 0
        f0_std = np.std(f0[f0 > 0]) if np.any(f0 > 0) else 0

        # 2. Jitter (упрощенно)
        if len(f0[f0 > 0]) > 1:
            f0_clean = f0[f0 > 0]
            jitter = np.mean(np.abs(np.diff(f0_clean)) / f0_clean[:-1]) * 100
        else:
            jitter = 0.5

        # 3. Shimmer (упрощенно)
        rms = librosa.feature.rms(y=y, hop_length=512)[0]
        if len(rms) > 1:
            shimmer = np.mean(np.abs(np.diff(rms)) / (rms[:-1] + 1e-10)) * 100
        else:
            shimmer = 2.0

        # 4. HNR (упрощенно)
        try:
            autocorr = librosa.autocorrelate(y)
            hnr = 10 * np.log10(np.max(autocorr) / (np.mean(np.abs(autocorr)) + 1e-10))
        except:
            hnr = 20.0

        # 5. MFCC (13 коэффициентов)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=512)
        mfccs_mean = np.mean(mfccs, axis=1)

        # Собираем все признаки в один вектор
        features = np.concatenate([
            [f0_mean, f0_std, jitter, shimmer, hnr],
            mfccs_mean
        ])

        return features

    except Exception as e:
        print(f"   Ошибка при обработке {audio_path}: {e}")
        return None


def load_vocal_data(data_path, time_steps=50):
    """
    Загрузка всех записей из структурированных папок

    Parameters:
    -----------
    data_path : str
        Путь к корневой папке с подпапками normal/ и fatigue/
    time_steps : int
        Сколько временных шагов использовать (сколько файлов на вокалиста)

    Returns:
    --------
    X, y : numpy arrays
        Данные и метки
    """
    X = []
    y = []
    singer_names = []

    # Проверяем существование папок
    normal_path = os.path.join(data_path, 'normal')
    fatigue_path = os.path.join(data_path, 'fatigue')

    if not os.path.exists(normal_path):
        print(f"ОШИБКА: Папка {normal_path} не найдена!")
        return np.array([]), np.array([])

    if not os.path.exists(fatigue_path):
        print(f"ОШИБКА: Папка {fatigue_path} не найдена!")
        return np.array([]), np.array([])

    # Загружаем нормальные голоса (метка 0)
    print("\nЗагрузка нормальных голосов (до нагрузки)...")
    for singer_folder in os.listdir(normal_path):
        singer_dir = os.path.join(normal_path, singer_folder)
        if not os.path.isdir(singer_dir):
            continue

        print(f"  Обработка вокалиста: {singer_folder}")

        # Получаем все аудиофайлы
        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.m4a', '*.flac']:
            audio_files.extend(list(Path(singer_dir).glob(ext)))

        if not audio_files:
            print(f"    Нет аудиофайлов в {singer_folder}")
            continue

        # Сортируем по имени (предполагаем хронологический порядок)
        audio_files.sort(key=lambda x: str(x))

        # Берем первые time_steps файлов
        singer_features = []
        for audio_file in audio_files[:time_steps]:
            print(f"    Обработка: {audio_file.name}")
            features = extract_features_from_file(str(audio_file))
            if features is not None:
                singer_features.append(features)

        # Если получили достаточно признаков
        if len(singer_features) >= 5:  # минимум 5 файлов
            # Дополняем или обрезаем до time_steps
            while len(singer_features) < time_steps:
                singer_features.append(singer_features[-1])  # дублируем последний

            X.append(np.array(singer_features[:time_steps]))
            y.append(0)
            singer_names.append(singer_folder)
            print(f"    ✓ Добавлено: {len(singer_features)} временных шагов")

    # Загружаем усталые голоса (метка 1)
    print("\nЗагрузка усталых голосов (после нагрузки)...")
    for singer_folder in os.listdir(fatigue_path):
        singer_dir = os.path.join(fatigue_path, singer_folder)
        if not os.path.isdir(singer_dir):
            continue

        print(f"  Обработка вокалиста: {singer_folder}")

        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.m4a', '*.flac']:
            audio_files.extend(list(Path(singer_dir).glob(ext)))

        if not audio_files:
            print(f"    Нет аудиофайлов в {singer_folder}")
            continue

        audio_files.sort(key=lambda x: str(x))

        singer_features = []
        for audio_file in audio_files[:time_steps]:
            print(f"    Обработка: {audio_file.name}")
            features = extract_features_from_file(str(audio_file))
            if features is not None:
                singer_features.append(features)

        if len(singer_features) >= 5:
            while len(singer_features) < time_steps:
                singer_features.append(singer_features[-1])

            X.append(np.array(singer_features[:time_steps]))
            y.append(1)
            singer_names.append(singer_folder)
            print(f"    ✓ Добавлено: {len(singer_features)} временных шагов")

    print(f"\nИТОГО: загружено {len(X)} вокалистов")
    print(f"  - Норма: {sum(1 for label in y if label == 0)}")
    print(f"  - Усталость: {sum(1 for label in y if label == 1)}")
    print(f"  - Формат данных: {np.array(X).shape if X else 'нет данных'}")

    return np.array(X), np.array(y)


def main():
    parser = argparse.ArgumentParser(description='Загрузка готовых записей вокалистов')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Путь к папке с данными (normal/fatigue подпапки)')
    parser.add_argument('--time_steps', type=int, default=50,
                        help='Количество временных шагов')

    args = parser.parse_args()

    X, y = load_vocal_data(args.data_path, args.time_steps)

    if len(X) > 0:
        # Сохраняем в numpy формат для быстрой загрузки
        np.save('X_vocal_data.npy', X)
        np.save('y_vocal_data.npy', y)
        print(f"\nДанные сохранены в X_vocal_data.npy и y_vocal_data.npy")
        print(f"Для загрузки используйте: X = np.load('X_vocal_data.npy')")
    else:
        print("\nДанные не загружены!")


if __name__ == "__main__":
    main()