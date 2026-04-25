# 🎙️ Анализ вокальной усталости с использованием гибридных нейросетевых моделей

## 📌 Описание проекта

Вокальная усталость представляет собой серьезную проблему для
специалистов голосоречевых профессий (вокалисты, преподаватели, дикторы
и др.). Традиционные методы диагностики опираются на субъективные оценки
или статические акустические параметры, что ограничивает возможность
отслеживания динамики состояния.

В данной работе предложен гибридный подход к анализу временных рядов
акустических параметров голоса, включающий:

-   Частоту основного тона (F0)
-   Jitter
-   Shimmer
-   Harmonics-to-Noise Ratio (HNR)
-   Mel-frequency cepstral coefficients (MFCC)

Ключевая особенность --- интеграция биофизической модели
голосообразования Titze в архитектуру Physics-Informed Neural Network
(PINN).

## 🧠 Архитектура моделей

В работе реализованы и сравнены три модели:

-   LSTM --- базовая рекуррентная модель для временных рядов\
-   PINN (гибридная модель)\
-   Random Forest --- интерпретируемая модель

## 📊 Результаты

Эксперименты проведены на сбалансированном синтетическом датасете (322
образца):

  Модель          Точность   Полнота
  --------------- ---------- ---------
  LSTM            89.8%      100%
  Random Forest   98%        100%
  PINN            100%       100%

Ключевые признаки: - jitter - shimmer - F0

## ⚙️ Требования

-   Python 3.8+
-   ≥ 4 GB RAM

## 🚀 Установка

python -m venv venv

Windows: venv`\Scripts`{=tex}`\activate`{=tex}

Linux/macOS: source venv/bin/activate

pip install numpy==1.24.3 pip install pandas==2.0.3 pip install
scikit-learn==1.3.0 pip install tensorflow==2.13.0 pip install
matplotlib==3.7.2 pip install imbalanced-learn==0.11.0

## ▶️ Быстрый запуск

python main.py

Результаты: - experiment_results_final.csv -
model_comparison_balanced.png

## 🎧 Работа с реальными данными

pip install librosa pip install soundfile pip install sounddevice

Структура: ваши_записи/ ├── normal/ └── fatigue/

Загрузка: python load_existing_data.py --data_path ./ваши_записи
--time_steps 50

Обучение: python main_with_real.py

## ⚠️ Возможные проблемы

librosa: pip install numba==0.57.0 pip install librosa==0.10.0

soundfile (Linux): sudo apt-get install libsndfile1

## 📌 Перспективы

-   реальные данные
-   онлайн мониторинг
-   мобильные приложения
