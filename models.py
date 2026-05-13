import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Dropout, Input, Concatenate, BatchNormalization, Bidirectional
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import keras.backend as K


class TitzeBaseline:
    """
    Пороговый классификатор на основе биофизической модели Titze (2006).

    Логика: вычисляет нормированное пороговое давление фонации Pth по формуле:
        Pth = B * c * k / (2 * L * F0_mean * xi)
    и сравнивает с адаптивным порогом, откалиброванным по обучающей выборке.

    Параметры модели:
        tissue_damping      (B)  — демпфирование тканей голосовых складок
        muscle_tension           — мышечное напряжение (масштабирует xi)
        vocal_fold_length   (L)  — длина голосовых складок
        subglottal_pressure      — подскладочное давление (используется как c-proxy)
    """

    def __init__(self, physics_params):
        self.B   = physics_params.get('tissue_damping', 0.1)
        self.T   = physics_params.get('muscle_tension', 0.8)
        self.L   = physics_params.get('vocal_fold_length', 1.2)
        self.P   = physics_params.get('subglottal_pressure', 0.5)
        self.threshold_ = 0.5   # будет откалиброван на train

    def _compute_scores(self, X):
        """
        X: (n_samples, time_steps, n_features)
        Признак 0 — F0, признак 1 — jitter, признак 2 — shimmer, признак 3 — HNR.
        Возвращает score: чем выше — тем выше вероятность усталости.
        """
        # Средняя F0 по временному ряду (признак 0); защита от нуля
        f0_mean = np.mean(X[:, :, 0], axis=1) + 1e-6

        # Нормированное пороговое давление (чем выше — тем больше нагрузка)
        xi   = self.T + 0.1          # коэффициент преобразования, зависит от напряжения
        c    = self.P + 0.3          # proxy скорости волны
        k    = 0.5                   # коэффициент формы (фиксирован)
        pth  = (self.B * c * k) / (2.0 * self.L * f0_mean * xi)

        # Дополнительные акустические признаки усталости
        jitter_mean  = np.mean(X[:, :, 1], axis=1)
        shimmer_mean = np.mean(X[:, :, 2], axis=1)
        hnr_trend    = np.mean(np.diff(X[:, :, 3], axis=1), axis=1)  # падение HNR = усталость

        # Нормируем и складываем
        def norm01(v):
            vmin, vmax = v.min(), v.max()
            return (v - vmin) / (vmax - vmin + 1e-9)

        score = (
            0.35 * norm01(pth) +
            0.30 * norm01(jitter_mean) +
            0.25 * norm01(shimmer_mean) +
            0.10 * norm01(-hnr_trend)     # отрицательный тренд HNR → усталость
        )
        return score

    def fit(self, X, y):
        """Калибровка порога по обучающей выборке (максимизация F1)."""
        scores = self._compute_scores(X)
        best_thr, best_f1 = 0.5, 0.0
        for thr in np.linspace(0.1, 0.9, 81):
            pred = (scores >= thr).astype(int)
            tp = np.sum((pred == 1) & (y == 1))
            fp = np.sum((pred == 1) & (y == 0))
            fn = np.sum((pred == 0) & (y == 1))
            p  = tp / (tp + fp + 1e-9)
            r  = tp / (tp + fn + 1e-9)
            f1 = 2 * p * r / (p + r + 1e-9)
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
        self.threshold_ = best_thr
        return self

    def predict_proba(self, X):
        return self._compute_scores(X)

    def predict(self, X):
        return (self._compute_scores(X) >= self.threshold_).astype(int)


def focal_loss(gamma=2., alpha=0.25):
    """
    Focal Loss для работы с дисбалансом классов
    """

    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)

        # Calculate focal loss
        pt_1 = tf.where(K.equal(y_true, 1), y_pred, K.ones_like(y_pred))
        pt_0 = tf.where(K.equal(y_true, 0), y_pred, K.zeros_like(y_pred))

        loss = -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
        return loss

    return focal_loss_fixed


class BaseLSTM:
    """Базовая LSTM с улучшенной архитектурой"""

    def __init__(self, input_shape, lstm_units=64, class_weights=None):
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.class_weights = class_weights
        self.model = self._build_model()
        self.history = None

    def _build_model(self):
        inp = Input(shape=self.input_shape)
        x = Bidirectional(LSTM(self.lstm_units, return_sequences=True,
                               kernel_regularizer=l2(0.001),
                               dropout=0.3,
                               recurrent_dropout=0.2))(inp)
        x = BatchNormalization()(x)
        x = Bidirectional(LSTM(self.lstm_units // 2, return_sequences=True,
                               kernel_regularizer=l2(0.001),
                               dropout=0.3))(x)
        x = BatchNormalization()(x)
        x = Bidirectional(LSTM(self.lstm_units // 4, return_sequences=False,
                               kernel_regularizer=l2(0.001),
                               dropout=0.3))(x)
        x = BatchNormalization()(x)
        x = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = Dropout(0.4)(x)
        x = Dense(16, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = Dropout(0.3)(x)
        out = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=inp, outputs=out)

        model.compile(optimizer=Adam(learning_rate=0.0005),
                      loss='binary_crossentropy',
                      metrics=['accuracy',
                               keras.metrics.Precision(name='precision'),
                               keras.metrics.Recall(name='recall'),
                               keras.metrics.AUC(name='auc')])
        return model

    def train(self, X, y, X_val=None, y_val=None, epochs=150, batch_size=16, verbose=1):
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_recall' if X_val is not None else 'loss',
                mode='max',
                patience=25,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=8, min_lr=0.00001
            )
        ]

        validation_data = (X_val, y_val) if X_val is not None else None

        self.history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            validation_split=0.15 if validation_data is None else 0,
            class_weight=self.class_weights,
            callbacks=callbacks,
            verbose=verbose
        ).history
        return self.history

    def predict(self, X):
        return self.model.predict(X, verbose=0).flatten()


# physics_model_fix.py
# Добавьте этот код в models.py для PhysicsInformedNN

class PhysicsInformedNN:
    """
    Гибридная Physics-Informed Neural Network.

    Гибридность реализована через ДВА входных потока:
      1. Акустический поток — LSTM обрабатывает временной ряд (50 x 20).
      2. Физиологический поток — для каждого образца вычисляются
         индивидуальные физические признаки из его временного ряда
         на основе модели Titze (2006):
           - pth_norm    : нормированное пороговое давление фонации
           - jitter_mean : средняя частотная нестабильность
           - shimmer_mean: средняя амплитудная нестабильность
           - hnr_slope   : тренд отношения сигнал/шум (индикатор усталости)
           - f0_slope    : тренд основной частоты
           - f0_std      : вариабельность основной частоты

    Оба вектора конкатенируются перед классификационными слоями.
    Физический поток играет роль регуляризатора: он ограничивает
    пространство поиска физиологически правдоподобными состояниями.
    """

    # Индексы признаков в исходном векторе
    IDX_F0      = 0
    IDX_JITTER  = 1
    IDX_SHIMMER = 2
    IDX_HNR     = 3
    N_PHYSICS   = 6   # размер физического вектора

    def __init__(self, input_shape, physics_params, class_weights=None):
        self.input_shape   = input_shape
        self.physics_params = physics_params          # базовые параметры модели Titze
        self.B = physics_params.get('tissue_damping',      0.1)
        self.T = physics_params.get('muscle_tension',      0.8)
        self.L = physics_params.get('vocal_fold_length',   1.2)
        self.P = physics_params.get('subglottal_pressure', 0.5)
        self.class_weights = class_weights
        self.model         = self._build_model()
        self.history       = None
        self.phys_scaler   = StandardScaler()

    def _extract_physics_features(self, X):
        """
        Вычисляет индивидуальный физический вектор для каждого образца.
        X: (n_samples, time_steps, n_features)
        Возвращает: (n_samples, N_PHYSICS)
        """
        n = X.shape[0]
        feats = np.zeros((n, self.N_PHYSICS), dtype=np.float32)

        xi   = self.T + 0.1
        c    = self.P + 0.3
        k    = 0.5
        t    = np.arange(X.shape[1], dtype=np.float32)

        for i in range(n):
            f0      = X[i, :, self.IDX_F0]
            jitter  = X[i, :, self.IDX_JITTER]
            shimmer = X[i, :, self.IDX_SHIMMER]
            hnr     = X[i, :, self.IDX_HNR]

            f0_mean = np.mean(f0) + 1e-6

            # Признак 1: нормированное пороговое давление Titze
            pth = (self.B * c * k) / (2.0 * self.L * f0_mean * xi)

            # Признак 2-3: средние нестабильности
            jitter_mean  = np.mean(jitter)
            shimmer_mean = np.mean(shimmer)

            # Признак 4: тренд HNR (отрицательный => усталость)
            hnr_slope = np.polyfit(t, hnr, 1)[0]

            # Признак 5-6: тренд и вариабельность F0
            f0_slope = np.polyfit(t, f0, 1)[0]
            f0_std   = np.std(f0)

            feats[i] = [pth, jitter_mean, shimmer_mean, hnr_slope, f0_slope, f0_std]

        return feats

    def _build_model(self):
        acoustic_input = Input(shape=self.input_shape, name='acoustic_input')
        x = LSTM(128, return_sequences=True, dropout=0.2)(acoustic_input)
        x = BatchNormalization()(x)
        x = LSTM(64,  return_sequences=True, dropout=0.2)(x)
        x = BatchNormalization()(x)
        x = LSTM(32,  return_sequences=False, dropout=0.2)(x)
        x = BatchNormalization()(x)

        physics_input = Input(shape=(self.N_PHYSICS,), name='physics_input')
        p = Dense(16, activation='relu')(physics_input)
        p = Dropout(0.2)(p)

        combined = Concatenate()([x, p])
        z = Dense(32, activation='relu')(combined)
        z = Dropout(0.3)(z)
        z = Dense(16, activation='relu')(z)
        output = Dense(1, activation='sigmoid')(z)

        model = Model(inputs=[acoustic_input, physics_input], outputs=output)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy',
                     keras.metrics.Precision(name='precision'),
                     keras.metrics.Recall(name='recall'),
                     keras.metrics.AUC(name='auc')]
        )
        return model

    def train(self, X, y, X_val=None, y_val=None, epochs=150, batch_size=32, verbose=1):
        physics_features = self._extract_physics_features(X)
        physics_features = self.phys_scaler.fit_transform(physics_features)

        validation_data = None
        if X_val is not None:
            physics_val = self._extract_physics_features(X_val)
            physics_val = self.phys_scaler.transform(physics_val)
            validation_data = ([X_val, physics_val], y_val)

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=20, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5, patience=5
            )
        ]

        self.history = self.model.fit(
            [X, physics_features], y,
            epochs=epochs, batch_size=batch_size,
            validation_data=validation_data,
            validation_split=0.15 if validation_data is None else 0,
            class_weight=self.class_weights,
            callbacks=callbacks, verbose=verbose
        ).history
        return self.history

    def predict(self, X):
        physics_features = self._extract_physics_features(X)
        physics_features = self.phys_scaler.transform(physics_features)
        return self.model.predict([X, physics_features], verbose=0).flatten()

class InterpretableModel:
    """Улучшенная интерпретируемая модель"""

    def __init__(self, model_type='random_forest', class_weight='balanced'):
        self.model_type = model_type
        self.class_weight = class_weight
        self.scaler = StandardScaler()

        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_split=6,
                min_samples_leaf=3,
                class_weight={0: 1, 1: 3},  # Усиливаем вес класса усталости
                random_state=42,
                n_jobs=-1
            )
        self.feature_importance = None

    def _extract_features(self, X_timeseries):
        """Расширенное извлечение признаков"""
        n_samples = X_timeseries.shape[0]
        n_features_orig = X_timeseries.shape[2]

        features_list = []
        for i in range(n_samples):
            sample = X_timeseries[i]
            stats = []

            for j in range(n_features_orig):
                feature_series = sample[:, j]

                # Основные статистики
                stats.extend([
                    np.mean(feature_series),
                    np.std(feature_series),
                    np.percentile(feature_series, 10),
                    np.percentile(feature_series, 25),
                    np.percentile(feature_series, 50),
                    np.percentile(feature_series, 75),
                    np.percentile(feature_series, 90),
                    np.max(feature_series) - np.min(feature_series),
                ])

                # Динамические характеристики
                diff_series = np.diff(feature_series)
                stats.extend([
                    np.mean(np.abs(diff_series)),
                    np.std(diff_series),
                    np.max(np.abs(diff_series)),
                    np.sum(diff_series > 0) / len(diff_series),  # доля положительных изменений
                    np.sum(diff_series < 0) / len(diff_series),  # доля отрицательных изменений
                ])

                # Тренд (коэффициент линейной регрессии)
                if len(np.unique(feature_series)) > 1:
                    time = np.arange(len(feature_series))
                    slope = np.polyfit(time, feature_series, 1)[0]
                    stats.append(slope)
                else:
                    stats.append(0)

            features_list.append(stats)

        return np.array(features_list)

    def train(self, X_timeseries, y):
        X_features = self._extract_features(X_timeseries)
        X_scaled = self.scaler.fit_transform(X_features)
        self.model.fit(X_scaled, y)

        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
        return self.model

    def predict(self, X_timeseries):
        X_features = self._extract_features(X_timeseries)
        X_scaled = self.scaler.transform(X_features)
        return self.model.predict(X_scaled)

    def predict_proba(self, X_timeseries):
        X_features = self._extract_features(X_timeseries)
        X_scaled = self.scaler.transform(X_features)
        return self.model.predict_proba(X_scaled)[:, 1]