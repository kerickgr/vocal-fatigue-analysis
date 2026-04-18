import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate, BatchNormalization, Bidirectional, \
    Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import tensorflow.keras.backend as K


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
        model = Sequential([
            Bidirectional(LSTM(self.lstm_units, return_sequences=True,
                               kernel_regularizer=l2(0.001),
                               dropout=0.3,
                               recurrent_dropout=0.2),
                          input_shape=self.input_shape),
            BatchNormalization(),
            Bidirectional(LSTM(self.lstm_units // 2, return_sequences=True,
                               kernel_regularizer=l2(0.001),
                               dropout=0.3)),
            BatchNormalization(),
            Bidirectional(LSTM(self.lstm_units // 4, return_sequences=False,
                               kernel_regularizer=l2(0.001),
                               dropout=0.3)),
            BatchNormalization(),
            Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.4),
            Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer=Adam(learning_rate=0.0005),
                      loss='binary_crossentropy',
                      metrics=['accuracy',
                               tf.keras.metrics.Precision(name='precision'),
                               tf.keras.metrics.Recall(name='recall'),
                               tf.keras.metrics.AUC(name='auc')])
        return model

    def train(self, X, y, X_val=None, y_val=None, epochs=150, batch_size=16, verbose=1):
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_recall' if X_val is not None else 'loss',
                mode='max',
                patience=25,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
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
    """Исправленная гибридная модель"""

    def __init__(self, input_shape, physics_params, class_weights=None):
        self.input_shape = input_shape
        self.physics_params = physics_params
        self.class_weights = class_weights
        self.model = self._build_model()
        self.history = None
        self.scaler = StandardScaler()

    def _build_model(self):
        # Вход для временных рядов
        acoustic_input = Input(shape=self.input_shape, name='acoustic_input')

        # Упрощенная, но эффективная архитектура
        x = LSTM(128, return_sequences=True, dropout=0.2)(acoustic_input)
        x = BatchNormalization()(x)
        x = LSTM(64, return_sequences=True, dropout=0.2)(x)
        x = BatchNormalization()(x)
        x = LSTM(32, return_sequences=False, dropout=0.2)(x)
        x = BatchNormalization()(x)

        # Физические параметры
        physics_input = Input(shape=(len(self.physics_params),), name='physics_input')
        p = Dense(16, activation='relu')(physics_input)
        p = Dropout(0.2)(p)

        # Объединение
        combined = Concatenate()([x, p])

        # Выходные слои
        z = Dense(32, activation='relu')(combined)
        z = Dropout(0.3)(z)
        z = Dense(16, activation='relu')(z)
        output = Dense(1, activation='sigmoid')(z)

        model = Model(inputs=[acoustic_input, physics_input], outputs=output)

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy',
                     tf.keras.metrics.Precision(name='precision'),
                     tf.keras.metrics.Recall(name='recall'),
                     tf.keras.metrics.AUC(name='auc')]
        )
        return model

    def train(self, X, y, X_val=None, y_val=None, epochs=150, batch_size=32, verbose=1):
        # Нормализуем физические параметры
        physics_features = np.array([list(self.physics_params.values()) for _ in range(len(X))])
        physics_features = self.scaler.fit_transform(physics_features)

        validation_data = None
        if X_val is not None:
            physics_val = np.array([list(self.physics_params.values()) for _ in range(len(X_val))])
            physics_val = self.scaler.transform(physics_val)
            validation_data = ([X_val, physics_val], y_val)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=20,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=5
            )
        ]

        self.history = self.model.fit(
            [X, physics_features], y,
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
        physics_features = np.array([list(self.physics_params.values()) for _ in range(len(X))])
        physics_features = self.scaler.transform(physics_features)
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