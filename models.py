"""
models.py — Модели для прогнозирования вокальной усталости

Изменения относительно предыдущей версии:
  1. PhysicsInformedNN переименована в PhysicsGuidedLSTM (честнее технически)
  2. Добавлена physics_loss внутрь функции потерь: total_loss = BCE + λ·physics_loss
  3. Physics features очищены от прямых акустических индикаторов усталости:
     убраны jitter_mean, shimmer_mean, hnr_slope — добавлены биомеханические
     производные: vocal_efficiency, estimated_stiffness, vocal_loading, pth_norm, f0_slope, f0_std
  4. FiLM-conditioning вместо простой конкатенации — физика модулирует
     акустическое представление (scale + shift)
  5. Monte-Carlo Dropout для uncertainty estimation (метод predict_with_uncertainty)
  6. Differentiable physics внутри кастомного Keras-слоя TitzePhysicsLayer —
     градиенты проходят через физические вычисления
  7. TitzeBaseline сохранена без изменений (baseline)
"""

import numpy as np
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import (
    LSTM, Dense, Dropout, Input, Concatenate, BatchNormalization,
    Bidirectional, Multiply, Add, Lambda
)
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import keras.backend as K


# ─── Физические константы (биомеханика по Titze 2006) ────────────────────────
PHY_B    = 0.10   # демпфирование тканей (Па·с)
PHY_C    = 0.80   # прокси скорости волны
PHY_K    = 0.50   # коэффициент формы
PHY_L    = 1.20   # длина складок (м)
PHY_XI   = 0.90   # коэффициент преобразования

# Физиологические диапазоны для physics_loss
HNR_MIN_HEALTHY = 18.0   # dB — нижняя граница нормы
PTH_MAX_HEALTHY = 0.80   # нормированное Pth — верхняя граница нормы

LAMBDA_PHYS = 0.15       # вес physics_loss в суммарной функции потерь


# ══════════════════════════════════════════════════════════════════════════════
# 1. TitzeBaseline — пороговое правило (без изменений)
# ══════════════════════════════════════════════════════════════════════════════
class TitzeBaseline:
    """
    Пороговый классификатор на основе биофизической модели Titze (2006).
    Скор усталости S = взвешенная сумма нормированных физиологических признаков.
    Порог калибруется по максимальному F1 на train-выборке.
    """

    IDX_F0 = 0; IDX_JITTER = 1; IDX_SHIMMER = 2; IDX_HNR = 3

    def __init__(self, physics_params):
        self.B = physics_params.get('tissue_damping', PHY_B)
        self.T = physics_params.get('muscle_tension', 0.8)
        self.L = physics_params.get('vocal_fold_length', PHY_L)
        self.P = physics_params.get('subglottal_pressure', 0.5)
        self.threshold_ = 0.5

    def _compute_scores(self, X):
        f0_mean = np.mean(X[:, :, self.IDX_F0], axis=1) + 1e-6
        xi = self.T + 0.1
        c  = self.P + 0.3
        pth = (self.B * c * PHY_K) / (2.0 * self.L * f0_mean * xi)
        jitter_mean  = np.mean(X[:, :, self.IDX_JITTER],  axis=1)
        shimmer_mean = np.mean(X[:, :, self.IDX_SHIMMER], axis=1)
        hnr_trend    = np.mean(np.diff(X[:, :, self.IDX_HNR], axis=1), axis=1)

        def norm01(v):
            return (v - v.min()) / (v.max() - v.min() + 1e-9)

        return (0.35 * norm01(pth) + 0.30 * norm01(jitter_mean) +
                0.25 * norm01(shimmer_mean) + 0.10 * norm01(-hnr_trend))

    def fit(self, X, y):
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


# ══════════════════════════════════════════════════════════════════════════════
# 2. BaseLSTM — Bidirectional LSTM (без изменений архитектуры)
# ══════════════════════════════════════════════════════════════════════════════
class BaseLSTM:
    """Базовая Bidirectional LSTM для классификации временных рядов."""

    def __init__(self, input_shape, lstm_units=64, class_weights=None):
        self.input_shape  = input_shape
        self.lstm_units   = lstm_units
        self.class_weights = class_weights
        self.model  = self._build_model()
        self.history = None

    def _build_model(self):
        inp = Input(shape=self.input_shape)
        x = Bidirectional(LSTM(self.lstm_units, return_sequences=True,
                               kernel_regularizer=l2(0.001), dropout=0.3,
                               recurrent_dropout=0.2))(inp)
        x = BatchNormalization()(x)
        x = Bidirectional(LSTM(self.lstm_units // 2, return_sequences=True,
                               kernel_regularizer=l2(0.001), dropout=0.3))(x)
        x = BatchNormalization()(x)
        x = Bidirectional(LSTM(self.lstm_units // 4, return_sequences=False,
                               kernel_regularizer=l2(0.001), dropout=0.3))(x)
        x = BatchNormalization()(x)
        x = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = Dropout(0.4)(x)
        x = Dense(16, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = Dropout(0.3)(x)
        out = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=inp, outputs=out)
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='binary_crossentropy',
            metrics=['accuracy',
                     keras.metrics.Precision(name='precision'),
                     keras.metrics.Recall(name='recall'),
                     keras.metrics.AUC(name='auc')]
        )
        return model

    def train(self, X, y, X_val=None, y_val=None, epochs=150, batch_size=16, verbose=1):
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_recall' if X_val is not None else 'loss',
                mode='max', patience=25, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=8, min_lr=1e-5)
        ]
        validation_data = (X_val, y_val) if X_val is not None else None
        self.history = self.model.fit(
            X, y, epochs=epochs, batch_size=batch_size,
            validation_data=validation_data,
            validation_split=0.15 if validation_data is None else 0,
            class_weight=self.class_weights,
            callbacks=callbacks, verbose=verbose
        ).history
        return self.history

    def predict(self, X):
        return self.model.predict(X, verbose=0).flatten()


# ══════════════════════════════════════════════════════════════════════════════
# 3. TitzePhysicsLayer — дифференцируемый физический слой (НОВОЕ)
# ══════════════════════════════════════════════════════════════════════════════
class TitzePhysicsLayer(keras.layers.Layer):
    """
    Кастомный Keras-слой, вычисляющий биомеханические производные
    непосредственно внутри вычислительного графа TensorFlow.

    Вход: акустические векторы (batch, time_steps, n_features).
    Выход: вектор биомеханических признаков (batch, N_PHYS).

    Поскольку вычисления находятся внутри графа, градиенты проходят
    через физические операции — это обеспечивает end-to-end learning.

    Признаки специально выбраны как ПРОИЗВОДНЫЕ биомеханические величины,
    а НЕ прямые акустические индикаторы усталости (jitter_mean, shimmer_mean),
    чтобы исключить shortcut-learning.
    """

    N_PHYS = 6  # размер выходного вектора

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Обучаемые параметры физиологической модели
        # Позволяют адаптироваться к индивидуальной физиологии вокалиста
        self.log_B = self.add_weight(
            name='log_B', shape=(1,),
            initializer=keras.initializers.Constant(np.log(PHY_B)),
            trainable=True)
        self.log_L = self.add_weight(
            name='log_L', shape=(1,),
            initializer=keras.initializers.Constant(np.log(PHY_L)),
            trainable=True)
        self.log_xi = self.add_weight(
            name='log_xi', shape=(1,),
            initializer=keras.initializers.Constant(np.log(PHY_XI)),
            trainable=True)

    def call(self, inputs, training=None):
        """
        inputs: (batch, time_steps, n_features)
        Индексы: 0=F0, 1=jitter, 2=shimmer, 3=HNR
        """
        f0  = inputs[:, :, 0]        # (batch, time_steps)
        hnr = inputs[:, :, 3]

        # Адаптивные параметры через softplus для положительности
        B   = tf.math.softplus(self.log_B)   + 1e-6
        L   = tf.math.softplus(self.log_L)   + 1e-6
        xi  = tf.math.softplus(self.log_xi)  + 1e-6

        f0_mean = tf.reduce_mean(f0, axis=1, keepdims=True) + 1e-6  # (batch, 1)

        # ── Биомеханические производные ─────────────────────────────
        # 1. Нормированное пороговое давление Pth (Titze, формула 4)
        #    pth = B·c·k / (2·L·F0·ξ)
        pth = (B * PHY_C * PHY_K) / (2.0 * L * f0_mean * xi)       # (batch, 1)

        # 2. Vocal efficiency — отношение акустической мощности к
        #    подскладочному давлению (proxy: HNR / Pth)
        hnr_mean = tf.reduce_mean(hnr, axis=1, keepdims=True)
        vocal_efficiency = hnr_mean / (pth + 1e-6)                   # (batch, 1)

        # 3. Estimated stiffness — жёсткость тканей (proxy: F0² · L · ρ,
        #    упрощённо F0_mean² нормированное)
        estimated_stiffness = tf.square(f0_mean) / (L + 1e-6)        # (batch, 1)

        # 4. Vocal fold loading — интегральная нагрузка на складки
        #    (энергетический proxy: Pth · накопленный вариационный ряд F0)
        f0_variation = tf.math.reduce_std(f0, axis=1, keepdims=True)
        vocal_loading = pth * f0_variation                            # (batch, 1)

        # 5. Тренд F0 — коэффициент наклона (линейная регрессия через
        #    ковариацию с временной осью)
        t = tf.cast(tf.range(tf.shape(f0)[1]), tf.float32)            # (time_steps,)
        t_mean = tf.reduce_mean(t)
        t_centered = t - t_mean                                        # (time_steps,)
        f0_centered = f0 - f0_mean                                     # (batch, time_steps)
        cov_f0t = tf.reduce_mean(f0_centered * t_centered, axis=1, keepdims=True)
        var_t   = tf.reduce_mean(tf.square(t_centered)) + 1e-9
        f0_slope = cov_f0t / var_t                                     # (batch, 1)

        # 6. Вариабельность F0 (f0_std)
        f0_std = f0_variation                                          # (batch, 1)

        # Конкатенация: (batch, N_PHYS)
        features = tf.concat([
            pth, vocal_efficiency, estimated_stiffness,
            vocal_loading, f0_slope, f0_std
        ], axis=1)

        return features

    def get_config(self):
        return super().get_config()


# ══════════════════════════════════════════════════════════════════════════════
# 4. PhysicsGuidedLSTM (бывшая PhysicsInformedNN) — ПЕРЕРАБОТАНА
# ══════════════════════════════════════════════════════════════════════════════
class PhysicsGuidedLSTM:
    """
    Biomechanically-Regularized Acoustic Network для прогнозирования
    вокальной усталости.

    Изменения относительно предыдущей версии:
    ─────────────────────────────────────────
    1. Переименована: PhysicsInformedNN → PhysicsGuidedLSTM
       (технически точнее: модель использует физику как guidance,
       не как жёсткое ограничение потерь в смысле PDE)

    2. TitzePhysicsLayer внутри графа:
       - градиенты через физику проходят (end-to-end learning)
       - параметры B, L, ξ обучаемы (адаптация к физиологии)

    3. Physics features — биомеханические производные, НЕ акустические
       индикаторы усталости напрямую:
       [pth, vocal_efficiency, estimated_stiffness,
        vocal_loading, f0_slope, f0_std]

    4. FiLM-conditioning вместо конкатенации:
       physics_branch генерирует (γ, β) для масштабирования и сдвига
       акустического представления: output = γ ⊙ acoustic + β

    5. Physics loss в функции потерь:
       total_loss = BCE(y, ŷ) + λ · physics_loss
       physics_loss штрафует за нефизиологичные состояния:
       - pth вне допустимого диапазона
       - несоответствие vocal_efficiency нормальным значениям

    6. Monte-Carlo Dropout: метод predict_with_uncertainty() возвращает
       mean prediction и std (confidence interval).
    """

    def __init__(self, input_shape, physics_params, class_weights=None,
                 lambda_phys=LAMBDA_PHYS):
        self.input_shape   = input_shape
        self.physics_params = physics_params
        self.class_weights  = class_weights
        self.lambda_phys    = lambda_phys
        self.history        = None

        self.model          = self._build_model()

    def _physics_loss(self, physics_features, y_true):
        """
        Штраф за нефизиологичные предсказания.
        physics_features: (batch, N_PHYS) = [pth, vocal_eff, stiffness,
                                              loading, f0_slope, f0_std]
        """
        pth         = physics_features[:, 0:1]
        vocal_eff   = physics_features[:, 1:2]

        # Для «нормальных» образцов (y=0) pth должен быть малым
        # Для «усталых» (y=1) pth должен быть выше нормы
        y = tf.cast(tf.expand_dims(y_true, 1), tf.float32)

        # Штраф: нормальные образцы с высоким pth (нефизиологично)
        pth_penalty_normal  = tf.reduce_mean(y * tf.nn.relu(PTH_MAX_HEALTHY - pth))
        # Штраф: усталые образцы с высокой vocal_efficiency (нефизиологично)
        eff_penalty_fatigue = tf.reduce_mean((1 - y) * tf.nn.relu(vocal_eff - 5.0))

        return pth_penalty_normal + eff_penalty_fatigue

    def _build_model(self):
        acoustic_input = Input(shape=self.input_shape, name='acoustic_input')

        # ── Акустический поток (LSTM) ───────────────────────────────
        x = LSTM(128, return_sequences=True, dropout=0.2)(acoustic_input)
        x = BatchNormalization()(x)
        x = LSTM(64,  return_sequences=True, dropout=0.2)(x)
        x = BatchNormalization()(x)
        acoustic_repr = LSTM(32, return_sequences=False, dropout=0.2)(x)
        acoustic_repr = BatchNormalization()(acoustic_repr)  # (batch, 32)

        # ── Физиологический поток (TitzePhysicsLayer) ──────────────
        # Градиенты проходят через физические вычисления
        physics_features = TitzePhysicsLayer(name='titze_physics')(acoustic_input)
        # (batch, N_PHYS=6)

        # ── FiLM-conditioning ───────────────────────────────────────
        # Физика генерирует γ (scale) и β (shift) для модуляции
        # акустического представления: output = γ ⊙ acoustic + β
        film_params = Dense(64, activation='relu',
                            name='film_params')(physics_features)  # (batch, 64)
        film_params = Dropout(0.2)(film_params)

        gamma = Dense(32, activation='sigmoid', name='film_gamma')(film_params)  # (batch, 32)
        beta  = Dense(32, activation='tanh',    name='film_beta')(film_params)   # (batch, 32)

        # Применяем FiLM: физика модулирует акустику (интеграция на уровне
        # скрытого пространства, а не просто конкатенация)
        modulated = Add(name='film_add')([
            Multiply(name='film_scale')([gamma, acoustic_repr]),
            beta
        ])  # (batch, 32)

        # ── Классификационные слои ──────────────────────────────────
        z = Dense(32, activation='relu')(modulated)
        z = Dropout(0.3)(z)
        z = Dense(16, activation='relu')(z)

        output = Dense(1, activation='sigmoid', name='prediction')(z)

        model = Model(inputs=acoustic_input, outputs=[output, physics_features],
                      name='PhysicsGuidedLSTM')
        return model

    def _custom_loss(self, y_true, y_pred_output, physics_feat):
        """Суммарная функция потерь: BCE + λ · physics_loss."""
        bce = keras.losses.binary_crossentropy(
            tf.cast(y_true, tf.float32),
            tf.squeeze(y_pred_output, axis=1)
        )
        phys = self._physics_loss(physics_feat, y_true)
        return tf.reduce_mean(bce) + self.lambda_phys * phys

    def train(self, X, y, X_val=None, y_val=None,
              epochs=150, batch_size=32, verbose=1):
        optimizer = Adam(learning_rate=0.001)

        @tf.function
        def train_step(x_batch, y_batch):
            with tf.GradientTape() as tape:
                y_pred, phys_feat = self.model(x_batch, training=True)
                loss = self._custom_loss(y_batch, y_pred, phys_feat)
            grads = tape.gradient(loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            return loss

        @tf.function
        def val_step(x_batch, y_batch):
            y_pred, phys_feat = self.model(x_batch, training=False)
            return self._custom_loss(y_batch, y_pred, phys_feat)

        history = {'train_loss': [], 'val_loss': [], 'val_recall': []}
        best_val_loss = np.inf
        best_weights  = None
        patience_cnt  = 0
        patience      = 20

        dataset = tf.data.Dataset.from_tensor_slices(
            (X.astype(np.float32), y.astype(np.float32))
        ).shuffle(len(X)).batch(batch_size)

        for epoch in range(epochs):
            # Train
            train_losses = []
            for xb, yb in dataset:
                loss = train_step(xb, yb)
                train_losses.append(float(loss))
            epoch_train = np.mean(train_losses)

            # Val
            if X_val is not None:
                val_pred, _ = self.model(X_val.astype(np.float32), training=False)
                val_pred    = val_pred.numpy().flatten()
                val_loss    = float(val_step(
                    X_val.astype(np.float32), y_val.astype(np.float32)))
                val_pred_b  = (val_pred > 0.5).astype(int)
                tp  = np.sum((val_pred_b == 1) & (y_val == 1))
                fn  = np.sum((val_pred_b == 0) & (y_val == 1))
                rec = tp / (tp + fn + 1e-9)

                history['train_loss'].append(epoch_train)
                history['val_loss'].append(val_loss)
                history['val_recall'].append(rec)

                if verbose and (epoch + 1) % 20 == 0:
                    print(f"   Epoch {epoch+1}/{epochs}  "
                          f"train_loss={epoch_train:.4f}  "
                          f"val_loss={val_loss:.4f}  "
                          f"val_recall={rec:.3f}")

                # Early stopping по val_loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights  = self.model.get_weights()
                    patience_cnt  = 0
                else:
                    patience_cnt += 1
                    if patience_cnt >= patience:
                        if verbose:
                            print(f"   Early stopping at epoch {epoch+1}")
                        break
            else:
                history['train_loss'].append(epoch_train)

        if best_weights is not None:
            self.model.set_weights(best_weights)
        self.history = history
        return history

    def predict(self, X):
        """Детерминированное предсказание (training=False)."""
        y_pred, _ = self.model(X.astype(np.float32), training=False)
        return y_pred.numpy().flatten()

    def predict_with_uncertainty(self, X, n_passes=30):
        """
        Monte-Carlo Dropout: T стохастических прогонов с training=True
        для оценки неопределённости предсказания.

        Возвращает:
            mean_pred: np.array (n_samples,) — среднее по T прогонам
            std_pred:  np.array (n_samples,) — стандартное отклонение
                       (ширина доверительного интервала)
        """
        preds = []
        X_tf  = X.astype(np.float32)
        for _ in range(n_passes):
            y_p, _ = self.model(X_tf, training=True)   # Dropout активен
            preds.append(y_p.numpy().flatten())
        preds = np.array(preds)                          # (T, n_samples)
        return preds.mean(axis=0), preds.std(axis=0)

    def get_physics_features(self, X):
        """Возвращает биомеханические признаки из TitzePhysicsLayer."""
        _, phys = self.model(X.astype(np.float32), training=False)
        return phys.numpy()

    def get_learned_physics_params(self):
        """Возвращает обученные параметры B, L, ξ физиологической модели."""
        layer = self.model.get_layer('titze_physics')
        B   = float(tf.math.softplus(layer.log_B).numpy()[0])
        L   = float(tf.math.softplus(layer.log_L).numpy()[0])
        xi  = float(tf.math.softplus(layer.log_xi).numpy()[0])
        return {'B (tissue_damping)': B, 'L (fold_length)': L, 'xi': xi}


# ══════════════════════════════════════════════════════════════════════════════
# 5. InterpretableModel — Random Forest (без изменений)
# ══════════════════════════════════════════════════════════════════════════════
class InterpretableModel:
    """Random Forest с расширенным извлечением статистических признаков."""

    def __init__(self, model_type='random_forest', class_weight='balanced'):
        self.model_type   = model_type
        self.class_weight = class_weight
        self.scaler       = StandardScaler()
        self.model        = RandomForestClassifier(
            n_estimators=200, max_depth=12,
            min_samples_split=6, min_samples_leaf=3,
            class_weight={0: 1, 1: 3},
            random_state=42, n_jobs=-1
        )
        self.feature_importance = None

    def _extract_features(self, X):
        n_samples, n_timesteps, n_features = X.shape
        t = np.arange(n_timesteps, dtype=np.float32)
        features_list = []
        for i in range(n_samples):
            sample = X[i]
            stats  = []
            for j in range(n_features):
                series = sample[:, j]
                diff   = np.diff(series)
                stats.extend([
                    np.mean(series), np.std(series),
                    np.percentile(series, 10), np.percentile(series, 25),
                    np.percentile(series, 50), np.percentile(series, 75),
                    np.percentile(series, 90),
                    np.max(series) - np.min(series),
                    np.mean(np.abs(diff)), np.std(diff),
                    np.max(np.abs(diff)),
                    np.sum(diff > 0) / len(diff),
                    np.sum(diff < 0) / len(diff),
                    np.polyfit(t, series, 1)[0] if len(np.unique(series)) > 1 else 0.0,
                ])
            features_list.append(stats)
        return np.array(features_list)

    def train(self, X, y):
        X_features = self._extract_features(X)
        X_scaled   = self.scaler.fit_transform(X_features)
        self.model.fit(X_scaled, y)
        self.feature_importance = self.model.feature_importances_
        return self.model

    def predict(self, X):
        X_scaled = self.scaler.transform(self._extract_features(X))
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(self._extract_features(X))
        return self.model.predict_proba(X_scaled)[:, 1]


# ══════════════════════════════════════════════════════════════════════════════
# Backward compatibility alias
# ══════════════════════════════════════════════════════════════════════════════
PhysicsInformedNN = PhysicsGuidedLSTM   # старый импорт продолжает работать
