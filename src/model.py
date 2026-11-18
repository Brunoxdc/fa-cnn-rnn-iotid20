"""Arquitectura híbrida CNN-RNN (dos ramas en paralelo, concatenadas)."""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models

from . import config as C


def build_cnn_rnn(n_features: int, n_classes: int = 2) -> tf.keras.Model:
    inp = layers.Input(shape=(n_features, 1), name="flujo")

    # Rama CNN: patrones locales entre columnas vecinas
    c = layers.Conv1D(C.CONV_FILTERS, C.CONV_KERNEL, activation="relu", name="conv1d")(inp)
    c = layers.MaxPooling1D(C.POOL_SIZE, name="maxpool")(c)
    c = layers.Flatten(name="flatten")(c)
    c = layers.Dropout(C.DROPOUT, name="drop_cnn")(c)

    # Rama RNN: lee las columnas como una secuencia
    rnn_cls = {"SimpleRNN": layers.SimpleRNN, "LSTM": layers.LSTM, "GRU": layers.GRU}[C.RNN_TYPE]
    r = rnn_cls(C.RNN_UNITS, activation="relu", name="rnn")(inp)
    r = layers.Dropout(C.DROPOUT, name="drop_rnn")(r)

    x = layers.Concatenate(name="concat")([c, r])
    if n_classes == 2:
        out = layers.Dense(1, activation="sigmoid", name="salida")(x)
        loss = "binary_crossentropy"
    else:
        out = layers.Dense(n_classes, activation="softmax", name="salida")(x)
        loss = "sparse_categorical_crossentropy"

    model = models.Model(inp, out, name="FA_CNN_RNN")
    model.compile(optimizer=tf.keras.optimizers.Adam(C.LEARNING_RATE),
                  loss=loss, metrics=["accuracy"])
    return model
