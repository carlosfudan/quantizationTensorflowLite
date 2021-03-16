# -*- coding: utf-8 -*-

from tensorflow.keras import layers, activations, Model, optimizers, models
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import config as cfg


def MAML_model(num_classes, width=cfg.width, height=cfg.height, channel=cfg.channel):
    cnn = models.Sequential([
        layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu",
                      input_shape=[width, height, channel]),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=2, strides=2),

        layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=2, strides=2),

        layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=2, strides=2),

        layers.Conv2D(filters=16, kernel_size=3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=2, strides=2),

        layers.Flatten(),
        layers.Dense(num_classes),
    ])
    ds_cnn = models.Sequential([
        layers.SeparableConv2D(32, kernel_size=3, padding="same", activation="relu",
                      input_shape=[width, height, channel]),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=2, strides=2),

        layers.SeparableConv2D(32, kernel_size=3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=2, strides=2),

        layers.SeparableConv2D(32, kernel_size=3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=2, strides=2),

        layers.SeparableConv2D(16, kernel_size=3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=2, strides=2),
        layers.Flatten(),
        layers.Dense(num_classes),
    ])

    model = ds if cfg.quantize_model else cnn
    return model


