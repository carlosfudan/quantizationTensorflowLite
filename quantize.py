# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import numpy as np
import config as cfg

quantize_mode = cfg.quantize_mode
output = cfg.quantize_ds_output if cfg.quantize_model else cfg.quantize_cnn_output
output = output+quantize_mode+".tflite"

# model = tf.keras.applications.MobileNetV2(input_shape=(224,224,3))

path = cfg.save_ds_path if cfg.quantize_model else cfg.save_cnn_path
model = tf.keras.models.load_model(path)
# 不指定FLAGS.quantize_mode，则为没有量化的tflite模型
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# float16

def representative_data_gen():
    for _ in range(100):
        data = np.random.rand(1,64,64,3)
        yield [data.astype(np.float32)]


if quantize_mode == 'float16':
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
    converter.target_spec.supported_types = [tf.float16]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.allow_custom_ops = True
    converter.post_training_quantize = True

    # int8
elif quantize_mode == 'int8':
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8  # or tf.uint8
    converter.inference_output_type = tf.uint8  # or tf.uint8
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.allow_custom_ops = True

    # 动态模型
elif quantize_mode == 'for_size':
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    converter.representative_dataset = representative_data_gen
    converter.allow_custom_ops = True


tflite_model = converter.convert()
open(output, 'wb').write(tflite_model)