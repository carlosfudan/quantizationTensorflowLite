# -*- coding: utf-8 -*-

import time
import config as cfg
import tensorflow as tf
import os
import numpy as np

class TfLiteModel:
    def __init__(self, model_content):
        self.model_content = bytes(model_content)
        self.interpreter = tf.lite.Interpreter(model_content=self.model_content)
        self.interpreter.allocate_tensors()
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        print(input_details, output_details)
        self.input_index = input_details[0]['index']
        self.output_index = output_details[0]['index']

        self.input_scale, self.input_zero_point = input_details[0]['quantization']
        self.output_scale, self.output_zero_point = output_details[0]['quantization']


    @profile
    def forward(self, data_in):
        if cfg.quantize_mode == "int8":
            test_input = np.array(data_in / self.input_scale + self.input_zero_point, dtype=np.uint8)
        else:
            test_input = data_in
        self.interpreter.set_tensor(self.input_index, test_input)
        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.output_index)[0]
        if cfg.quantize_mode == "int8":
            output_data = (np.array(output_data, dtype=np.float32) - self.output_zero_point) * self.output_scale
        return output_data




def process_img(img_path, width=cfg.width, height=cfg.height, channel=cfg.channel):
    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image,channels=channel)
    image = tf.image.resize(image, [width, height])
    image = tf.expand_dims(image, 0)
    image = tf.image.convert_image_dtype(image, tf.float32)

    return image


def inference(img_path, model_path):
    with open(model_path, 'rb') as f:
        model_content = f.read()

    model = TfLiteModel(model_content)


    dirs = os.listdir(img_path)
    imgs = []
    for file in dirs:
        path = os.path.join(img_path, file)
        image = process_img(path)
        imgs.append(image)
    t = []

    for image in imgs:
        start_time = time.time()
        result = model.forward(image)
        end_time = time.time()
        t.append((end_time-start_time)/len(imgs)*1000)
    print(t)
    # print("推理时间：",(end_time-start_time)/len(imgs)*1000)
    # print("预测结果： ", result)

    # print("预测结果： ", np.argmax(result, axis=1)[0])
tflite_path = cfg.quantize_ds_output if cfg.quantize_model else cfg.quantize_cnn_output
tflite_path = tflite_path + cfg.quantize_mode + ".tflite"
print("model:",cfg.quantize_mode)
if cfg.quantize_model:
    print("ds model")
else:
    print("cnn model")
inference("./data/", tflite_path)