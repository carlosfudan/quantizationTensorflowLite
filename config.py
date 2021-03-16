# -*- coding: utf-8 -*-

batch_size = 16
eval_batch_size = 8
epochs = 1

inner_lr = 0.05
outer_lr = 1e-3

n_way = 5
k_shot = 1
q_query = 1

width = 64
height = 64
channel = 3

save_cnn_path = "./models/logs/model/cnn/maml_cnn"
save_cnn_model_path = "./models/logs/model/cnn/"

save_ds_path = "./models/logs/model/ds/maml_ds"
save_ds_model_path = "./models/logs/model/ds/"

log_dir = "./models/logs/summary/"

# 0: cnn
# 1: ds
quantize_model = 1


#float16 int8 for_size origin
quantize_mode = "for_size"
quantize_cnn_output = './outputs/qua_cnn_'
quantize_ds_output = './outputs/qua_ds_'
