# -*- coding: utf-8 -*-

from models.dataReader import *
from models.train import maml_train_on_batch, maml_eval
from models.net import MAML_model
import models.config as cfg

import tensorflow as tf
import os
from tqdm import tqdm
import PIL

datasets = ["bolt", "cable", "canister", "knobs", "all"]


def demo(img_dir, labels, outputPath):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    maml_model = MAML_model(num_classes=cfg.n_way)
    maml_model.load_weights(cfg.save_path)
    # 直接进行前向传播，不然权重就是空的（前向传播不会改变权值），如果是用keras的Sequential来建立模型就自动初始化了
    valid_list = read_dataset(img_dir, 2)
    valid_dataset = task_split(valid_list, q_query=cfg.q_query, n_way=cfg.n_way, k_shot=cfg.k_shot)

    valid_iter = DataIter(valid_dataset)
    valid_step = len(valid_dataset) // cfg.eval_batch_size

    val_acc = []
    val_loss = []

    # valid
    # process_bar = tqdm(range(valid_step), ncols=100, desc="model demo", unit="step")
    for _ in range(valid_step):
        batch_task = get_meta_batch(valid_iter, cfg.eval_batch_size)
        loss, acc = maml_train_on_batch(maml_model,
                                        batch_task,
                                        n_way=cfg.n_way,
                                        k_shot=cfg.k_shot,
                                        q_query=cfg.q_query,
                                        lr_inner=cfg.inner_lr,
                                        lr_outer=cfg.outer_lr,
                                        inner_train_step=3,
                                        meta_update=False)
        val_loss.append(loss)
        val_acc.append(acc)

        print("test_loss:{:.4f} test_accuracy:{:.4f}".format(np.mean(val_loss), np.mean(val_acc)))
    maml_model.save(outputPath)
    return np.mean(val_acc)
    # maml_model.save_weights(outputPath)
        # process_bar.set_postfix({'val_loss': '{:.5f}'.format(loss), 'val_acc': '{:.5f}'.format(acc)})

        # 输出平均后的验证结果
        # print("\rvalidation_loss:{:.4f} validation_accuracy:{:.4f}\n".format(np.mean(val_loss), np.mean(val_acc)))
