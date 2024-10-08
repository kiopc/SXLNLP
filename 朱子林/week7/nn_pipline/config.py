# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "../data/train_tag_news.json",
    "valid_data_path": "../data/valid_tag_news.json",
    "vocab_path":"chars.txt",
    "model_type":"bert",
    "max_length": 463,
    "hidden_size": 128,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 5,
    "batch_size": 10,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"D:\Personal\BaDou\八斗精品班\第六周 预训练模型\bert-base-chinese",
    "seed": 996
}

