# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "ner_data/schema.json",
    "train_data_path": "ner_data/train",
    "valid_data_path": "ner_data/test",
    "vocab_path":"chars.txt",
    "max_length": 100,
    "hidden_size": 256,
    "num_layers": 2,
    "epoch": 20,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 1e-4,
    "use_crf": False,
    "class_num": 9,
    "bert_path": r"E:\八斗学院AI课程\NLP\八斗精品班\第六周 预训练模型\bert-base-chinese\bert-base-chinese"
}

