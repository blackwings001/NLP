"""
Author: guominglei
Date: 2019-07-08
Readme: main程序用于测试不同的分词模型的效果
"""
import os
from HMM import HMM


if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.dirname(os.getcwd())) #  NLP文件夹
    SEGMENT_LITTLE_FILE = os.path.join(BASE_DIR, "Data", "Law", "segment", "segment_little.txt")
    SEGMENT_FILE = os.path.join(BASE_DIR, "Data", "Law", "segment", "segment.txt")
    HMM_MODEL_FILE = os.path.join(BASE_DIR, "Model", "Segment", "HMM", "model.pkl")

    TEXT = "被告人驾车撞倒了李某某，并且迅速逃离。"

    hmm = HMM(HMM_MODEL_FILE)
    seg_text = hmm.train(SEGMENT_FILE).cut(TEXT)

    print(list(seg_text))

