"""
Author: guominglei
Date: 2019-07-02
Readme: 实现HMM算法进行中文分词
"""
import os
import pickle

class HMM:
    def __init__(self, model_file):
        # 初始化一些变量，状态转移矩阵，发射矩阵，初始概率等，HMM模型实际内容是上面的几个矩阵
        self.model_file = model_file
        self.state_list = ['B', 'M', 'E', 'S']

    def load_or_train_model(self, flag="load"):
        if flag == "load":
            with open(self.model_file, "rb") as f:
                self.transform_dic = pickle.load(f)
                self.launch_dic = pickle.load(f)
                self.initial_dic = pickle.load(f)
        else:
            self.transform_dic = {}
            self.launch_dic = {}
            self.initial_dic = {}



    def train(self):
        # 根据分词语料（每行一句话，逗号分开也是一句话，词之间使用空格进行分隔），训练HMM模型, 目的是获得状态转移矩阵，发射矩阵和初始概率矩阵
        # 训练前重置几个矩阵
        self.load_or_train_model(flag="train")

        count_dic = {}  # 统计各状态出现的次数

        def init_parameters():
            for state in self.state_list:
                self.transform_dic[state] = {s: 0.0 for s in self.state_list}
                self.launch_dic[state] = {}
                self.initial_dic[state] = 0.0
                count_dic[state] = 0.0

        def make_label(word):
            # 计算词的BMES状态
            label = []
            if len(word) == 1:
                label.append("S")
            else:
                label += ["B"] + ["M"] * (len(word) - 2) + ["S"]
            return label








        pass

    def cut(self):
        # 使用HMM模型(几个矩阵)，对文本进行分词，分词过程中用到了viterbi算法
        pass

