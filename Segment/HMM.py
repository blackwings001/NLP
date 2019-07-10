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
                self.initial_dic = pickle.load(f)
                self.transform_dic = pickle.load(f)
                self.launch_dic = pickle.load(f)
        else:
            self.initial_dic = {}
            self.transform_dic = {}
            self.launch_dic = {}


    def train(self, path):
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
                label += ["B"] + ["M"] * (len(word) - 2) + ["E"]
            return label

        init_parameters()
        line_num = -1

        # 读取文本进行训练语料
        words = set()
        with open(path, encoding="utf-8") as f:
            for line in f.readlines():
                line_num += 1

                line = line.strip()
                if not line:
                    continue

                # 更新字集合
                word_list = [i for i in line if i != " "]
                words |= set(word_list)

                line_list = line.split()
                line_state = []
                for word in line_list:
                    line_state.extend(make_label(word))

                assert len(word_list) == len(line_state)

                for k, v in enumerate(line_state):
                    count_dic[v] += 1
                    if k == 0:
                        self.initial_dic[v] += 1
                    else:
                        self.transform_dic[line_state[k-1]][v] += 1
                    self.launch_dic[v][word_list[k]] = self.launch_dic[v].get(word_list[k], 0.0) + 1

        self.initial_dic = {k : v * 1.0 / line_num for k, v in self.initial_dic.items()}
        self.transform_dic = {k : {k1 : v1 / count_dic[k] for k1, v1 in v.items()} for k, v in self.transform_dic.items()}
        self.launch_dic = {k : {k1 : (v1 + 1) / count_dic[k] for k1, v1 in v.items()} for k, v in self.launch_dic.items()}  # 指数平滑

        with open(self.model_file, "wb") as f:
            pickle.dump(self.initial_dic, f)
            pickle.dump(self.transform_dic, f)
            pickle.dump(self.launch_dic, f)

        print("初始矩阵为: {}".format(self.initial_dic))
        print("转移矩阵为: {}".format(self.transform_dic))
        print("发射矩阵为: {}".format(self.launch_dic))
        print("词典大小为: {}".format(len(words)))

        return self




    def cut(self, text):
        # 使用HMM模型(几个矩阵)，对文本进行分词，分词过程中用到了viterbi算法
        if os.path.exists(self.model_file):
            self.load_or_train_model(flag="load")

        def viterbi(text, state_list, initial_dic, transform_dic, launch_dic):
            # 使用动态规划，找到概率最大的分词方法
            V = [{}] # V记录每一步，每一状态的概率值
            path = {} # path记录每一步，最终状态为BMES的以往最优路径

            # 第一个字
            for y in state_list:
                V[0][y] = initial_dic.get(y, 0.0) * launch_dic[y].get(text[0], 0.0)
                path[y] = [y]

            for i in range(1, len(text)):
                V.append({})
                tmp_path = {}   # 临时更新path
                char = text[i]

                neverSeen = char not in launch_dic['B'].keys() and \
                            char not in launch_dic['M'].keys() and \
                            char not in launch_dic['E'].keys() and \
                            char not in launch_dic['S'].keys()

                # 计算该位置为各状态的最大概率以及前一状态值
                for s in state_list:
                    launch_pro = launch_dic[s].get(char, 0.0) if not neverSeen else 1   # 四个发射矩阵都没有出现过的字单独成词
                    (prob, pre_s) = max([(V[i-1][p_s] * transform_dic[p_s].get(s, 0.0) * launch_pro, p_s) for p_s in state_list if V[i-1][p_s] > 0])
                    V[i][s] = prob
                    tmp_path[s] = path[pre_s] + [s] # 直接更新path会影响到后面的最优路径计算

                path = tmp_path

            (prob, final_state) = max([(V[len(text) - 1][s], s) for s in state_list])

            return (prob, path[final_state])

        prob, pos_list = viterbi(text, state_list=self.state_list, initial_dic=self.initial_dic, transform_dic=self.transform_dic, launch_dic=self.launch_dic)
        print("分词状态列表为: {}".format(pos_list))
        print("上面状态列表的概率为: {}".format(prob))

        begin, end = 0, 0
        for i, char in enumerate(text):
            pos = pos_list[i]
            if pos == "B":
                begin = i
            elif pos == "E":
                end = i + 1
                yield text[begin : end]
            elif pos == "S":
                end = i + 1
                yield text[i]
        if end < len(text):
            yield text[end:]



















