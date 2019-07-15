"""
Author: guominglei
Date: 2019-07-15
Readme: 本脚本保存神经网络模型
"""

from keras.models import Sequential, load_model
from keras.layers import Conv1D, MaxPooling1D, Input, concatenate
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.layers.wrappers import Bidirectional  # 构建双向循环网络
from keras.models import Model


class Network:
    def __init__(self, token_words, max_lenth, embedding_dim, output_dim):
        # 嵌入层参数
        self.token_words = token_words                # 字典维度
        self.max_lenth = max_lenth                    # padding后句子长度
        self.embedding_output = embedding_dim         # 词向量维度
        self.output_dim = output_dim                  # 最后输出层的维度

    def mlp(self):      # 多层感知机神经网络结构
        model = Sequential()    # 堆叠式模型
        # 网络的第一层为Embedding层,嵌入层将单词数字变为向量, output_dim嵌入层的输出维度，每个单词用多少维向量表示。input_length为输入句子长度，　input_dim为字典维度。
        model.add(Embedding(input_dim=self.token_words, input_length=self.max_lenth, output_dim=self.embedding_output))
        model.add(Flatten())  # 加入Flattern层，变为３２００个神经元
        model.add(Dropout(0.2))  # 加入Dropout层，防止过拟合
        model.add(Dense(units=256, activation="relu"))  # 隐藏层
        model.add(Dropout(0.5))
        model.add(Dense(units=self.output_dim, activation="softmax"))  # 输出层
        print("神经网络的结构如下：")
        print(model.summary())
        return model

    def rnn(self):
        model = Sequential()   # 堆叠式模型
        # 网络的第一层为Embedding层,嵌入层将单词数字变为向量, output_dim嵌入层的输出维度，每个单词用多少维向量表示。input_length为输入句子长度，　input_dim为字典维度。
        model.add(Embedding(input_dim=self.token_words, input_length=self.max_lenth, output_dim=self.embedding_output))
        model.add(Dropout(0.2))  # 加入Dropout层，防止过拟合
        model.add(SimpleRNN(units=16))
        model.add(Dense(units=256, activation="relu"))  # 隐藏层
        model.add(Dropout(0.5))
        model.add(Dense(units=self.output_dim, activation="softmax"))  # 输出层
        print("神经网络的结构如下：")
        print(model.summary())
        return model

    def birnn(self):
        model = Sequential()   # 堆叠式模型
        # 网络的第一层为Embedding层,嵌入层将单词数字变为向量, output_dim嵌入层的输出维度，每个单词用多少维向量表示。input_length为输入句子长度，　input_dim为字典维度。
        model.add(Embedding(input_dim=self.token_words, input_length=self.max_lenth, output_dim=self.embedding_output))
        model.add(Dropout(0.2))  # 加入Dropout层，防止过拟合
        model.add(Bidirectional(SimpleRNN(units=16, return_sequences=True), merge_mode="concat"))     # 加入Flattern层，变为３２００个神经元
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(units=256, activation="relu"))  # 隐藏层
        model.add(Dropout(0.5))
        model.add(Dense(units=self.output_dim, activation="softmax"))  # 输出层
        print("神经网络的结构如下：")
        print(model.summary())
        return model

    def lstm(self):
        model = Sequential()   # 堆叠式模型
        # 网络的第一层为Embedding层,嵌入层将单词数字变为向量, output_dim嵌入层的输出维度，每个单词用多少维向量表示。input_length为输入句子长度，　input_dim为字典维度。
        model.add(Embedding(input_dim=self.token_words, input_length=self.max_lenth, output_dim=self.embedding_output))
        model.add(Dropout(0.2))  # 加入Dropout层，防止过拟合
        model.add(LSTM(units=32))     # 加入Flattern层，变为３２００个神经元
        model.add(Dense(units=256, activation="relu"))  # 隐藏层
        model.add(Dropout(0.5))
        model.add(Dense(units=self.output_dim, activation="softmax"))  # 输出层
        print("神经网络的结构如下：")
        print(model.summary())
        return model

    def gru(self):
        model = Sequential()   # 堆叠式模型
        # 网络的第一层为Embedding层,嵌入层将单词数字变为向量, output_dim嵌入层的输出维度，每个单词用多少维向量表示。input_length为输入句子长度，　input_dim为字典维度。
        model.add(Embedding(input_dim=self.token_words, input_length=self.max_lenth, output_dim=self.embedding_output))
        model.add(Dropout(0.2))  # 加入Dropout层，防止过拟合
        model.add(GRU(units=32))     # 加入Flattern层，变为３２００个神经元
        model.add(Dense(units=256, activation="relu"))  # 隐藏层
        model.add(Dropout(0.5))
        model.add(Dense(units=self.output_dim, activation="softmax"))  # 输出层
        print("神经网络的结构如下：")
        print(model.summary())
        return model

    def text_cnn(self): # 还有些方法不是很懂
        sentence_seq = Input(shape=[self.max_lenth], name="X_seq")  # 输入
        embedding_layer = Embedding(input_dim=self.token_words, output_dim=self.embedding_output)(sentence_seq)  # 词嵌入层
        # 卷积层如下：
        convs = []
        filter_sizes = [2, 3, 4, 5]  # 4种卷积核
        for fsz in filter_sizes:
            l_conv = Conv1D(filters=100, kernel_size=fsz, activation="relu")(embedding_layer)    # 100个(max_lenth-fsz+1, 1)维的向量
            l_pool = MaxPooling1D(self.max_lenth-fsz+1)(l_conv)    # 100个1维向量
            l_pool = Flatten()(l_pool)   # 拉平
            convs.append(l_pool)
        merge = concatenate(convs, axis=1)
        out = Dropout(0.35)(merge)
        output = Dense(512, activation="relu")(out)
        output = Dropout(0.35)(output)
        output = Dense(256, activation="relu")(output)
        output = Dense(self.output_dim, activation="softmax")(output)
        model = Model([sentence_seq], output)
        print("神经网络的结构如下：")
        print(model.summary())
        return model