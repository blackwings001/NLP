# 使用keras进行文本分类
# coding=utf-8
# 使用的网络结构包含MLP， RNN， LSTM, GRU, CNN等。
from helper import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, load_model
from keras.layers import Conv1D, MaxPooling1D, Input, concatenate
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.layers.wrappers import Bidirectional  # 构建双向循环网络
from keras.models import Model
import time


class NetWork:
    def __init__(self, token_words, max_lenth, embedding_output):
        # 嵌入层参数
        self.token_words = token_words                # 字典维度
        self.max_lenth = max_lenth                    # padding后句子长度
        self.embedding_output = embedding_output      # 词向量维度

    def MLP_NETWORK(self):      # 多层感知机神经网络结构
        model = Sequential()    # 堆叠式模型
        # 网络的第一层为Embedding层,嵌入层将单词数字变为向量, output_dim嵌入层的输出维度，每个单词用多少维向量表示。input_length为输入句子长度，　input_dim为字典维度。
        model.add(Embedding(input_dim=self.token_words, input_length=self.max_lenth, output_dim=self.embedding_output))
        model.add(Flatten())  # 加入Flattern层，变为３２００个神经元
        model.add(Dropout(0.2))  # 加入Dropout层，防止过拟合
        model.add(Dense(units=256, activation="relu"))  # 隐藏层
        model.add(Dropout(0.5))
        model.add(Dense(units=20, activation="softmax"))  # 输出层
        print("神经网络的结构如下：")
        print(model.summary())
        return model

    def RNN_NETWORK(self):
        model = Sequential()   # 堆叠式模型
        # 网络的第一层为Embedding层,嵌入层将单词数字变为向量, output_dim嵌入层的输出维度，每个单词用多少维向量表示。input_length为输入句子长度，　input_dim为字典维度。
        model.add(Embedding(input_dim=self.token_words, input_length=self.max_lenth, output_dim=self.embedding_output))
        model.add(Dropout(0.2))  # 加入Dropout层，防止过拟合
        model.add(SimpleRNN(units=16))
        model.add(Dense(units=256, activation="relu"))  # 隐藏层
        model.add(Dropout(0.5))
        model.add(Dense(units=20, activation="softmax"))  # 输出层
        print("神经网络的结构如下：")
        print(model.summary())
        return model

    def BIRNN_NETWORK(self):
        model = Sequential()   # 堆叠式模型
        # 网络的第一层为Embedding层,嵌入层将单词数字变为向量, output_dim嵌入层的输出维度，每个单词用多少维向量表示。input_length为输入句子长度，　input_dim为字典维度。
        model.add(Embedding(input_dim=self.token_words, input_length=self.max_lenth, output_dim=self.embedding_output))
        model.add(Dropout(0.2))  # 加入Dropout层，防止过拟合
        model.add(Bidirectional(SimpleRNN(units=16, return_sequences=True), merge_mode="concat"))     # 加入Flattern层，变为３２００个神经元
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(units=256, activation="relu"))  # 隐藏层
        model.add(Dropout(0.5))
        model.add(Dense(units=20, activation="softmax"))  # 输出层
        print("神经网络的结构如下：")
        print(model.summary())
        return model

    def LSTM_NETWORK(self):
        model = Sequential()   # 堆叠式模型
        # 网络的第一层为Embedding层,嵌入层将单词数字变为向量, output_dim嵌入层的输出维度，每个单词用多少维向量表示。input_length为输入句子长度，　input_dim为字典维度。
        model.add(Embedding(input_dim=self.token_words, input_length=self.max_lenth, output_dim=self.embedding_output))
        model.add(Dropout(0.2))  # 加入Dropout层，防止过拟合
        model.add(LSTM(units=32))     # 加入Flattern层，变为３２００个神经元
        model.add(Dense(units=256, activation="relu"))  # 隐藏层
        model.add(Dropout(0.5))
        model.add(Dense(units=20, activation="softmax"))  # 输出层
        print("神经网络的结构如下：")
        print(model.summary())
        return model

    def GRU_NETWORK(self):
        model = Sequential()   # 堆叠式模型
        # 网络的第一层为Embedding层,嵌入层将单词数字变为向量, output_dim嵌入层的输出维度，每个单词用多少维向量表示。input_length为输入句子长度，　input_dim为字典维度。
        model.add(Embedding(input_dim=self.token_words, input_length=self.max_lenth, output_dim=self.embedding_output))
        model.add(Dropout(0.2))  # 加入Dropout层，防止过拟合
        model.add(GRU(units=32))     # 加入Flattern层，变为３２００个神经元
        model.add(Dense(units=256, activation="relu"))  # 隐藏层
        model.add(Dropout(0.5))
        model.add(Dense(units=20, activation="softmax"))  # 输出层
        print("神经网络的结构如下：")
        print(model.summary())
        return model

    def TEXT_CNN_NETWORK(self): # 还有些方法不是很懂
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
        output = Dense(20, activation="softmax")(output)
        model = Model([sentence_seq], output)
        print("神经网络的结构如下：")
        print(model.summary())
        return model

if __name__ == '__main__':
    # import os
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # 获取bunch中的labels和contents
    print("正在获取数据》》》》")
    allbunch = genBunch(DATA_FILE, STOPWORDS_PATH)

    # 将labels处理为20维的one_hot向量
    transformer = MultiLabelBinarizer()
    allbunch.labels = [[CLASS_DICT[i]] for i in allbunch.labels]
    allbunch.new_lables = transformer.fit_transform(allbunch.labels)

    # 相关参数如下：
    token_words = 2000   # 单词字典的单词数。
    max_lenth = 100      # 选取句子的长度。
    output_dim = 32      # 词向量维度

    print("正在进行数据预处理》》》》")
    st = time.time()
    # 建立Token词典
    token = Tokenizer(num_words=token_words)  # 设置词典规模
    token.fit_on_texts(allbunch.contents)   # 建立字典模型

    # 将文字列表转化为数字列表
    allbunch.contents_seq = token.texts_to_sequences(allbunch.contents)
    # 对数字列表进行padding，截长补短。处理后的数据输入神经网络进行训练。
    allbunch.contents_seq_pad = sequence.pad_sequences(allbunch.contents_seq, maxlen=max_lenth)

    train_x, test_x, train_y, test_y = train_test_split(allbunch.contents_seq_pad, allbunch.new_lables, stratify=allbunch.new_lables)
    et = time.time()
    print("数据预处理完成！！！！用时：{:.3f}s".format(et-st))

    # 构建神经网络，
    network = NetWork(token_words, max_lenth, output_dim)
    model = network.TEXT_CNN_NETWORK()

    # 定义训练方法
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

    # 开始训练
    train_history = model.fit(train_x, train_y, batch_size=100, epochs=30, verbose=1, validation_split=0.2)

    # 评估模型准确率
    score = model.evaluate(test_x, test_y, verbose=2)
    print("模型准确率为：{:.3f}".format(score[1]))


