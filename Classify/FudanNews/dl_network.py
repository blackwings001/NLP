"""
Author: guominglei
Date: 2019-07-15
Readme: 本脚本保存神经网络模型
"""
import keras
from keras.models import Sequential, load_model
from keras.layers import Conv1D, MaxPooling1D, Input, concatenate, GlobalAveragePooling1D, Permute, multiply
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

    def fasttext(self):
        """
        fasttext的优点是非常迅速，可以在文本处理的时候使用bigram，trigram词组特征，从而考虑到词序问题，效果也会更好一点
        """
        model = Sequential()
        model.add(Embedding(input_dim=self.token_words, input_length=self.max_lenth, output_dim=self.embedding_output))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(units=self.output_dim, activation="softmax"))
        print("fasttext的神经网络结构如下：\n", model.summary())
        return model


    def textcnn(self):
        """
        使用textcnn进行文本分类
        :return: textcnn模型
        """
        input = Input(shape=[self.max_lenth], name="X_seq")
        embedding_layer = Embedding(input_dim=self.token_words, output_dim=self.embedding_output)(input)

        convs = []
        filter_sizes = [2, 3, 4, 5]  # 4种卷积核
        for fsz in filter_sizes:
            l_conv = Conv1D(filters=100, kernel_size=fsz, activation="relu")(embedding_layer)    # 100个(max_lenth-fsz+1, 1)维的向量
            l_pool = MaxPooling1D(self.max_lenth - fsz + 1)(l_conv)    # 100个1维向量
            l_pool = Flatten()(l_pool)   # 拉平
            convs.append(l_pool)
        merge = concatenate(convs, axis=1)

        out = Dropout(0.35)(merge)
        print(out)
        print(out.shape)
        output = Dense(512, activation="relu")(out)
        output = Dropout(0.35)(output)
        output = Dense(256, activation="relu")(output)
        output = Dense(self.output_dim, activation="softmax")(output)
        model = Model([input], output)
        print("textcnn的神经网络结构如下：\n", model.summary())
        return model

    def textrnn(self):
        """
        使用的是双向lstm，textcnn的问题是filter_size固定了视野，无法获得更长的序列信息
        双向LSTM的输出可以有很多种，例如全序列输出的平均值，序列最后位置的输出等，这里采用的是平均值。
        :return:
        """
        model = Sequential()
        model.add(Embedding(input_dim=self.token_words, input_length=self.max_lenth, output_dim=self.embedding_output))
        model.add(Bidirectional(LSTM(units=64, return_sequences=True), merge_mode="concat"))     # 加入Flattern层，变为３２００个神经元
        model.add(GlobalAveragePooling1D())
        model.add(Dropout(0.5))
        model.add(Dense(units=64, activation="relu"))  # 隐藏层
        model.add(Dropout(0.5))
        model.add(Dense(units=self.output_dim, activation="softmax"))  # 输出层
        print("textrnn的神经网络结构如下：\n", model.summary())
        return model


    def textrnn_attention(self):
        """
        对上面的模型使用attention机制，包含词级别的attention和句子级别的attention
        :return:
        """

        def attention_block(input):
            """
            实现attention功能
            :param input: 双向lstm的输出结果
            :return: 经过attention的输出
            """
            attention_input = Permute((2,1))(input)
            attention_weight = Dense(self.max_lenth, activation="softmax")(attention_input)
            attention_weight = Permute((2, 1), name="attention_weight")(attention_weight)
            attention_output = multiply([input, attention_weight], name="attention_mul")
            return attention_output

        input = Input(shape=[self.max_lenth], name="X_seq")
        embedding_layer = Embedding(input_dim=self.token_words, output_dim=self.embedding_output)(input)
        lstm_output = Bidirectional(LSTM(units=64, return_sequences=True), merge_mode="concat")(embedding_layer)
        attention = attention_block(lstm_output)

        output = Flatten()(attention)
        output = Dense(512, activation="relu")(output)
        output = Dropout(0.35)(output)
        output = Dense(256, activation="relu")(output)
        output = Dense(self.output_dim, activation="softmax")(output)
        model = Model([input], output)
        print("textcnn的神经网络结构如下：\n", model.summary())
        return model





    def mlp(self):
        """
        一般文本分类不会使用mlp，这里只是测试一下mlp进行文本分类的效果
        """
        model = Sequential()
        model.add(Embedding(input_dim=self.token_words, input_length=self.max_lenth, output_dim=self.embedding_output))
        model.add(Flatten()).add(Dropout(0.2))
        model.add(Dense(units=256, activation="relu")).add(Dropout(0.5))
        model.add(Dense(units=self.output_dim, activation="softmax"))
        print("mlp的神经网络结构如下：\n", model.summary())
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
        model.add(Bidirectional(SimpleRNN(units=24, return_sequences=True), merge_mode="concat"))     # 加入Flattern层，变为３２００个神经元
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
