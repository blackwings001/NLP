"""
Author: guominglei 
Date: 2019-07-15
Readme: 使用深度学习方法进行文本分类，涉及多种网络结构
"""
import os
from time import time
from sklearn.externals import joblib
from sklearn.datasets.base import Bunch
from utility import NEWS_FILE, STOPWORDS_FILE, CATE_LIST, CATE_DICT, DL_MODEL_PATH, DL_PROCESSED_PATH

class DlClassify:
    def __init__(self, NEWS_FILE, STOPWORDS_FILE, CATE_LIST, CATE_DICT, DL_PROCESSED_PATH, process_data=True):
        # 初始化文件路径以及相关参数

        self.news_file = NEWS_FILE
        self.stopwords_file = STOPWORDS_FILE
        self.cate_list = CATE_LIST
        self.cate_dict = CATE_DICT

        self.processed_data_file = DL_PROCESSED_PATH + "/data.pkl"
        self.processed_data_exists = False

        self.data_bunch = Bunch(labels=[], contents=[])

        if not process_data and os.path.exists(self.processed_data_file):
            self.processed_data_exists = True


    def process_raw_data(self):
        # 使用bunch保存raw_data中的content和label
        t = time()

        if self.processed_data_exists:
            return

        stop_words = open(self.stopwords_file, "r", encoding="utf-8").read().splitlines()
        sum_lines = len(open(self.news_file, "r", encoding="utf-8").read().splitlines())

        with open(self.news_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f.readlines()):
                label = line.split(",")[0]
                content = line.split(",")[-1].strip()

                # 去掉停用词， 建议在数据预处理的时候就去掉停用词
                # content = " ".join(word for word in content.split() if word not in stop_words)
                if label in ('C11-Space', 'C3-Art', 'C7-History', 'C31-Enviornment', 'C32-Agriculture', "C34-Economy", "C19-Computer", 'C38-Politics', "C39-Sports"):
                    self.data_bunch.labels.append(label)
                    self.data_bunch.contents.append(content)

                if i % 1000 == 0:
                    print("使用bunch保存content和label，进度：{:.2f}%".format(i / sum_lines * 100))

        print("使用bunch保存content和label, 完成！用时:{:.2f}s".format(time() - t))



    def process_feature_label(self, token_words, max_lenth, output_dim):
        """
        :param token_words: 单词字典的单词数。
        :param max_lenth: 选取句子的长度。
        :param output_dim: 词向量维度
        :return: 将数据处理成神经网络需要的格式
        """
        from sklearn.preprocessing import MultiLabelBinarizer
        from keras.preprocessing import sequence
        from keras.preprocessing.text import Tokenizer

        t = time()

        self.token_words = token_words
        self.max_lenth = max_lenth
        self.output_dim = output_dim

        if self.processed_data_exists:
            return

        # 将labels处理为one_hot向量
        transformer = MultiLabelBinarizer()
        self.data_bunch.labels = [[CATE_DICT[i]] for i in self.data_bunch.labels]
        self.data_bunch.lables_one_hot = transformer.fit_transform(self.data_bunch.labels)

        # 将contents进行token并且padding
        token = Tokenizer(num_words=token_words)  # 设置词典规模
        token.fit_on_texts(self.data_bunch.contents)  # 建立字典模型

        self.data_bunch.contents_seq = token.texts_to_sequences(self.data_bunch.contents)
        self.data_bunch.contents_seq_pad = sequence.pad_sequences(self.data_bunch.contents_seq, maxlen=max_lenth)

        joblib.dump(self.data_bunch.contents_seq_pad, self.processed_data_file)
        joblib.dump(self.data_bunch.lables_one_hot, self.processed_data_file)

        print("处理feature和label, 完成！用时:{:.2f}s".format(time() - t))



    def train_model(self, batch_size, epochs):
        """
        使用不同的分类器进行分类模型训练
        :param model_cate: 选择不同的分类器
        :param save_model: 是否保存模型
        :return: 正确标签和预测标签
        """
        # 将数据分为训练集和测试集, 进行分类训练, 变成列表后才可以及进行切分、
        from dl_network import Network
        from sklearn.model_selection import train_test_split
        from numpy import argmax

        t = time()

        if self.processed_data_exists:
            X = joblib.load(self.processed_data_file)
            Y = joblib.load(self.processed_data_file)
        else:
            X = self.data_bunch.contents_seq_pad
            Y = self.data_bunch.lables_one_hot

        train_x, test_x, train_y, test_y = train_test_split(X, Y, stratify=Y)

        net = Network(self.token_words, self.max_lenth, self.output_dim, Y.shape[1])

        model = net.text_cnn()
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
        train_history = model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(test_x, test_y))

        # 评估模型准确率
        score = model.evaluate(test_x, test_y, verbose=2)
        print("模型准确率为：{:.3f}".format(score[1]))

        predict_y = model.predict(test_x)
        self.predict_y = [argmax(one_hot) for one_hot in predict_y]
        self.true_y = [argmax(one_hot) for one_hot in test_y]

        print("训练模型完成！用时:{:.2f}s".format(time() - t))


    def evaluate_result(self):
        t = time()
        from sklearn import metrics
        result = {}
        result["accuracy"] = metrics.accuracy_score(y_true=self.true_y, y_pred=self.predict_y)
        result["micro_precision"] = metrics.precision_score(y_true=self.true_y, y_pred=self.predict_y, average="micro")
        result["macro_precision"] = metrics.precision_score(y_true=self.true_y, y_pred=self.predict_y, average="macro")
        result["micro_recall"] = metrics.recall_score(y_true=self.true_y, y_pred=self.predict_y, average="micro")
        result["macro_recall"] = metrics.recall_score(y_true=self.true_y, y_pred=self.predict_y, average="macro")
        result["report"] = metrics.classification_report(y_true=self.true_y, y_pred=self.predict_y)

        print("评估结果完成！用时:{:.2f}s".format(time() - t))

        with open("dl_metrics.txt", "w", encoding="utf-8") as f:
            for key, value in result.items():
                f.writelines("评价指标为: {}\n{}\n".format(key, value))
    
    
    


if __name__ == '__main__':

    TOKEN_WORDS = 2000
    MAX_LENGTH = 100
    OUTPUT_DIM = 32

    BATCH_SIZE = 256
    EPOCHS = 10

    dlclassify = DlClassify(NEWS_FILE, STOPWORDS_FILE, CATE_LIST, CATE_DICT, DL_PROCESSED_PATH, process_data=False)
    dlclassify.process_raw_data()
    dlclassify.process_feature_label(token_words=TOKEN_WORDS, max_lenth=MAX_LENGTH, output_dim=OUTPUT_DIM)
    dlclassify.train_model(BATCH_SIZE, EPOCHS)
    dlclassify.evaluate_result()





