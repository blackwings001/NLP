"""
Author: guominglei
Date: 2019-07-09
Readme: 使用机器学习方法进行文本分类
"""
from time import time
from utility import NEWS_FILE, STOPWORDS_FILE, CATE_LIST, CATE_DICT, ML_MODEL_PATH

class MlClassify:
    def __init__(self, NEWS_FILE, STOPWORDS_FILE, CATE_LIST, CATE_DICT, ML_MODEL_PATH):
        # 初始化文件路径以及相关参数
        from sklearn.datasets.base import Bunch

        self.news_file = NEWS_FILE
        self.stopwords_file = STOPWORDS_FILE
        self.cate_list = CATE_LIST
        self.cate_dict = CATE_DICT
        self.model_path = ML_MODEL_PATH

        self.data_bunch = Bunch(labels=[], contents=[])


    def process_raw_data(self):
        # 使用bunch保存raw_data中的content和label
        t = time()

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



    def process_feature_label(self, feature_extraction=False):
        """
        处理feature和label，用于最后的训练
        :param feature_extraction: 是否进行特征选择
        """
        t = time()

        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf = TfidfVectorizer()

        self.data_bunch.labels = [self.cate_dict[label] for label in self.data_bunch.labels]    # 将labels由文字变为数字
        self.data_bunch.tfidfs = tfidf.fit_transform(self.data_bunch.contents)
        print("tfidfs维度为:{}".format(self.data_bunch.tfidfs.shape))

        if feature_extraction:
            from sklearn.feature_selection import SelectFromModel
            from sklearn import svm

            selector = SelectFromModel(svm.LinearSVC(C=1, penalty="l1", dual=False))
            self.data_bunch.tfidfs = selector.fit_transform(self.data_bunch.tfidfs, self.data_bunch.labels)
            print("选择后的tfidfs维度为:{}".format(self.data_bunch.tfidfs.shape))

        print("处理feature和label, 完成！用时:{:.2f}s".format(time() - t))



    def train_model(self, model_cate=1, save_model=False):
        """
        使用不同的分类器进行分类模型训练
        :param model_cate: 选择不同的分类器
        :param save_model: 是否保存模型
        :return: 正确标签和预测标签
        """
        # 将数据分为训练集和测试集, 进行分类训练, 变成列表后才可以及进行切分
        t = time()

        from sklearn import model_selection

        X = self.data_bunch.tfidfs
        Y = self.data_bunch.labels
        train_x, test_x, train_y, test_y = model_selection.train_test_split(X, Y, test_size=0.3, stratify=Y)  # stratrify=Y 使用分层抽样

        from sklearn import svm
        from sklearn import linear_model

        if model_cate == 0:
            clf = svm.SVC(kernel="linear")
        elif model_cate == 1:
            clf = linear_model.LogisticRegression(n_jobs=-1)
        else:
            clf = svm.SVC(kernel="linear")

        clf.fit(train_x, train_y)
        predict_y = clf.predict(test_x)

        print("训练模型完成！用时:{:.2f}s".format(time() - t))

        if save_model == True:
            from sklearn.externals import joblib
            joblib.dump(clf, self.model_path + "/clf.m")

        self.predict_y = predict_y
        self.true_y = test_y


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

        with open("ml_metrics.txt", "w", encoding="utf-8") as f:
            for key, value in result.items():
                f.writelines("评价指标为: {}\n{}\n".format(key, value))




if __name__ == '__main__':
    mlclassify = MlClassify(NEWS_FILE, STOPWORDS_FILE, CATE_LIST, CATE_DICT, ML_MODEL_PATH)
    mlclassify.process_raw_data()
    mlclassify.process_feature_label()
    mlclassify.train_model()
    mlclassify.evaluate_result()






