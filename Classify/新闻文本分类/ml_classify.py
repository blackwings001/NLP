# 主函数完成对预处理后文件的特征选择，特征权重计算以及训练预测
from helper import *
from sklearn import model_selection, svm, preprocessing, linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics
from sklearn.datasets.base import Bunch

from time import time





#　输出分类的评估结果
def printResult(predict_y, true_y):
    accuracy = metrics.accuracy_score(y_true=true_y, y_pred=predict_y)
    print("准确率如下：{:.3f}".format(accuracy))
    micro_precision = metrics.precision_score(y_true=true_y, y_pred=predict_y, average="micro")
    print("微平均准确率为：{}".format(micro_precision))
    macro_precision = metrics.precision_score(y_true=true_y, y_pred=predict_y, average="macro")
    print("宏平均准确率为：{}".format(macro_precision))
    micro_recall = metrics.recall_score(y_true=true_y, y_pred=predict_y, average="micro")
    print("微平均召回率为：{}".format(micro_recall))
    macro_recall = metrics.recall_score(y_true=true_y, y_pred=predict_y, average="macro")
    print("宏平均召回率为：{}".format(macro_recall))
    report = metrics.classification_report(y_true=true_y, y_pred=predict_y)
    print("分类报告如下：\n" + report)




if __name__ == '__main__':


    # 获取allbunch中的labels和contents
    allbunch = genBunch(DATA_FILE, STOPWORDS_PATH)

    label_dict = {}
    for label in allbunch.labels:
        if label not in label_dict.keys():
            label_dict[label] = 1
        else:
            label_dict[label] += 1
    print(label_dict)

    allbunch.labels = [CLASS_DICT[i] for i in allbunch.labels]                     # 先将标签转化为数字, 类型为列表，训练分类器的时候用

    # 获取tfidf值
    print("开始提取tfidf值》》》》")
    word2tfidf = TfidfVectorizer()
    allbunch.tfidf = word2tfidf.fit_transform(allbunch.contents)
    print("tfidf维度为:{}".format(allbunch.tfidf.shape))

    # 进行基于SVM模型的特征选择
    selector = SelectFromModel(svm.LinearSVC(C=1, penalty="l1", dual=False))
    allbunch.select_features = selector.fit_transform(allbunch.tfidf, allbunch.labels)
    print("选择后的tfidf维度为:{}".format(allbunch.select_features.shape))

    # 将数据分为训练集和测试集, 进行分类训练, 变成列表后才可以及进行切分
    X = allbunch.select_features
    Y = allbunch.labels

    train_x, test_x, train_y, test_y = model_selection.train_test_split(X, Y, stratify=Y)  # stratrify=Y 使用分层抽样

    # 进行训练
    print("开始进行训练》》》》"); st = time()

    # 指定分类器
    # clf = svm.SVC(kernel="linear", probability=True)
    clf = linear_model.LogisticRegression(n_jobs=-1)
    clf.fit(train_x, train_y)
    predict_y = clf.predict(test_x)

    print("分类训练及预测总计用时：{:.3f}秒".format(time() - st))

    # 输出分类结果
    printResult(predict_y, test_y)






