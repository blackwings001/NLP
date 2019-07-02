# 储存文件路径, 基本参数和辅助函数
import os
from sklearn.datasets.base import Bunch

BASE_DIR = os.getcwd()

DATA_FILE = os.path.join(BASE_DIR, "data", "news_process_data.txt")                 # 新闻数据文件
STOPWORDS_PATH = os.path.join(BASE_DIR, "data", "stopWords/stopWords.txt")          # 停止词文件

# 类别列表
CLASS_LIST = [3,4,5,6,7,11,15,16,17,19,23,29,31,32,34,35,36,37,38,39]

CLASS_DICT = {"C3-Art": 3, "C4-Literature": 4, "C5-Education": 5, "C6-Philosophy": 6, "C7-History": 7,
                "C11-Space": 11, "C15-Energy": 15, "C16-Electronics": 16, "C17-Communication": 17,
                "C19-Computer": 19,
                "C23-Mine": 23, "C29-Transport": 29, "C31-Enviornment": 31, "C32-Agriculture": 32,
                "C34-Economy": 34,
                "C35-Law": 35, "C36-Medical": 36, "C37-Military": 37, "C38-Politics": 38, "C39-Sports": 39, }



# 获得所有样本内容以及标签，分别储存在allbunch中的labels和contents中。
def genBunch(processDataPath, stopWordsPath):
    # 读取停止词列表
    with open(stopWordsPath, "r", encoding="utf-8") as f:
        stopWordsList = f.read().splitlines()

    allbunch = Bunch(labels=[], contents=[])
    with open(processDataPath, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip()

            label = line.split(",")[0]
            content = line.split(",")[-1]

            # 去掉停用词， 建议在数据预处理的时候就去掉停用词
            # contentList = content.split()
            # content = " ".join(word for word in contentList if word not in stopWordsList)

            allbunch.labels.append(label)
            allbunch.contents.append(content)

    return allbunch

if __name__ == '__main__':
    print(BASE_DIR)
