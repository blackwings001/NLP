# 储存文件路径, 基本参数和辅助函数
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))   # NLP文件夹路径

NEWS_FILE = os.path.join(BASE_DIR, "Data", "FudanNews", "process_data.txt")         # 新闻数据文件夹
STOPWORDS_FILE = os.path.join(BASE_DIR, "Data", "StopWords", "stopWords.txt")       # 停止词文件

ML_MODEL_PATH = os.path.join(BASE_DIR, "Model", "Classify", "FudanNews", "Ml")
DL_MODEL_PATH = os.path.join(BASE_DIR, "Model", "Classify", "FudanNews", "Dl")

ML_PROCESSED_PATH = os.path.join(BASE_DIR, "Data", "FudanNews", "Ml")
DL_PROCESSED_PATH = os.path.join(BASE_DIR, "Data", "FudanNews", "Dl")

CATE_LIST = [3,4,5,6,7,11,15,16,17,19,23,29,31,32,34,35,36,37,38,39]
CATE_DICT = {"C3-Art": 3, "C4-Literature": 4, "C5-Education": 5, "C6-Philosophy": 6,
             "C7-History": 7, "C11-Space": 11, "C15-Energy": 15, "C16-Electronics": 16,
             "C17-Communication": 17, "C19-Computer": 19, "C23-Mine": 23, "C29-Transport": 29,
             "C31-Enviornment": 31, "C32-Agriculture": 32, "C34-Economy": 34, "C35-Law": 35,
             "C36-Medical": 36, "C37-Military": 37, "C38-Politics": 38, "C39-Sports": 39, }





if __name__ == '__main__':
    print(BASE_DIR)
