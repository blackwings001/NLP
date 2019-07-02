import gensim
import numpy as np
from time import time

class WordVectorModel:
    def __init__(self, model_file):
        st = time()
        if ".model" in model_file:
            print("正在加载模型, 大约需要100秒...")
            self.model = gensim.models.Word2Vec.load(model_file)
            # 查看可用的方法
            # self.model.wv.
        elif ".bin" in model_file:
            print("正在加载模型, 大约需要60秒...")
            self.model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=True)
        elif ".npy" in model_file:
            matrix = np.load(model_file); print("词向量矩阵的维度为: {}".format(matrix.shape))

        print("加载模型用时：{}秒！".format(time()-st))

    def word_basic_info(self, word, basic_file):
        f = open(basic_file, "w", encoding="utf-8")
        word_vector = self.model.wv.get_vector(word); f.writelines("'{}'的词向量为：\n {}".format(word, word_vector) + "\n"); f.writelines("'{}'的词向量维度是：{}".format(word, word_vector.shape[0]) + "\n")
        similar_words = self.model.wv.most_similar(word); f.writelines("'{}'的最相似词为：\n {}".format(word, similar_words) + "\n")
        f.close()

    def two_words_similar(self, word1, word2):
        similarity = self.model.wv.similarity(word1, word2)
        print("'{}'和'{}'的相似度为：{}".format(word1, word2, similarity))

if __name__ == '__main__':
    MODEL_FILE = "model/" + "baike_26g_news_13g_novel_229g.bin"
    WORD_BASIC_INFO_FILE = "word_vector_basic_info.txt"
    WORD = "开心"

    wv_model = WordVectorModel(MODEL_FILE)
    wv_model.word_basic_info(WORD, WORD_BASIC_INFO_FILE)
    wv_model.two_words_similar("蓝色", "羽绒服")
    wv_model.two_words_similar("篮色", "羽绒服")
