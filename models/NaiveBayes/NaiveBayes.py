import pandas as pd
import jieba
import os


class NaiveBayes():
    ######Type0,Type1,Type2分别为0，1，2类文本的词典，记录单词及该单词出现在多少个文本中

    total0 = 525  # Train.csv中label为0的文本数量
    total1 = 2379  # Train.csv中label为1的文本数量
    total2 = 1964  # Train.csv中label为2的文本数量
    count0 = 35715  # 出现在0类文本中的不同单词的总数
    count1 = 115980  # 出现在1类文本中的不同单词的总数
    count2 = 70500  # 出现在2类文本中的不同单词的总数
    dict0 = {}  # 存放出现在0类文本的单词，该单词出现在多少个0类文本中
    dict1 = {}  # 存放出现在1类文本的单词，该单词出现在多少个0类文本中
    dict2 = {}  # 存放出现在2类文本的单词，该单词出现在多少个0类文本中
    d0 = []  # 读入Type0.csv
    d1 = []  # 读入Type1.csv
    d2 = []  # 读入Type2.csv

    def __init__(self):  # 加载Type0，将其导入dict0，其余同理
        dir = os.path.dirname(os.path.abspath(__file__))
        self.d0 = pd.read_csv(dir + '/Type0.csv').astype(str)
        self.d0 = self.d0.iloc[:, 0:2]
        self.d1 = pd.read_csv(dir + '/Type1.csv').astype(str)
        self.d1 = self.d1.iloc[:, 0:2]
        self.d2 = pd.read_csv(dir + '/Type2.csv').astype(str)
        self.d2 = self.d2.iloc[:, 0:2]
        for i in range(0, self.count0):
            self.dict0[self.d0['item'][i]] = self.d0['weight'][i]
        for i in range(0, self.count1):
            self.dict1[self.d1['item'][i]] = self.d1['weight'][i]
        for i in range(0, self.count2):
            self.dict2[self.d2['item'][i]] = self.d2['weight'][i]

    def classification(self, text: str):
        wordlist = (jieba.cut(text))
        p0 = 1  # 当前文本情感类型为0的相对概率
        p1 = 1  # 当前文本情感类型为1的相对概率
        p2 = 1  # 当前文本情感类型为2的相对概率
        result = 3

        for i in wordlist:
            num = 1
            if i in self.dict0:
                num += int(self.dict0[i])  # 计算相对概率，采用加一平滑避免相对概率为0
            p0 = p0 * 10 * num / (self.total0 + 1)
            num = 1
            if i in self.dict1:
                num += int(self.dict1[i])
            p1 = p1 * 10 * num / (self.total1 + 1)
            num = 1
            if i in self.dict2:
                num += int(self.dict2[i])
            p2 = p2 * 10 * num / (self.total2 + 1)

        p0 = p0 * 10 * (self.total0 + 1) / (self.total0 + self.total1 + self.total2 + 1)
        p1 = p1 * 10 * (self.total1 + 1) / (self.total0 + self.total1 + self.total2 + 1)
        p2 = p2 * 10 * (self.total2 + 1) / (self.total0 + self.total1 + self.total2 + 1)

        if (p0 > max(p1, p2)):
            result = 0
        if (p1 > max(p0, p2)):
            result = 1
        if (p2 > max(p0, p1)):
            result = 2

        return result
