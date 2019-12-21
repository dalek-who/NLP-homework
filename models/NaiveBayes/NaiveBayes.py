import pandas as pd
import jieba
import os

class NaiveBayes():
    total0=506762
    total1=2925103
    total2=177864
    count0=44080
    count2=87430
    count1=87450
    dict0={}
    dict1={}
    dict2={}
    d0=[]
    d1=[]
    d2=[]
    def __init__(self):
        dir = os.path.dirname(os.path.abspath(__file__))
        self.d0=pd.read_csv(dir+'/wordset0.csv').astype(str)
        self.d0=self.d0.iloc[:,0:2]
        self.d1=pd.read_csv(dir+'/wordset1.csv').astype(str)
        self.d1=self.d1.iloc[:,0:2]
        self.d2=pd.read_csv(dir+'/wordset2.csv').astype(str)
        self.d2=self.d2.iloc[:,0:2]
        for i in range(0,self.count0):
            self.dict0[self.d0['item'][i]]=self.d0['weight'][i]
        for i in range(0,self.count2):
            self.dict2[self.d2['item'][i]]=self.d2['weight'][i]
        for i in range(0,self.count1):
            self.dict1[self.d1['item'][i]]=self.d1['weight'][i]


    def classification(self, text: str):
        wordlist=(jieba.cut(text))
        p0=1    #当前文本情感类型为0的相对概率
        p1=1    #当前文本情感类型为1的相对概率
        p2=1    #当前文本情感类型为2的相对概率
        result=3

        for i in wordlist:
            num=1
            if i in self.dict0:
               num+=int(self.dict0[i])                 #计算相对概率，采用加一平滑避免相对概率为0
            p0=p0*1000*num/(self.total0 + 1)
            num=1
            if i in self.dict1:
               num+=int(self.dict1[i])
            p1=p1*1000*num/(self.total1 + 1)
            num=1
            if i in self.dict2:
               num+=int(self.dict2[i])
            p2=p2*1000*num/(self.total2 + 1)

        p0=p0*1000*(self.total0 + 1)/(self.total0 + self.total1 + self.total2 + 1)
        p1=p1*1000*(self.total1 + 1)/(self.total0 + self.total1 + self.total2 + 1)
        p2=p2*1000*(self.total2 + 1)/(self.total0 + self.total1 + self.total2 + 1)

        if (p0>max(p1,p2)):
            result=0
        if (p1>max(p0,p2)):
            result=1
        if (p2>max(p0,p1)):
            result=2

        return result
