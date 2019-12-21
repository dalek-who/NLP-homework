#!/usr/bin/env python
from collections import Counter
import jieba
import csv
import pandas as pd

from BaseAPI import BaseAPI
from models.NaiveBayes.NaiveBayes import NaiveBayes

class NaiveBayes_API(BaseAPI):
    def __init__(self):
        super(NaiveBayes_API, self).__init__()
        self.model = NaiveBayes()            # 加载你自己的模型对象NaiveBayes,代码实现在NaiveBayes.py

    def run_example(self, text: str):
        result = self.model.classification(text)                   #.......运行你的模型得到结果
        return result


if __name__ == "__main__":
    interface = NaiveBayes_API()
    text = "小明是个好孩子，大家都喜欢他"
    result = interface.run_example(text)