from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import to_categorical
from keras.models import load_model

import os
import pandas as pd
import jieba
import re
import numpy as np
from pathlib import Path
from multiprocessing import Queue, Process, current_process
from argparse import ArgumentParser
import json


vocab_size = 1000#单词总数
input_shape = 200#序列长度
vocab_dim = 100
batch_size = 128
epoch_n = 30
label_size = 2


word_dic_path = 'word_dict.pk'

def find_chinese(file):
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    remain_chinese = re.sub(pattern, '', file)
    return remain_chinese

def load_data(filepath):
    data = pd.read_csv(filepath)
    x = []
    y = []
    word_list = [line.strip() for line in open('sorted_words.txt',encoding='utf-8').readlines() ]
    word_dic = {word:i+1 for i,word in enumerate(word_list)}
    for i in range(len(data)):
        x_temp = []
        title = str(data.iloc[i]['title'])
        content = str(data.iloc[i]['content'])
        string = find_chinese(title + content)
        for word in jieba.lcut(string):
            if word in word_list:
                x_temp.append(word_dic[word])
        x.append(x_temp)
        y.append(data.iloc[i]['label'])
    x = pad_sequences(maxlen=input_shape, sequences=x, padding='post', value=0)
    # y = [np_utils.to_categorical(str(label), num_classes=label_size) for label in y]
    y = to_categorical(y,num_classes=3)
    return x,y


def load_data1(filepath):
    data = pd.read_csv(filepath)
    x,y=[],[]
    word_list = [line.strip() for line in open('sorted_words.txt', encoding='utf-8').readlines()]
    for i in range(len(data)):
        title_vec,content_vec = [0 for _ in range(1000)],[0 for _ in range(1000)]
        title = find_chinese(str(data.iloc[i]['title']))
        content = find_chinese(str(data.iloc[i]['content']))
        for word in jieba.lcut(title):
            if word in word_list:
                title_vec[word_list.index(word)]=1
        for word in jieba.lcut(content):
            if word in word_list:
                content_vec[word_list.index(word)]=1
        x.append([title_vec,content_vec])
        y.append(data.iloc[i]['label'] )
    x = np.array(x)
    y = to_categorical(y,num_classes=3)
    return x,y

def train_lstm(model_path,train_x, train_y, test_x, test_y):
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = Sequential()
        # model.add(Embedding(input_dim=vocab_size + 1,
        #                     output_dim=vocab_dim,
        #                     mask_zero=True,
        #                     input_length=input_shape))
        model.add(LSTM(units=50,
                       input_shape=(2,1000),
                       activation='relu',
                       recurrent_activation='relu',
                       # return_sequences=True,
                       dropout=0.5
                       ))
        model.add(Dense(100))
        model.add(Activation('relu'))
        # model.add(Dense(50))
        # model.add(Activation('relu'))
        model.add(Dropout(0.5))
        # model.add(LSTM(units=30,
        #                activation='sigmoid',
        #                recurrent_activation='hard_sigmoid'))
        # model.add(Dropout(0.5))
        model.add(Dense(3))
        model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(train_x, train_y, batch_size=32, epochs=epoch_n,verbose=2)
    model.summary()
    score, acc = model.evaluate(test_x, test_y, batch_size=batch_size)
    print('Test score:', score)
    print ('Test accuracy:', acc)
    model.save(model_path)


# class LSTM_API(BaseAPI):
#     def __init__(self):
#         super(LSTM_API, self).__init__()
#         self.path = Path(__file__)
#         self.model = load_model(self.path.parent / 'LSTM.h5')
#         self.model.compile(loss='categorical_crossentropy',
#                       optimizer='adam',
#                       metrics=['accuracy'])
#
#     def run_example(self, text: str):
#         word_list = [line.strip() for line in open(self.path.parent / 'sorted_words.txt', encoding='utf-8').readlines()]
#         str = find_chinese(text)
#         title_vec ,content_vec=[0 for _ in range(1000)],[0 for _ in range(1000)]
#         for word in jieba.lcut(text):
#             if word in word_list:
#                 title_vec[word_list.index(word)]=1
#                 content_vec[word_list.index(word)] = 1
#         x = np.array([title_vec,content_vec])
#         x = np.array([x])
#         y = self.model.predict(x)
#         return np.argmax(y)



if __name__ =='__main__':
    # train_filepath = 'train.csv'
    # test_filepath = 'test.csv'
    # model_path = 'LSTM.h5'
    # model = load_model(model_path)
    # model.summary()
    # train_x ,train_y = load_data1(train_filepath)
    # test_x ,test_y = load_data1(test_filepath)
    # train_lstm(model_path,train_x,train_y,test_x,test_y)

    # model = LSTM_API()
    # res =model.run_example('"热烈庆祝中华人民共和国成立七十周年\n 喜闻乐见，大快人心，普天同庆，奔走相告"')
    # print(res)
    parser = ArgumentParser()
    parser.add_argument("--text", type=str)
    args = parser.parse_args()
    text = args.text

    path = Path(__file__)
    model = load_model(path.parent / 'LSTM.h5')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    word_list = [line.strip() for line in open(path.parent / 'sorted_words.txt', encoding='utf-8').readlines()]
    title_vec, content_vec = [0 for _ in range(1000)], [0 for _ in range(1000)]
    for word in jieba.lcut(text):
        if word in word_list:
            title_vec[word_list.index(word)] = 1
            content_vec[word_list.index(word)] = 1
    x = np.array([title_vec, content_vec])
    x = np.array([x])
    y = model.predict(x)
    result = np.argmax(y)
    with open(path.parent / "result.txt", "w") as f:
        f.write(str(result))

