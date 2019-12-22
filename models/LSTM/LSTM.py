from pathlib import Path
from multiprocessing import Queue, Process, current_process
from argparse import ArgumentParser
import json
import jieba
from keras.models import load_model
import re
import os
import numpy as np
from keras.preprocessing.sequence import pad_sequences

from gensim.models.word2vec import Word2Vec
if __name__ =='__main__':
    parser = ArgumentParser()
    parser.add_argument("--text", type=str)
    args = parser.parse_args()
    text = args.text
    input_shape = 200
    path = Path(__file__)

    model = load_model(path.parent / 'LSTM.h5')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # w2v_file = path.parent /'trained.kv'
    w2v_file: Path = str((path.parent / 'trained.kv').absolute())
    w2v_model = Word2Vec.load(w2v_file)

    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    text = re.sub(pattern, '', text)
    vec = []
    for word in jieba.lcut(text):
        if len(vec) < input_shape:
            try:
                vec.append(w2v_model[word])
            except KeyError:
                pass
    x = pad_sequences(maxlen=input_shape, sequences=[vec], padding='post', value=0)
    y = model.predict(x)
    result = np.argmax(y)
    # print(result)

    with open(path.parent / "result.txt", "w") as f:
        f.write(str(result))