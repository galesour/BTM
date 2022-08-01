"""
@Author: 一蓑烟雨任平生
@Date: 2020-02-18 17:08:33
@LastEditTime: 2020-03-08 15:54:21
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /BTMpy/src/main.py
"""
# -*- coding: utf-8 -*-
import time
from Model import *


def train_BTM():
    print("===== Run BTM, Topic Number=" + str(K) + ", alpha=" + str(alpha) + ", beta=" +
          str(beta) + ", n_iter=" + str(n_iter) + ", save_step=" + str(save_step) + "=====")

    clock_start = time.clock()
    model = Model(K, alpha, beta, n_iter, save_step)
    model.train(doc_pt, output_dir)
    clock_end = time.clock()

    print("procedure time : %f seconds" % (clock_end - clock_start))

    return model


def display_biterm(bs, vocal):
    voc = {}
    for i, line in enumerate(open(vocal).readlines()):
        wid, word = line.strip().split()
        voc[i] = word

    for bi in bs:
        w1 = bi.get_wi()    # 词对中的一个词序号
        w2 = bi.get_wj()    # 词对中的第二个词序号
        print("%s\t%s\t%d" % (voc[w1], voc[w2], bi.get_z()))


if __name__ == "__main__":
    K = 5
    alpha = 0.5
    beta = 0.5
    n_iter = 200
    save_step = 100

    output_dir = "../output/"
    input_dir = "../data/"
    doc_pt = input_dir + "test_2.dat"             # 输入的文档
    model_dir = output_dir + "model/"           # 模型存放的文件夹
    voca_pt = output_dir + "vocabulary.txt"     # 生成的词典

    print("\n\n================ Topic Learning =============")
    my_model = train_BTM()
    # display_biterm(my_model.bs, voca_pt)

    print("\n\n================ Topic Inference =============")
    my_model.infer(doc_pt, model_dir, voca_pt)

    # print("================ Topic Display =============")
    # topicDisplay.run_topicDisplay(['topicDisplay', model_dir, K, voca_pt])
