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
import sys
import indexDocs
import topicDisplay
import os


def usage():
    print("Training Usage: \
    btm est <K> <W> <alpha> <beta> <n_iter> <save_step> <docs_pt> <model_dir>\n\
    \tK  int, number of topics, like 20\n \
    \tW  int, size of vocabulary\n \
    \talpha   double, Pymmetric Dirichlet prior of P(z), like 1.0\n \
    \tbeta    double, Pymmetric Dirichlet prior of P(w|z), like 0.01\n \
    \tn_iter  int, number of iterations of Gibbs sampling\n \
    \tsave_step   int, steps to save the results\n \
    \tdocs_pt     string, path of training docs\n \
    \tmodel_dir   string, output directory")


def BTM(params):
    if len(params) < 4:
        usage()
    else:
        if params[0] == "est":
            K = params[1]
            W = params[2]
            alpha = params[3]
            beta = params[4]
            n_iter = params[5]
            save_step = params[6]
            docs_pt = params[7]
            dir = params[8]
            print("===== Run BTM, K=" + str(K) + ", W=" + str(W) + ", alpha=" + str(alpha) + ", beta=" + str(
                beta) + ", n_iter=" + str(n_iter) + ", save_step=" + str(save_step) + "=====")
            clock_start = time.clock()
            model = Model(K, W, alpha, beta, n_iter, save_step)
            model.run(docs_pt, dir)
            clock_end = time.clock()
            print("procedure time : %f seconds" % (clock_end - clock_start))

            return model

        else:
            usage()


def display_biterm(bs, vocal):
    voc = {}
    for i, line in enumerate(open(vocal).readlines()):
        voc[i] = line.strip().split()[1]

    for bi in bs:
        w1 = bi.get_wi()  # 词对中的一个词序号
        w2 = bi.get_wj()  # 词对中的第二个词序号
        print("%s\t%s\t%d" % (voc[w1], voc[w2], bi.get_z()))


if __name__ == "__main__":
    mode = "est"
    K = 5
    alpha = 0.5
    beta = 0.5
    n_iter = 100  # 十次迭代
    save_step = 100
    output_dir = "../output/"
    input_dir = "../sample-data/"
    doc_pt = input_dir + "test.dat"  # 输入的文档

    model_dir = output_dir + "model/"  # 模型存放的文件夹
    voca_pt = output_dir + "voca.txt"  # 生成的词典
    dwid_pt = output_dir + "doc_wids.txt"  # 每篇文档由对应的序号单词组成

    print("=============== Index Docs =============")

    W = indexDocs.run_indexDocs(['indexDocs', doc_pt, dwid_pt, voca_pt])
    print("Vocabulary size : " + str(W))

    my_params = [mode, K, W, alpha, beta, n_iter, save_step, dwid_pt, model_dir]

    print("=============== Topic Learning =============")
    my_model = BTM(my_params)
    display_biterm(my_model.bs, voca_pt)

    print("================ Topic Display =============")
    topicDisplay.run_topicDisplay(['topicDisplay', model_dir, K, voca_pt])
