# -*- coding: utf-8 -*-
import time
from Model import *
from coherenceScore import cal_coherence


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
    n_iter = 100
    save_step = 1000

    output_dir = "../output/"
    input_dir = "../data/"
    doc_pt = input_dir + "test.dat"                     # input documents
    model_dir = output_dir + "model/"                   # dictionary to save model
    vocabulary_path = output_dir + "vocabulary.txt"     # generated vocabulary

    print("\n\n================ Topic Learning =============")
    my_model = train_BTM()
    # display_biterm(my_model.bs, vocabulary_path)

    print("\n\n================ Topic Inference =============")
    topic_dict = my_model.infer(doc_pt, model_dir, vocabulary_path)

    for T in [5, 10, 20]:
        print("Coherence Score(T=%d) = %f" % (T, cal_coherence(topic_dict, my_model.topic_words, top_n=T)))
