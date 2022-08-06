# -*- coding: utf-8 -*-
import numpy as np
import indexDocs
from doc import Doc
from sampler import *
import os


class Model:
    def __init__(self, k, alpha, beta, n_iter, save_step, has_b=False):
        self.K = k                      # number of topics
        self.vocabulary_size = None     # vocabulary size

        self.alpha = alpha          # hyper-parameters of p(z)
        self.beta = beta            # hyper-parameters of p(w|z)

        self.pw_b = None            # the background word distribution
        self.nw_z = None            # n(w,z), size K*W 各单词被分配到主题z上的次数.
        self.nb_z = np.zeros(k)     # n(b|z), size K*1 各biterm被分配到主题z上的次数,在论文中是用来计算Nz的
        self.bs = list()            # list of all biterms

        self.pz = None              # the probability proportion of K topics
        self.pw_z = None            # the probability proportion of each word in each topic

        # If true, the topic 0 is set to a background topic that equals to the empirical word distribution.
        # It can be used to filter out common words
        self.has_background = False

        self.n_iter = n_iter
        self.save_step = save_step
        self.has_background = has_b

    def train(self, doc_path, output_dir):
        """
        @description: 生成模型运行函数，狄利克雷-多项 共轭分布，Gibbs采样
        @param {type}
        @return:
        """
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
            os.mkdir(output_dir + "/model")
        vocabulary_path = output_dir + "/vocabulary.txt"
        index_docs = self.load_docs(doc_path, vocabulary_path)
        self.model_init(index_docs)

        print("Begin iteration")
        model_dir = output_dir + "/model/k" + str(self.K) + "."
        for i in range(1, self.n_iter + 1):
            print("\riter " + str(i) + "/" + str(self.n_iter), end='...')
            for b in range(len(self.bs)):
                # 根据每个biterm更新文章中的参数
                # 计算核心代码，self.bs中保存的是词对的biterm
                self.update_biterm(self.bs[b])
            if i % self.save_step == 0:
                self.save_model(model_dir)

        self.save_model(model_dir)

    def infer(self, doc_path, model_dir, vocabulary_path):
        index_docs = self.load_docs(doc_path, vocabulary_path, if_load_voc=True)
        if self.pz is None and self.pw_z is None:
            self.load_model(model_dir)

        indexToWord = sorted(index_docs.wordToIndex.keys(), key=lambda x: index_docs.wordToIndex[x])
        for each in index_docs.docIndex:
            pz_d = np.zeros(self.K)  # the probability proportion of the Doc in each Topic

            d = Doc(each)
            biterms = []
            d.gen_biterms(biterms)
            for bi in biterms:
                # calculate pz_d via probability proportion of each biterm
                pz_b = self.compute_pz_b(bi)

                for i in range(self.K):
                    # ？？？原作者的实现仅对p(z|b)进行求和，与论文中Sum(p(z|d) * p(z|b))不一致
                    # => 由于p(z|b)近于均匀分布，因此这里不必再进行计算
                    pz_d[i] += pz_b[i]

            pz_d = self.normalize_ndarray(pz_d)
            sentence = list(map(lambda x: indexToWord[x], each))
            print("Topic: %d\t %s" % (int(np.argmax(pz_d)), sentence))

    def model_init(self, index_docs):
        """
        @description: 初始化模型的代码。
        @param :None
        @return: 初始化self.nv_z 和self.nwz，
        """
        self.pw_b = np.zeros(self.vocabulary_size)
        self.nw_z = np.zeros((self.K, self.vocabulary_size))

        for each in index_docs.docIndex:
            d = Doc(each)
            biterms = []
            d.gen_biterms(biterms)
            # statistic the empirical word distribution
            for i in range(d.size()):
                w = d.get_w(i)
                self.pw_b[w] += 1  # 统计词频
            for b in biterms:
                self.bs.append(b)  # self.bs中添加的是一个biterm类。类的内容是这段文本中所有可能的词的组合.

        # rf = open(doc_path)
        # if not rf:
        #     print("file not found: " + doc_path)
        #
        # for line in rf.readlines():
        #     d = Doc(line)
        #     biterms = []  # 一句话里的单词能组成的词对。
        #     d.gen_biterms(biterms)
        #     # statistic the empirical word distribution
        #     for i in range(d.size()):
        #         w = d.get_w(i)
        #         self.pw_b[w] += 1  # 这行代码是在统计词频
        #     for b in biterms:
        #         self.bs.append(b)  # self.bs中添加的是一个biterm类。类的内容是这段文本中所有可能的词的组合.

        # 做归一化处理,现在 pw_b中保存的是 词：词频率。
        self.pw_b = self.normalize_ndarray(self.pw_b)

        for biterm in self.bs:
            # k表示的是从0-K之间的随机数。用来将biterm随机分配给各个topic
            k = uni_sample(self.K)
            self.assign_biterm_topic(biterm, k)  # 入参是一个词对(biterm)和他对应的主题

    def load_model(self, model_dir):
        # load pz - the probability proportion of K topics
        pt = open(model_dir + "k" + str(self.K) + ".pz")
        if not pt:
            Exception("Model file not found!")

        for line in pt.readlines():
            info = map(lambda x: float(x), line.strip().split())
            self.pz = np.asarray(list(info))
        assert (abs(self.pz.sum() - 1) < 1e-4)

        # load pw_z - the probability proportion of each word in each topic
        pt_2 = open(model_dir + "k" + str(self.K) + ".pw_z")
        if not pt_2:
            Exception("Model file not found!")

        tmp = []
        for line in pt_2.readlines():
            info = map(lambda x: float(x), line.strip().split())
            tmp.append(list(info))
        self.pw_z = np.asarray(tmp)
        print("n(z)=%d, n(w)=%d\n" % (self.pw_z.shape[0], self.pw_z.shape[1]))
        assert (self.pw_z.shape[0] > 0 and abs(self.pw_z[0].sum() - 1) < 1e-4)

    def load_docs(self, doc_path, vocabulary_path, if_load_voc=False):
        """
        @description: 读取文档并做indexing，生成self.pw_b 和 self.bs
        """

        print("load docs: " + doc_path)

        index_docs = indexDocs.IndexDocs(if_load_voc=if_load_voc)
        self.vocabulary_size = index_docs.run_indexDocs(doc_path, vocabulary_path)

        return index_docs

    def normalize_ndarray(self, array, smoother=0):
        t_sum = array.sum()

        array = (array + smoother) / (t_sum + self.K * smoother)
        return array

    def update_biterm(self, bi):
        self.reset_biterm_topic(bi)

        # comput p(z|b)
        pz_b = self.compute_pz_b(bi)

        # sample topic for biterm b
        k = mul_sample(pz_b)
        self.assign_biterm_topic(bi, k)  # 更新论文中的Nz,N_wiz,N_wjz.

    def reset_biterm_topic(self, bi):
        k = bi.get_z()
        w1 = bi.get_wi()
        w2 = bi.get_wj()

        self.nb_z[k] -= 1
        self.nw_z[k][w1] -= 1
        self.nw_z[k][w2] -= 1
        assert (self.nb_z[k] > -10e-7 and self.nw_z[k][w1] > -10e-7 and self.nw_z[k][w2] > -10e-7)
        bi.reset_z()

    def assign_biterm_topic(self, bi, k):
        # bi是每一个词对，K是主题的个数。
        bi.set_z(k)
        w1 = bi.get_wi()  # 词对中的一个词
        w2 = bi.get_wj()  # 词对中的第二个词
        self.nb_z[k] += 1  # self.nb_z: 表示的是在那么多的词对中，每个主题出现的次数。
        self.nw_z[k][w1] += 1  # self.nwz[1][1] 表示的是在主题1中，1号单词出现的次数。
        self.nw_z[k][w2] += 1  # self.nwz[2][3] 表示的是在出题2中，2号单词出现的次数。

    def compute_pz_b(self, bi):
        pz_b = np.zeros(self.K)
        w1 = bi.get_wi()  # 取到词对中的第一个词编号。
        w2 = bi.get_wj()  # 取到词对中的第二个词编号。

        for k in range(self.K):
            if self.pz is None and self.pw_z is None:
                if self.has_background and k == 0:
                    pw1k = self.pw_b[w1]
                    pw2k = self.pw_b[w2]
                else:
                    pw1k = (self.nw_z[k][w1] + self.beta) / (2 * self.nb_z[k] + self.vocabulary_size * self.beta)
                    pw2k = (self.nw_z[k][w2] + self.beta) / (2 * self.nb_z[k] + 1 + self.vocabulary_size * self.beta)

                # len(self.bs)表示的是在文档中以后多少的词对
                pk = (self.nb_z[k] + self.alpha) / (len(self.bs) + self.K * self.alpha)
                pz_b[k] = pk * pw1k * pw2k
            else:
                pz_b[k] = self.pz[k] * self.pw_z[k][w1] * self.pw_z[k][w2]

        pz_b = self.normalize_ndarray(pz_b)
        return pz_b

    def save_model(self, output_dir):
        pt = output_dir + "pz"
        print("\nwrite p(z): " + pt)
        self.save_pz(pt)

        pt2 = output_dir + "pw_z"
        print("write p(w|z): " + pt2)
        self.save_pw_z(pt2)

    # p(z) is determinated by the overall proportions of biterms in it
    # 函数计算的是每个主题的分布。
    def save_pz(self, pt):
        self.pz = np.asarray(self.nb_z)
        self.pz = self.normalize_ndarray(self.pz, self.alpha)

        wf = open(pt, 'w')
        wf.write(str(self.pz.tolist()).strip("[]").replace(",", ""))

    # 函数计算的是每个主题下各个单词的分布
    def save_pw_z(self, pt):
        self.pw_z = np.ones((self.K, self.vocabulary_size))  # 生成5行2700列的矩阵。用来保存每个主题中，各个单词出现的概率。
        wf = open(pt, 'w')
        for k in range(self.K):
            for w in range(self.vocabulary_size):
                # 计算每个词在这个主题中出现的概率。
                self.pw_z[k][w] = (self.nw_z[k][w] + self.beta) / (self.nb_z[k] * 2 + self.vocabulary_size * self.beta)
                wf.write(str(self.pw_z[k][w]) + ' ')
            wf.write("\n")
