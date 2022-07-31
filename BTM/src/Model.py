# -*- coding: utf-8 -*-
import numpy as np
from doc import Doc
from sampler import *


class Model:
    """
    @deprecated: 函数的功能是生成模型
    @param
    @return:
    """
    W = 0  # vocabulary size
    K = 0  # number of topics
    n_iter = 0  # maximum number of iteration of Gibbs Sampling，吉布斯采样的最大迭代次数。
    save_step = 0
    alpha = 0  # hyper-parameters of p(z)
    beta = 0  # hyper-parameters of p(w|z)
    nb_z = np.zeros(1)  # n(b|z), size K*1 所有的词对biterm分配到主题z上的次数,在论文中是用来计算Nz的
    nwz = np.zeros((1, 1))  # n(w,z), size K*W，表示的是单词分配到主题z上的次数.
    pw_b = np.zeros(1)  # the background word distribution。这三个参数都是为了计算每个主题中单词的分布而存在的。
    bs = []

    '''
        If true, the topic 0 is set to a background topic that 
        equals to the empirical word distribution. It can filter
        out common words
    '''
    has_background = False

    def __init__(self, k, W, a, b, n_iter, save_step, has_b=False):
        self.K = k
        self.W = W
        self.alpha = a
        self.beta = b
        self.n_iter = n_iter
        self.save_step = save_step
        self.has_background = has_b
        self.pw_b.resize(W)
        self.nwz.resize((k, W))
        self.nb_z.resize(k)

    def run(self, doc_pt, res_dir):
        """
        @description: 生成模型运行函数，狄利克雷-多项 共轭分布，Gibbs采样
        @param {type}
        @return:
        """

        self.load_docs(doc_pt)
        self.model_init()

        print("Begin iteration")
        out_dir = res_dir + "k" + str(self.K) + "."
        for i in range(1, self.n_iter + 1):
            print("\riter " + str(i) + "/" + str(self.n_iter), end='\r')
            for b in range(len(self.bs)):
                # 根据每个biterm更新文章中的参数
                self.update_biterm(self.bs[b])  # 计算核心代码，self.bs中保存的是词对的biterm类，代码是对每一个词对进行更新的。
            if i % self.save_step == 0:
                self.save_res(out_dir)

        self.save_res(out_dir)

    def model_init(self):
        """
        @description: 初始化模型的代码。
        @param :None
        @return: 生成self.nv_z 和self.nwz，
        @self.nb_z: 表示的是在那么多的词对中，每个主题出现的次数。self.nv_z[1]:表示的第一个主题出现的次数。
        @self.nwz[2][3] 表示的是在主题2中，2号单词出现的次数。
        """
        for biterm in self.bs:
            # print('==========')
            # print(biterm.get_wi(), biterm.get_wj())
            k = uni_sample(self.K)  # k表示的是从0-K之间的随机数。用来将biterm随机分配给各个topic

            # print(k)
            self.assign_biterm_topic(biterm, k)  # 入参是一个词对和他对应的主题？
            # print('============')
            print('\n')

    def load_docs(self, docs_pt):
        """
        @description: 生成self.pw_b 和 self.bs
            self.pw_b表示的是每个单词对应的词频，若一共有7个单词，那么pw_b的size就是7，
            self.bs表示的是所有的biterm
        @param docs_pt:
        @return:
        """
        print("load docs: " + docs_pt)
        rf = open(docs_pt)
        if not rf:
            print("file not found: " + docs_pt)

        for line in rf.readlines():
            d = Doc(line)
            biterms = []  # 一句话里的单词能组成的词对。
            d.gen_biterms(biterms)
            # statistic the empirical word distribution
            for i in range(d.size()):
                w = d.get_w(i)
                self.pw_b[w] += 1  # 这行代码是在统计词频
            for b in biterms:
                self.bs.append(b)  # self.bs中添加的是一个biterm类。类的内容是这段文本中所有可能的词的组合.

        self.pw_b = self.normalize_ndarray(self.pw_b)
        # self.pw_b.normalize()  # 做归一化处理,现在 pw_b中保存的是 词：词频率。

    def normalize_ndarray(self, array, smoother=0):
        t_sum = array.sum()

        array = (array + smoother) / (t_sum + self.K * smoother)
        return array

    def update_biterm(self, bi):
        # print('-----------')
        # print(bi.get_wi(),bi.get_wj())
        self.reset_biterm_topic(bi)

        # comput p(z|b),相当于论文中计算Zb
        pz = np.zeros(self.K)
        self.comput_pz_b(bi, pz)  # 计算出来的结果，直接作用在pz上。
        # print(pz.size()) #pz是一个三个具体的数，如果主题的个数是5的话，那么pz就是5个具体的数。
        # print(pz.to_vector())  # pz.to_vector()表示将三个数转成向量。

        # sample topic for biterm b
        k = mul_sample(pz)  # k表示根据pz算出三个数中最大的主题。。这步是在干嘛呀。。
        # print(k)
        # print('-----------')
        # print('\n')
        self.assign_biterm_topic(bi, k)  # 更新论文中的Nz,N_wiz,N_wjz.

    def reset_biterm_topic(self, bi):
        k = bi.get_z()
        w1 = bi.get_wi()
        w2 = bi.get_wj()

        self.nb_z[k] -= 1
        self.nwz[k][w1] -= 1
        self.nwz[k][w2] -= 1
        assert (self.nb_z[k] > -10e-7 and self.nwz[k][w1] > -10e-7 and self.nwz[k][w2] > -10e-7)
        bi.reset_z()

    def assign_biterm_topic(self, bi, k):
        # bi是每一个词对，K是主题的个数。
        bi.set_z(k)
        w1 = bi.get_wi()  # 词对中的一个词
        w2 = bi.get_wj()  # 词对中的第二个词
        self.nb_z[k] += 1  # self.nb_z: 表示的是在那么多的词对中，每个主题出现的次数。
        self.nwz[k][w1] += 1  # self.nwz[1][1] 表示的是在主题1中，1号单词出现的次数。
        self.nwz[k][w2] += 1  # self.nwz[2][3] 表示的是在出题2中，2号单词出现的次数。

    def comput_pz_b(self, bi, pz):
        # 计算
        # pz.resize(self.K)
        w1 = bi.get_wi()  # 取到词对中的第一个词编号。
        w2 = bi.get_wj()  # 取到词对中的第二个词编号。

        for k in range(self.K):
            if self.has_background and k == 0:
                pw1k = self.pw_b[w1]
                pw2k = self.pw_b[w2]
            else:
                pw1k = (self.nwz[k][w1] + self.beta) / (2 * self.nb_z[k] + self.W * self.beta)
                pw2k = (self.nwz[k][w2] + self.beta) / (2 * self.nb_z[k] + 1 + self.W * self.beta)

            pk = (self.nb_z[k] + self.alpha) / (len(self.bs) + self.K * self.alpha)  # len(self.bs)表示的是在文档中以后多少的词对
            pz[k] = pk * pw1k * pw2k

    def save_res(self, res_dir):
        pt = res_dir + "pz"
        print("\nwrite p(z): " + pt)
        self.save_pz(pt)

        pt2 = res_dir + "pw_z"
        print("write p(w|z): " + pt2)
        self.save_pw_z(pt2)

    # p(z) is determinated by the overall proportions of biterms in it
    # 函数计算的是每个主题的分布。
    def save_pz(self, pt):
        # pz = Pvec(pvec_v=self.nb_z)
        pz = np.asarray(self.nb_z)
        pz = self.normalize_ndarray(pz, self.alpha)
        # pz.normalize(self.alpha)

        wf = open(pt, 'w')
        wf.write(str(pz.tolist()).strip("[]").replace(",", ""))
        # pz.write(pt)

    # 函数计算的是每个主题下各个单词的分布
    def save_pw_z(self, pt):
        pw_z = np.ones((self.K, self.W))  # 生成5行2700列的矩阵。用来保存每个主题中，各个单词出现的概率。
        wf = open(pt, 'w')
        for k in range(self.K):
            for w in range(self.W):
                pw_z[k][w] = (self.nwz[k][w] + self.beta) / (self.nb_z[k] * 2 + self.W * self.beta)  # 计算每个词在这个主题中出现的概率。
                wf.write(str(pw_z[k][w]) + ' ')
            wf.write("\n")
