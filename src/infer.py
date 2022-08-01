import numpy as np


class Infer:
    def __init__(self, infer_type, k):
        self.type = infer_type
        self.k = k
        self.pz = None      # the probability proportion of K topics
        self.pw_z = None    # the probability proportion of each word in each topic

    def run(self, docs_pt, model_dir):
        self.load_para(model_dir)

    def load_para(self, model_dir):
        # load pz - the probability proportion of K topics
        pt = open(model_dir + "k" + str(self.k) + ".pz")
        if not pt:
            Exception("Model file not found!")

        for line in pt.readlines():
            info = map(lambda x: float(x), line.strip().split())
            self.pz = np.asarray(list(info))
        assert (abs(self.pz.sum() - 1) < 1e-4)

        # load pw_z - the probability proportion of each word in each topic
        pt_2 = open(model_dir + "k" + str(self.k) + ".pw_z")
        if not pt_2:
            Exception("Model file not found!")

        tmp = []
        for line in pt_2.readlines():
            info = map(lambda x: float(x), line.strip().split())
            tmp.append(list(info))
        self.pw_z = np.asarray(tmp)
        print("n(z)=%d, n(w)=%d\n", self.pw_z.shape[0], self.pw_z.shape[1])
        assert(self.pw_z.shape[0] > 0 and abs(self.pw_z[0].sum() - 1) < 1e-4)

