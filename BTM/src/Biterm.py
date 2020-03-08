# -*- coding: utf-8 -*-


class Biterm():
    wi = 0
    wj = 0
    z = 0

    def __init__(self,w1=None,w2=None,s=None):
        if w1 != None and w2 != None:
            self.wi = min(w1,w2)
            self.wj = max(w1,w2)
        elif w1 == None and w2 == None and s != None:
            w = s.split(' ')
            self.wi = w[0]
            self.wj = w[1]
            self.z = w[2]

    def get_wi(self):
        return self.wi

    def get_wj(self):
        return self.wj

    def get_z(self):
        return self.z

    def set_z(self,k):
        self.z = k

    def reset_z(self):
        self.z = -1

    def str(self):
        _str = ""
        _str += str(self.wi) + '\t' + str(self.wj) + '\t\t' + str(self.z)
        return _str