# -*- coding: utf-8 -*-
# 这个类是相当于作者自己实现的一个numpy，作者自己也说这个类很鸡肋，如果大家想自己实现的话，只要用numpy就好了。

import random
import math

class Pvec():
    p = [] # type:vector

    def __init__(self,n=None,v=None,vector_v=None,pvec_v=None,line=None):
        if n != None and v == None:
            for i in range(n):
                self.p.append(i)
        elif v != None:
            for i in range(n):
                self.p.append(v)
        if vector_v != None:
            for i in range(len(self.p)):
                self.p[i] = vector_v[i]
        if pvec_v != None:
            if len(self.p) > len(pvec_v):
                self.p = self.p[:len(pvec_v)]
            else:
                for i in range(len(pvec_v)):
                    self.p.append(0)
            for i in range(len(pvec_v)):
                for i in range(len(pvec_v)):
                    self.p[i] = pvec_v[i]
        if line != None:
            self.loadString(line)

    def __setitem__(self, key, value):
        self.p[key] = value

    def size(self):
        '''
        @description: 返回长度，为了更好的调试，这里打印了里面的每一个元素。
        @param 不接收参数
        @return: 返回长度
        '''
        for i in range(len(self.p)):
            print(self.p[i])
        return len(self.p)

    def __len__(self):
        return len(self.p)

    def resize(self,n=None,v=None):
        if n != None and v == None:
            if len(self.p) > n:
                self.p = self.p[:n]
            else:
                for i in range(n):
                    self.p.append(0)
        elif n != None and v != None:
            if len(self.p) > n:
                self.p = self.p[:n]
            else:
                for i in range(n-len(self.p)):
                    self.p.append(v)

    def assign(self,n,v):
        self.p = []
        for i in range(n):
            self.p.append(v)

    def rand_init(self):
        for i in range(len(self.p)):
            self.p[i] = random.randrange(1,100)
        self.normalize()

    def fill(self,v):
        for i in range(len(self.p)):
            self.p[i] = v

    def uniform_init(self):
        for i in range(len(self.p)):
            self.p[i] = float(1)/len(self.p)

    def bias_init(self,v):
        assert(v<1)
        self.p[0] = v
        for i in range(len(self.p)):
            self.p[i] = float((1-v)/(len(self.p)-1))

    def push_back(self,v):
        self.p.append(v)

    def extend(self,vec):
        self.p += vec.p

    def sum(self):
        #函数的作用是统计出所有词频的和
        s = 0
        for i in range(len(self.p)):
            s += self.p[i]
        # for i in self.p[0:20]:
        #     print(i)
        # print("sum函数的输出：",s)
        return s

    def normalize(self,smoother=0.0):
        #函数的作用是将词编号：词频 => 词编号: 词频率,就是做了一个归一化处理.
        #👆在胡说八道，才不是简单的做了归一化那么简单呢，函数的作用是用来计算参数的。
        #函数的作用是吉布斯采样来估计参数。
        s = self.sum()
        assert(s>=0)

        K = len(self.p)
        for i in range(K):
            self.p[i] = (self.p[i] + smoother)/(s + K*smoother)

    def exp_normalize(self):
        tmp = self.p[:]
        for i in range(len(self.p)):
            s = 0.0
            for j in range(len(self.p)):
                s += math.exp(tmp[j] - tmp[i])
            assert(s>=1)
            self.p[i] = 1/s

    def smooth(self,smoother):
        for i in range(len(self.p)):
            if self.p[i] < smoother:
                self.p[i] = smoother

    def loadFileStream(self,rf):
        self.p = []
        for v in rf.split(" "):
            self.p.append(int(v))

    def norm(self):
        s = 0
        for i in range(len(self.p)):
            s += self.p[i]*self.p[i]
        return math.sqrt(s)

    def loadString(self,line):
        self.p = []
        for v in line.split(" "):
            self.p.append(int(v))

    # Method behind : Operator Overloaded

    def __add__(self, other):
        tp = self.p[:]
        if type(other) == float or type(other) == int:
            for i in range(len(self.p)):
                tp[i] = self.p[i] + other
        else:
            for i in range(len(self.p)):
                tp[i] = self.p[i] + other[i]
        return Pvec(pvec_v=tp)

    def __iadd__(self, other):
        if type(other) == float or type(other) == int:
            for i in range(len(self.p)):
                self.p[i] += other
        else:
            for i in range(len(self.p)):
                self.p[i] += other[i]
        return self

    def __sub__(self, other):
        tp = self.p[:]
        if type(other) == float or type(other) == int:
            for i in range(len(self.p)):
                tp[i] = self.p[i] - other
        else:
            for i in range(len(self.p)):
                tp[i] = self.p[i] - other[i]
        return Pvec(pvec_v=tp)

    def __isub__(self, other):
        if type(other) == float or type(other) == int:
            for i in range(len(self.p)):
                self.p[i] -= other
        else:
            for i in range(len(self.p)):
                self.p[i] -= other[i]
        return self

    def __mul__(self, other):
        tp = self.p[:]
        for i in range(len(self.p)):
            tp[i] = self.p[i] * other
        return Pvec(pvec_v=tp)

    def __imul__(self, other):
        for i in range(len(self.p)):
            self.p[i] *= other
        return self

    def __div__(self, other):
        tp = self.p[:]
        for i in range(len(self.p)):
            tp[i] = self.p[i] / other
        return Pvec(pvec_v=tp)

    def __idiv__(self, other):
        for i in range(len(self.p)):
            self.p[i] /= other
        return self

    # end of Operator Overloaded

    def add1_log(self):
        for i in range(len(self.p)):
            self.p[i] = math.log(1+self.p[i])

    def max(self):
        max_v = -10000000
        for i in range(len(self.p)):
            if self.p[i] > max_v:
                max_v = self.p[i]
        return max_v

    def max_idx(self):
        max_v = -10000000
        idx = 0
        for i in range(len(self.p)):
            if self.p[i] > max_v:
                max_v = self.p[i]
                idx = i
        return idx

    def erase(self,start,end):
        assert(end>=start and end<=len(self.p))
        self.p = self.p[:start] + self.p[end:]

    def clear(self):
        self.p = []

    def to_vector(self):
        return self.p

    def to_double(self):
        tmp = self.p[:]
        for i in range(len(self.p)):
            tmp[i] = float(tmp[i])
        return Pvec(pvec_v=tmp)

    def str(self,delim = ' '):
        _str = ""
        for i in range(len(self.p)):
            _str += (str(self.p[i]) + delim)
        return _str

    def sparse_str(self,v):
        _str = ""
        for i in range(len(self.p)):
            if self.p[i] > v:
                _str += (str(i)+':'+str(self.p[i])+' ')
        return _str

    def write(self,pt,delim=' '):
        output = open(pt,'w')
        for pp in self.p:
            output.write(str(pp) + delim)

    def __getitem__(self, item):
        return self.p[item]

    def test(self):
        print(self.p)

if __name__ == "__main__":
    a = Pvec(line="10 4 5 6 1 3")
    a.test()