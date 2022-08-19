"""
@Author: 一蓑烟雨任平生
@Date: 2020-02-19 11:31:01
@LastEditTime: 2020-03-03 14:40:39
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /BTMpy/src/sampler.py
"""
# -*- coding: utf-8 -*-
import random


def init():
    random.seed(1)


def uni_sample(k=0.0):
    if k == 0.0:
        k = ("%.4f" % random.random())
    else:
        k = random.randint(0, int(k - 1))
    return k


def mul_sample(vec_p):
    for i in range(1, len(vec_p)):
        vec_p[i] += vec_p[i - 1]
    # pz -= pz[0]

    u = random.random()
    k = 0
    for i, each in enumerate(vec_p):
        if each >= u:
            k = i
            break

    return k


def Bern_sample(p):
    u = random.random()
    return (u < p)


def systematic_sample(p, N, counts=None):
    counts = []
    for i in range(len(p)):
        counts.append(0)

    u = []
    for i in range(N):
        u.append(0)
    u[0] = uni_sample() / N
    for i in range(1, N):
        u[i] = u[0] + float(i) / N

    i = 0
    s1 = 0
    s2 = p[0]
    for n in range(N - 1):
        while (i < N and u[i] < s2):
            i += 1
            counts[n] += 1

        s1 = s2
        s2 += p[n + 1]

    counts[N - 1] = N - i


if __name__ == "__main__":
    systematic_sample([1, 3, 4, 5], 4)
