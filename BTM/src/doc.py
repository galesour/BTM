'''
@Author: 一蓑烟雨任平生
@Date: 2020-02-18 17:08:33
@LastEditTime: 2020-03-03 11:50:42
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /BTMpy/src/doc.py
'''
# -*- coding: utf-8 -*-
from Biterm import *

class Doc():
    '''
    @description: 处理文本的类
    @param {type} 
    @return: 
    '''
    ws = []

    def __init__(self,s):
        self.ws = []
        self.read_doc(s)

    def read_doc(self,s):
        for w in s.split(' '):
            self.ws.append(int(w))

    def size(self):
        return len(self.ws)

    def get_w(self,i):
        assert(i<len(self.ws))
        return self.ws[i]

    ''' 
      Extract biterm from a document
        'win': window size for biterm extraction
        'bs': the output biterms
    '''
    def gen_biterms(self,bs,win=15):
        if(len(self.ws)<2):
            return
        for i in range(len(self.ws)-1):
            for j in range(i+1,min(i+win,len(self.ws))):
                bs.append(Biterm(self.ws[i],self.ws[j]))



# if __name__ == "__main__":
#     s = '2 3 4 5'
#     d = Doc(s)
#     bs = []
#     print("test")
#     d.gen_biterms(bs)
#     for biterm in bs:
#         print('wi : ' + str(biterm.get_wi()) + ' wj : ' + str(biterm.get_wj()))