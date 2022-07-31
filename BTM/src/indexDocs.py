"""
@Author: 一蓑烟雨任平生
@Date: 2020-02-18 17:08:33
@LastEditTime: 2020-02-19 10:59:25
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /BTMpy/src/indexDocs.py
"""

# !/usr/bin/env python
# coding=utf-8
# translate word into id in documents

w2id = {}


def indexFile(pt, res_pt):
    print('index file: ' + str(pt))
    wf = open(res_pt, 'w', encoding="utf-8")
    for line in open(pt, encoding="utf-8"):
        ws = line.strip().split()
        for w in ws:
            if w not in w2id:
                w2id[w] = len(w2id)

        wids = [w2id[w] for w in ws]
        # print>>wf,' '.join(map(str, wids))
        print(' '.join(map(str, wids)), file=wf)

    print('write file: ' + str(res_pt))


def write_w2id(res_pt):
    print('write:' + str(res_pt))
    wf = open(res_pt, 'w')
    for w, wid in sorted(w2id.items(), key=lambda d: d[1]):
        print('%d\t%s' % (wid, w), file=wf)


def run_indexDocs(argv):
    if len(argv) < 4:
        print('Usage: python %s <doc_pt> <dwid_pt> <voca_pt>' % argv[0])
        print('\tdoc_pt    input docs to be indexed, each line is a doc with the format "word word ..."')
        print('\tdwid_pt   output docs after indexing, each line is a doc with the format "wordId wordId..."')
        print('\tvoca_pt   output vocabulary file, each line is a word with the format "wordId    word"')
        exit(1)

    doc_pt = argv[1]
    dwid_pt = argv[2]
    voca_pt = argv[3]
    indexFile(doc_pt, dwid_pt)
    print('n(w)=' + str(len(w2id)))
    write_w2id(voca_pt)
    return len(w2id)
