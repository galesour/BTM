# !/usr/bin/env python
# coding=utf-8
from nltk.corpus import stopwords


class IndexDocs:
    # translate word into id in documents

    def __init__(self, if_load_voc=False):
        self.texts = []
        self.wordToIndex = {}
        self.docIndex = []
        self.if_load_voc = if_load_voc

    def indexFile(self, doc_path):
        print('indexing doc file: ' + str(doc_path))
        sw_nltk = stopwords.words('english')
        # wf = open(res_pt, 'w', encoding="utf-8")

        for line in open(doc_path, encoding="utf-8"):
            ws = line.strip().split()
            temp_ws = []
            wids = []
            for w in ws:
                # preprocessing
                w = w.lower()
                if w in sw_nltk:
                    continue
                w = w.replace("!", "")
                w = w.replace(",", "")
                w = w.replace(".", "")
                w = w.replace("?", "")
                w = w.replace("@", "")
                w = w.replace("#", "")
                w = w.replace(":", "")
                w = w.replace("...", "")
                if w == "":
                    continue

                temp_ws.append(w)

                if self.if_load_voc:
                    if w in self.wordToIndex:
                        wids.append(self.wordToIndex[w])
                    else:
                        wids.append(self.wordToIndex["unknown"])
                else:
                    if w not in self.wordToIndex:
                        self.wordToIndex[w] = len(self.wordToIndex)
                    wids.append(self.wordToIndex[w])

            self.texts.append(temp_ws)
            self.docIndex.append(wids)
            # print>>wf,' '.join(map(str, wids))
            # print(' '.join(map(str, wids)), file=wf)

        if not self.if_load_voc:
            self.wordToIndex["unknown"] = len(self.wordToIndex)
        # print('write file: ' + str(res_pt))

    def load_voc(self, vocal_path):
        for i, line in enumerate(open(vocal_path).readlines()):
            word = line.strip().split()[1]
            self.wordToIndex[word] = i

    def write_voc(self, res_pt):
        print('write:' + str(res_pt))
        wf = open(res_pt, 'w')

        # for wid in range(len(self.wordToIndex)):
        #     word = self.wordToIndex[wid]
        #     print('%d\t%s' % (wid, word), file=wf)

        for w, wid in sorted(self.wordToIndex.items(), key=lambda d: d[1]):
            print('%d\t%s' % (wid, w), file=wf)
        wf.close()

    def run_indexDocs(self, doc_path, vocal_path):
        # if len(argv) < 4:
        #     print('Usage: python %s <doc_pt> <dwid_pt> <voca_pt>' % argv[0])
        #     print('\tdoc_pt    input docs to be indexed, each line is a doc with the format "word word ..."')
        #     print('\tdwid_pt   output docs after indexing, each line is a doc with the format "wordId wordId..."')
        #     print('\tvoca_pt   output vocabulary file, each line is a word with the format "wordId    word"')
        #     exit(1)

        if self.if_load_voc:
            self.load_voc(vocal_path)

        self.indexFile(doc_path)
        print('n(words)=' + str(len(self.wordToIndex)))

        if not self.if_load_voc:
            self.write_voc(vocal_path)

        return len(self.wordToIndex)
