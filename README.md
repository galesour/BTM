# BTM

This is a python implementation according to the Paper: 
[A Biterm model for short texts](https://www.researchgate.net/publication/262244963_A_biterm_topic_model_for_short_texts), 
which introduced a model based on LDA and **Biterm**. 
The ***BTM*** is proved that it works better than LDA on short texts. 

Based on the implementation of [jasperyang](https://github.com/jasperyang/BTMpy.git) 
and [galesour](https://github.com/galesour/BTM.git), 
**Topic Inference** and **Model Evaluation** are implemented by Qinyun Lin. 

Code structure has been rearranged, and many notes are added for understanding the algorithm.


## How to use 
- Run ```python src/evaluateBTM.py``` with default config and test data
- Topic Learning, Topic Inference and Model Evaluation will be performed.
- Edit configs in ```src/evaluateBTM.py```

---
#### 2022-8-8 Update
- Integrated stopwords for data preprocessing
- Add Model Evaluation with coherence score

---
#### 2022-7-31 Update
- Rearrange code structure
- Implement **Topic inference**
- Replace **pvec** class with numpy
- Reimplement mul_sample() in ```src/sample.py```

---
在原来的基础上，加了注释。
原github地址：https://github.com/jasperyang/BTMpy.git
知乎地址 ： https://www.zhihu.com/people/wen-rou-er-yi-43

## 什么是BTM

Xiaohui Yan提出的一个模型， 在LDA的基础上，加入了Biterm的概念。
主要提升了文本主题分类模型在短文本上的性能。

论文: A Biterm model for short texts