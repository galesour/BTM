from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel


def cal_coherence(topic_dict, topics, top_n=5):
    data_set = [text.split() for texts_per_topic in topic_dict.values() for text in texts_per_topic]
    dictionary = corpora.Dictionary(data_set)
    cm_u_mass = CoherenceModel(topics=topics, texts=data_set, dictionary=dictionary, coherence='u_mass', topn=top_n)

    return cm_u_mass.get_coherence()
