import gensim
from gensim import corpora, models, similarities
from janome.tokenizer import Tokenizer
from pydub import AudioSegment
import os
import re

stop_words = []

def analyze_topic(documents, num_topics, num_words):
    # 前処理：ストップワードの除去とトークン化
    t = Tokenizer()
    texts = []
    for document in documents:
        text = []
        for token in t.tokenize(document):
            if(token.part_of_speech.split(',')[0] == '名詞'):
                text.append(token.surface)
        texts.append(text)
    print(set(texts[0]))

    # 単語辞書の作成とBOW表現への変換
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # TF-IDFモデルの構築
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    # LDAモデルの構築
    lda_model = models.LdaModel(corpus = corpus_tfidf,
                                id2word = dictionary,
                                num_topics = num_topics,
                                passes = 20,
                                alpha = 'auto',
                                eta = 'auto')

    # トピックの解釈
    topics = {}
    for topic_id, topic in lda_model.print_topics(num_topics = num_topics, num_words = num_words):
        topics[topic_id] = topic
        # print(f"Topic #{topic_id}: {topic}")
    matches = re.findall(r'\"(.*?)\"', topics[0])
    return ', '.join(matches)

def noun(text):
    t = Tokenizer()
    texts = {}
    for token in t.tokenize(text):
        if(token.part_of_speech.split(',')[0] == '名詞'):
            if not token.surface in texts:
                texts[token.surface] = 1
            else:
                texts[token.surface] += 1
    print(texts)


from Transcription import transcription

if __name__ == '__main__':
    hoshino_path = './././Data/test_data/topic/hoshinogen02.wav'
    csv_file = '/'.join(hoshino_path.split('/')[:-1]) + '/info.csv'
    documents = []
    document = transcription.transcribe_faster(hoshino_path, 'small', csv_file)
    # print(document)
    documents.append(document)
    print(analyze_topic(documents, 1, 10))

    # document = 'ケーキは甘くて美味しいよね。ショートケーキなんかは苺がいいアクセントになっててほんとに美味しい。けど、ちょっと高いのが勿体無いところかな。'
    # document2 = 'ケーキは甘くて美味しいよね。ショートケーキなんかは苺がいいアクセントになっててほんとに美味しい。けど、ケーキはちょっと高いのが勿体無いところかな。'
    # documents.append(document2)
    t = analyze_topic(documents, 1, 10)
    print(t)
    # noun(document)