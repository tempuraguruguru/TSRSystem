from Transcription import transcription
import srash_transform

import os
import re
import MeCab
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# mecabで用いる辞書のパス
if os.name == 'posix':  # macOSやLinuxの場合
    neologd_path = '-d /opt/homebrew/lib/mecab/dic/mecab-ipadic-neologd'
elif os.name == 'nt':  # Windowsの場合
    neologd_path = '-d "C:/Program Files (x86)/MeCab/dic/ipadic" -u "C:/Program Files (x86)/MeCab/dic/NEologd/NEologd.20200910-u.dic"'

mecab = MeCab.Tagger(neologd_path)

# 指定したAudioPathが保持している名詞一覧とそれぞれ登場回数を取得
def Extract_Nouns(AudioPath, CsvPath):
    print(f"Process: {AudioPath}")
    if os.name == 'posix':  # macOSやLinuxの場合
        neologd_path = '-d /opt/homebrew/lib/mecab/dic/mecab-ipadic-neologd'
    elif os.name == 'nt':  # Windowsの場合
        AudioPath = AudioPath.replace('/', '\\')
    text = transcription.transcribe_faster(AudioPath, 'large', CsvPath)
    parsed = mecab.parse(text)
    nouns = []
    for line in parsed.splitlines():
        if line == 'EOS':
            break
        word, feature = line.split("\t")
        features = feature.split(",")
        if features[0] == "名詞" and (features[1] == "一般" or features[1] == "固有名詞"):
            # word = re.sub(r'(\d)+', '', word)
            nouns.append(word)
    return nouns

def Build_NounMatrix(RootPath):
    nouns = []
    pathes = []
    for current_dir, dirs, files in os.walk(RootPath):
        count_wav = 0
        for file in files:
            if file.endswith('.wav'):
                count_wav += 1
        if count_wav == 0:
            continue
        csv_path = os.path.join(current_dir, 'info.csv')
        srash_transform.srash(csv_path)

        for file in files:
            if file.endswith('.wav'):
                path = os.path.join(current_dir, file)
                pathes.append(path)
                nouns.append(Extract_Nouns(path, csv_path))
    return nouns, pathes

def TermFrequency(nouns, pathes):
    # 各単語の出現回数を計算
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([' '.join(noun_list) for noun_list in nouns])

    tf_df = pd.DataFrame(X.toarray(), columns = vectorizer.get_feature_names_out())
    tf_df.index = pathes
    tf_df = tf_df.div(tf_df.sum(axis = 1), axis = 0)
    return tf_df

def InverseDocumentFrequency(nouns):
    # 各単語の出現回数を計算
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([' '.join(noun_list) for noun_list in nouns])

    N = len(nouns)
    df = np.sum(X.toarray() > 0, axis = 0)
    idf = np.log((N+1) / (df+1)) + 1
    idf_df = pd.DataFrame([idf], columns = vectorizer.get_feature_names_out())
    return idf_df

def TFIDF(tf, idf):
    tfidf = tf.values * idf.values
    tfidf = pd.DataFrame(tfidf, columns = tf.columns, index = tf.index)
    return tfidf

if __name__ == '__main__':
    root = './././Data/test_data/mid_presentation'
    nouns, pathes = Build_NounMatrix(root)
    tf = TermFrequency(nouns, pathes)
    idf = InverseDocumentFrequency(nouns)
    tfidf = TFIDF(tf, idf)

    # tf.to_excel('tf.xlsx')
    # idf.to_excel('idf.xlsx')
    # tfidf.to_excel('tfidf.xlsx')

    mean = idf.iloc[0].mean()
    print(mean)
    result = idf.loc[:, idf.iloc[0] < mean - 2]
    result.to_excel('mean.xlsx')

    print("program is done")