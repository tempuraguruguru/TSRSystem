import MeCab
import re
import itertools
import collections
import pandas as pd
from Transcription import transcription

# mecabで用いる辞書のパス
neologd_path = '-d /opt/homebrew/lib/mecab/dic/mecab-ipadic-neologd'

#stopwordsの指定
with open('././Japanese.txt', 'r') as f:
    stopwords1 = f.read().split('\n')

stopwords2 =[]
stopwords = list(set(stopwords1 + stopwords2))

def mecab_tokenizer(text):
    replaced_text = text.lower()
    replaced_text = re.sub(r'[【】]', ' ', replaced_text)       # 【】の除去
    replaced_text = re.sub(r'[（）()]', ' ', replaced_text)     # （）の除去
    replaced_text = re.sub(r'[［］\[\]]', ' ', replaced_text)   # ［］の除去
    replaced_text = re.sub(r'[@＠]\w+', '', replaced_text)  # メンションの除去
    replaced_text = re.sub(r'\d+\.*\d*', '', replaced_text) #数字を0にする

    mecab = MeCab.Tagger(neologd_path)
    parsed_lines = mecab.parse(replaced_text).split("\n")[:-2]

    # #表層形を取得
    # surfaces = [l.split('\t')[0] for l in parsed_lines]

    #原形を取得
    token_list = [l.split("\t")[1].split(",")[6] for l in parsed_lines]

    #品詞を取得
    pos = [l.split('\t')[1].split(",")[0] for l in parsed_lines]

    # 名詞のみに絞り込み
    target_pos = ["名詞"]
    token_list = [t for t, p in zip(token_list, pos) if p in target_pos]

    # stopwordsの除去
    token_list = [t for t in token_list if t  not in stopwords]

    # ひらがなのみの単語を除く
    kana_re = re.compile("^[ぁ-ゖ]+$")
    token_list = [t for t in token_list if not kana_re.match(t)]

    return token_list


if __name__ == '__main__':
    input_path = '././Data/test_data/topic/hoshinogen01.wav'
    text = transcription.transcribe_faster(input_path, 'large')
    print(text)

    # 共起行列を作成
    sentences = [mecab_tokenizer(sentence) for sentence in text.split("。")]
    sentences2 = []
    for s in sentences:
        if s != []:
            sentences2.append(s)
    sentences = sentences2
    for s in sentences:
        print(s)

    sentences_combs = [list(itertools.combinations(sentence, 2)) for sentence in sentences]
    words_combs = [[tuple(sorted(words)) for words in sentence] for sentence in sentences_combs]
    target_combs = []
    for words_comb in words_combs:
        target_combs.extend(words_comb)

    ct = collections.Counter(target_combs)

    df = pd.DataFrame([{"1番目" : i[0][0], "2番目": i[0][1], "count":i[1]} for i in ct.most_common()])
    print(df)


# Neologd辞書のパスを指定
# neologd_path = '/opt/homebrew/lib/mecab/dic/mecab-ipadic-neologd'
# mecab = MeCab.Tagger(f'-d {neologd_path}')
# text = "私は学生です。"
# parsed_text = mecab.parse(text)
# parsed_texts = parsed_text.split('\n')
# for text in parsed_texts:
#     print(text.split('\t')[1].split(',')[6])