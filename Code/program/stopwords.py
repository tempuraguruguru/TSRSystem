from Transcription import transcription
import srash_transform

import os
import MeCab
import pandas as pd

# mecabで用いる辞書のパス
neologd_path = '-d /opt/homebrew/lib/mecab/dic/mecab-ipadic-neologd'
mecab = MeCab.Tagger(neologd_path)

if __name__ == '__main__':
    dict = {'noun': [], 'count': []}
    root = './././Data/test_data/mid_presentation'
    for current_dir, dirs, files in os.walk(root):
        files2 = []
        for file in files:
            if file.startswith('.'):
                continue
            files2.append(file)
        files = files2
        if len(files) == 0:
            print(f'{current_dir} don\'t have files')
            continue
        csv_path = os.path.join(current_dir, 'info.csv')
        srash_transform.srash(csv_path)
        print(f'use the {csv_path}')

        for file in files:
            if file.endswith('.wav'):
                path = os.path.join(current_dir, file)
                print(f'Processing the {path}')
                text = transcription.transcribe_faster(path, 'large', csv_path)

                parsed = mecab.parse(text)
                nouns = []
                for line in parsed.splitlines():
                    if line == "EOS":
                        break
                    word, feature = line.split("\t")
                    features = feature.split(",")
                    # 名詞または複合名詞を抽出
                    if features[0] == "名詞" and (features[1] == "一般" or features[1] == "固有名詞"):
                        nouns.append(word)

                for noun in nouns:
                    if noun in dict['noun']:
                        index = dict['noun'].index(noun)
                        dict['count'][index] += 1
                    else:
                        dict['noun'].append(noun)
                        dict['count'].append(1)

    df = pd.DataFrame(dict)
    df = df.sort_values('count', ascending = False)
    df.to_excel('nouns.xlsx')