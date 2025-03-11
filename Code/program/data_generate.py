from AudioSegmentation import spectral, sr_split
import os
from pydub import AudioSegment
import time
import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
    # print("CUDA is available. Using GPU.", torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')
    # print("CUDA is not available. Using CPU")

# スペクトル解析のハイパーパラメータ
spectral.THRESHOLD = 0.0025
spectral.DISTANCE = 30

# 話者認識のハイパーパラメータ
sr_split.MAIN_SPEAKER_TIME = 60
sr_split.MAIN_SPEAKER_SILENCE = 30.0

# 何秒以下のファイルを楽曲・CM部分として認識して削除するかのハイパーパラメータ
CM_MUSIC_TIME = 180


from Transcription import transcription
import topic_mecab
import playback_time

import pandas as pd
import numpy as np
from transformers import BertJapaneseTokenizer, BertModel
from sentence_transformers import SentenceTransformer


# モデル
tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
bert_model = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# モデルのハイパーパラメータ
MODEL_GRADE = "large-v3"
SIMILARITY_THRESHOLD = 0.80

# トピックセグメント分割におけるハイパーパラメータ
topic_mecab.TOPIC_PROBABILITY_THRESHOLD = 0.5
topic_mecab.TOPIC_CONTINUATION_THRESHOLD = 4


def make_list(str):
    list = []
    str = str[2:-2].split(',')
    for s in str:
        list.append(s.replace(' ', ''))
    return list

def make_dic(str):
    dic = {}
    str = str[2:-2].split(',')
    for s in str:
        s = s.split(':')
        dic[s[0].replace(' ', '')] = np.float64((s[1])[1:-1].replace(' ', ''))
    return dic

def dic_to_list(dic):
    res = []
    for key, value in dic.items():
        res.append(value)
    return res

def str_dic(str):
    dic = {}
    str = str[1:-1]
    str = str.replace(" ", "")
    list = str.split("],")
    for s in list:
        s += ']'
        s = s.replace("[[", "[")
        s = s.replace("]]", "]")
        ss = s.split(":")
        dic[ss[0].replace("'", "")] = eval(ss[1])
    return dic


# セグメントを作成
def make_segments(root):
    for current_dirs, dirs, files in os.walk(root):
        if 'info.csv' in files:
            print(f'segments exist')
            return None

    for current_dirs, dirs, files in os.walk(root):
        print(current_dirs)
        print(dirs)
        # カレントディレクトリ内の全てのwavファイルをスペクトル解析で分割
        for file in files:
            if file.endswith('.wav'):
                print(f'spectal :{file}')
                spectral.split(audio_path = current_dirs + '/' + file, output_path = current_dirs)
                print('remove raw file')
                os.remove(current_dirs + '/' + file)

    for current_dirs, dirs, files in os.walk(root):
        print(current_dirs)
        print(dirs)
        print(files)
        # カレントディレクトリ内の全てのwavファイルを話者認識で分割
        for file in files:
            if file.endswith('.wav'):
                print(f'speech recogniton: {file}')
                sr_split.SpeechRecognition(audio_path = current_dirs + '/' + file)
                print(f'remove {file}')
                os.remove(current_dirs + '/' + file)

    # カレントディレクトリ内の3分以内の音声をCM部分または楽曲部分として認識して削除
    for current_dirs, dirs, files in os.walk(root):
        print(current_dirs)
        print(dirs)
        print(files)
        for file in files:
            if file.endswith('.wav'):
                audio = AudioSegment.from_file(current_dirs + '/' + file, format = 'wav')
                if audio.duration_seconds <= CM_MUSIC_TIME:
                    os.remove(current_dirs + '/' + file)


# セグメント情報を作成
def make_segment_info(directory_path, num_topics):
    for root, dirs, files in os.walk(directory_path):
        if(len(files)) == 0:
            # print(f'{root} don\'t have files')
            continue
        if(len(dirs) != 0):
            continue

        # info.csvの設定
        csv_path = os.path.join(root, 'info.csv')
        # print(f'use the {csv_path}')

        # info.csvが存在するならば読み込み、存在しないならば新規作成
        if os.path.exists(csv_path):
            # print('exist')
            if os.path.getsize(csv_path) > 0:
                dict = {'file_path': [], 'time': []}
            else:
                print(f"{csv_path} は空のファイルです。")
        else:
            # print('no exist')
            data = {'file_path': [], 'transcription': [], 'transcription_timestamp': []}
            df = pd.DataFrame(data)
            df.to_csv(csv_path, index = False, encoding = 'utf-8')
            dict = {'file_path': [], 'time': []}

        # 分割済みの各ファイルに対して、文字起こしテキストと時間を取得
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)

                # セグのテキストデータを取得、文字起こしされていない場合は文字起こし
                transcription.transcribe_faster(file_path, MODEL_GRADE, csv_path)

                # セグメントの再生時間を取得
                time = playback_time.get_time(file_path)

                # info.csvに紐付け
                dict['file_path'].append(file_path)
                # dict['topics'].append([tags])
                dict['time'].append(time)

        # 新たなinfo.csvを作成
        df = pd.read_csv(csv_path, encoding = 'utf-8')
        df2 = pd.DataFrame(dict)
        df_merge = pd.merge(df[['file_path', 'transcription', 'transcription_timestamp']], df2, on = 'file_path', how = 'left')
        df_merge = df_merge.drop_duplicates(subset = 'file_path')
        df_merge.to_csv(csv_path, index = False, encoding = 'utf-8')

        # 各セグメントに対して、取得したテキストと時間からセグメントをさらに分割 => トピックセグメントを作成
        # トピックセグメントに対して、トピック分布と単語分布を求める => セグメント内でトピックが初めて現れる時刻を取得
    topic_mecab.Estimate_Topic_Distribution_Overall(directory_path, num_topics, 10)

    # timeカラムの設定
    df = pd.read_csv(f"./././Data/test_data/final_presentation/topic_segments_info_{topic_mecab.ROW_COUNT}_{num_topics}.csv")
    df['time'] = df.apply(lambda x: playback_time.get_time(x['file_path'].replace('\\', '/')), axis = 1)
    df.to_csv(f"./././Data/test_data/final_presentation/topic_segments_info_{topic_mecab.ROW_COUNT}_{num_topics}.csv", index = False, encoding = 'utf-8')

if __name__ == '__main__':
    mid_path = "./././Data/test_data/mid_presentation/hoshinogen"
    # for current_dir, dirs, files in os.walk(mid_path):
    #     wav_count = 0
    #     for file in files:
    #         if file.endswith('.wav'):
    #             wav_count += 1
    #     if wav_count > 0:
    #         make_segments(current_dir)

    # 自分実験
    # 確認項目
    # Topic Tilingがうまくいっているか？
    # トピックセグメントとトピックが一対一対応か？
    make_segment_info(mid_path, 200)
    print("Processing is ended")