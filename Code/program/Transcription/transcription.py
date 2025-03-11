from faster_whisper import WhisperModel
from gensim import corpora
from gensim.models.ldamodel import LdaModel
import torch
import pandas as pd
import MeCab
import re
import os
from itertools import groupby
import time
import nltk
from nltk.corpus import stopwords
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

if torch.cuda.is_available():
    device = torch.device('cuda')
    # print("CUDA is available. Using GPU.", torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')
    # print("CUDA is not available. Using CPU")

if os.name == 'posix':  # macOSやLinuxの場合
    neologd_path = '-d /opt/homebrew/lib/mecab/dic/mecab-ipadic-neologd'
    normal_path = '-d /opt/homebrew/lib/mecab/dic/ipadic'
    stopwords_path = "/Users/takuno125m/Documents/Research/Japanese.txt"
elif os.name == 'nt':  # Windowsの場合
    neologd_path = '-d "C:/Program Files (x86)/MeCab/dic/ipadic" -u "C:/Program Files (x86)/MeCab/dic/NEologd/NEologd.20200910-u.dic"'
    normal_path = '-d "C:/Program Files (x86)/Mecab/etc/../dic/ipadic"'
    stopwords_path = r"C:\Users\takun\Documents\laboratory\Japanese.txt"

mecab = MeCab.Tagger(neologd_path)
whisper_model_names = ['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large-v1', 'large-v2', 'large-v3', 'large', 'distil-large-v2', 'distil-medium.en', 'distil-small.en', 'distil-large-v3']


mecab = MeCab.Tagger(neologd_path)
mecab_filter = MeCab.Tagger(normal_path)
persons = {'星野': '星野源', '源': '星野源'}

nltk.download('stopwords')
stopwords_english = stopwords.words('english')
stopwords = pd.read_csv(stopwords_path, header = None)[0].to_list()
stopwords.append('笑')
stopwords.append('ー')
lowercase_letters = [chr(i) for i in range(ord('a'), ord('z') + 1)]
uppercase_letters = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
hiragana_list = [chr(i) for i in range(ord('あ'), ord('ん') + 1)]
alphabets = lowercase_letters + uppercase_letters + hiragana_list

def get_transcription_from_csv(csv_file, file_path):
    if os.path.exists(csv_file):
        # CSVファイルを読み込む
        df = pd.read_csv(csv_file)

        # ファイルのパスの規格を揃える
        file_path = file_path.replace('\\', '/')
        df['file_path'] = df['file_path'].str.replace('\\', '/', regex = False)

        # ファイルパスが一致する文字起こしデータがあるか確認
        if (file_path in df['file_path'].values):
            transcription = df[df['file_path'] == file_path]['transcription'].values[0]
            return transcription
    return None


def transcribe_faster(audio_path, model_level, csv_file):
    if not model_level in whisper_model_names:
        print(f'invalid model : {model_level}')
    existing_transcription = get_transcription_from_csv(csv_file, audio_path)
    if existing_transcription:
        return existing_transcription

    print(f"Transcribe...")
    model = WhisperModel(model_level, device = 'cuda') # device
    segments, _ = model.transcribe(audio_path, language = 'ja', without_timestamps = False)
    transcription = ''
    transcription_timestamp = ''
    for segment in segments:
        text = segment.text

        # 発話テキストからトピックを推定するには、フィラーや笑い声(笑など)、記号、感動詞は不必要 <= MeCabで分類可能
        # 文に名詞_一般, 名詞_固有名詞が含まれない場合は文を削除 & 行に対してストップワードを除いた名詞の数が0個だった場合 => 名詞のみを見れば良い
        parsed = mecab.parse(text)
        nouns = []
        for line in parsed.splitlines():
            if line == "EOS":
                break
            word, feature = line.split("\t")
            features = feature.split(",")
            if features[0] == "名詞" and (features[1] == "一般" or features[1] == "固有名詞" or features[1] == "サ変接続"):
                if len(word) == 1 and (word in alphabets):
                    continue
                nouns.append(word)

        nouns = [noun for noun in nouns if noun not in stopwords]
        nouns = [noun for noun in nouns if noun not in stopwords_english]
        nouns = [word for word in nouns if not (word.isdigit() or "ー" in word)]

        nouns2 = []
        for noun in nouns:
            filtered = mecab_filter.parse(noun)
            # 助詞・助動詞・名詞をカウント
            count_noun = 0
            count = 0
            for line2 in filtered.splitlines():
                if line2 == "EOS" or not line2.strip():
                    continue  # EOSや空行をスキップ
                parts2 = line2.split("\t")
                if len(parts2) < 2:
                    continue  # 分割後に予期しない行はスキップ
                word2, feature2 = parts2
                features2 = feature2.split(",")
                if features2[0] == "名詞" and (features2[1] == "一般" or features2[1] == "固有名詞" or features2[1] == "サ変接続"):
                    count_noun += 1
                if features2[0] == "名詞" and features2[1] == "接尾":
                    noun = noun.replace(word2, "")
                if (features2[0] == "助詞") or (features2[0] == "助動詞"):
                    count += 1
                # 登場する人名の一覧を列挙
                # if features2[0] == "名詞" and features2[2] == "人名":
                #     if word2 not in persons:
                #         persons[word2] = 1
                #     else:
                #         persons[word2] += 1

            # 助詞・助動詞が含まれていた場合、名詞が含まれていない場合
            if (count >= 1) or (count_noun == 0):
                continue
            nouns2.append(noun)

        nouns3 = []
        for noun in nouns2:
            if noun in persons:
                nouns3.append(persons[noun])
            else:
                nouns3.append(noun)

        filter_str = ['!', '?', '！', '？']
        nouns3 = [noun for noun in nouns3 if noun not in filter_str]
    
        if len(nouns3) == 0: # 名詞がそもそもない場合とストップワードしかなかった場合
            continue

        transcription += text + '\n'
        transcription_timestamp += f'[{segment.start:.2f}, {segment.end:.2f}]' + '::' + text + '\n'

    del model, segments

    min_repeats = 3
    # transcription: 同じ行が連続してmin_repeats文以上現れた場合、1行にまとめる
    lines = transcription.split('\n')
    cleaned_lines = []
    for line, group in groupby(lines):
        group_list = list(group)
        if len(group_list) >= min_repeats:
            cleaned_lines.append(line)
        else:
            cleaned_lines.extend(group_list)
    transcription = '\n'.join(cleaned_lines)


    # 新たな文字起こしデータをCSVに追加して保存
    transcriptions = pd.DataFrame([{
        'file_path': audio_path,
        'transcription': transcription,
        'transcription_timestamp': transcription_timestamp,
    }])
    if os.path.exists(csv_file):
        # 既存のCSVに追記
        transcriptions.to_csv(csv_file, mode='a', header=False, index=False, encoding = 'utf-8')
    else:
        # 新しいCSVファイルを作成
        transcriptions.to_csv(csv_file, index=False, encoding = 'utf-8')

    return transcription

def orthopedy(root):
    for current_dir, dirs, files in os.walk(root):
        for file in files:
            if file == 'info.csv':
                df = pd.read_csv(os.path.join(current_dir, file))
                # データフレームの情報
                file_paths = df['file_path'].tolist()
                transcriptions = df['transcription'].tolist()
                transcription_timestamps = df['transcription_timestamp'].tolist()
                times = df['time'].tolist()

                transcription_orthopedy = []
                for file_path, transcription in zip(file_paths, transcriptions):
                    # print(file_path)
                    if not isinstance(transcription, str):
                        transcription_orthopedy.append(transcription)
                        continue
                    lines = transcription.split('\n')
                    cleaned_lines = []
                    for line, group in groupby(lines):
                        group_list = list(group)
                        if len(group_list) >= 2:
                            cleaned_lines.append(line)
                        else:
                            cleaned_lines.extend(group_list)
                    transcription = '\n'.join(cleaned_lines)
                    transcription_orthopedy.append(transcription)

                transcription_timestamp_orthopedy = []
                for transcription_timestamp in transcription_timestamps:
                    if not isinstance(transcription_timestamp, str):
                        transcription_timestamp_orthopedy.append(transcription_timestamp)
                        continue
                    text = transcription_timestamp
                    pattern = r"\[(\d+\.\d+), (\d+\.\d+)\]::(.+)"
                    matches = re.findall(pattern, text)

                    # 文字列をまとめる処理
                    result = []
                    previous_text = None
                    start_time = None
                    end_time = None
                    repeat_count = 0

                    for match in matches:
                        current_start, current_end, current_text = match
                        if current_text == previous_text:
                            repeat_count += 1
                            end_time = current_end
                        else:
                            if previous_text:
                                result.append(f"[{start_time}, {end_time}]::{previous_text}")
                            previous_text = current_text
                            start_time = current_start
                            end_time = current_end
                            repeat_count = 1

                    # 最後の文字列を処理
                    if previous_text:
                        result.append(f"[{start_time}, {end_time}]::{previous_text}")

                    # 結果の表示
                    text = '\n'.join(result)
                    transcription_timestamp_orthopedy.append(text)

                data = {'file_path': file_paths,
                        'transcription': transcription_orthopedy,
                        'transcription_timestamp': transcription_timestamp_orthopedy,
                        'time': times
                        }
                df = pd.DataFrame(data)
                df.to_csv(os.path.join(current_dir, file), index = False, encoding = 'utf-8')
                break



if __name__ == '__main__':
    for current_dir, dirs, files in os.walk('././././Data/test_data/final_presentation/'):
        for file in files:
            print(9)