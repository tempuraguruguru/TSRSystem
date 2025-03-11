import os
import shutil
import spectral
from pydub import AudioSegment
from pydub.silence import split_on_silence
from inaSpeechSegmenter import Segmenter
from faster_whisper import WhisperModel
from transformers import BertJapaneseTokenizer, BertForNextSentencePrediction
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 無音部分の定義付け
MIN_SILENCE_LEN = 1000
SILENCE_THRESH = -42

def ss_split(input_path, output_path):
    # スペクトル解析で大まかに分割
    paths = input_path.rstrip('.wav').split('/')
    output_path += paths[-2] + '/' + paths[-1].split('_')[0] #  + '/' + paths[-1]
    print(f'output_path = {output_path}')
    os.makedirs(output_path, exist_ok = True)
    spectral.split(input_path, output_path) # ファイルには音声, CM, 楽曲, ジングルが含まれている

    # 音楽部分を検出するモデルを構築
    seg_model = Segmenter(vad_engine = 'sm', detect_gender = False)

    # 無音部分の前後を繋げるための文字起こしモデル
    whs_model = WhisperModel('large')

    # 前後の文章を繋げたときに自然な文章となるかどうか判別するモデルのロード
    nsp_model = BertForNextSentencePrediction.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

    # 指定したフォルダ内の音声ファイルから音楽部分とCM部分を削除
    rename_id = 0
    for file in os.listdir(output_path):
        # 発話部分と音楽部分の始まりと終わりの時間(s)を検出
        seg_path = output_path + '/' + file
        print(f'seg_path = {seg_path}')
        seg_data = seg_model(seg_path)
        print(seg_data)
        segments = []

        # 音楽部分のみ抽出
        for data in seg_data:
            if data[0] == 'music':
                segments.append(data)

        # 元の音声ファイルから音楽部分を削除
        audio = AudioSegment.from_file(output_path + '/' + file)
        switch = [0]
        for _, m_start, m_end in segments:
            # 開始時間と終了時間の差が60秒以上ならば楽曲、それ以外はジングル
            if (m_end - m_start) > 60:
                switch.append(m_start*1000)
                switch.append(m_end*1000)
        switch.append(len(audio))
        no_music_audio = AudioSegment.empty() # 音楽部分を削除した音声セグ
        for i in range(len(switch)//2 - 1):
            start_time = switch[2*i + 1]
            end_time = switch[2*i + 2]
            no_music_audio += audio[start_time * 1000, end_time * 1000] # ファイルには音声, CM, ジングルが含まれている

        # print(f'path : {output_path}/{file}')
        # no_music_audio = AudioSegment.from_file(output_path + '/' + file)

        # 1秒以上-42dB以下が続いたら無音部分とみなし、その部分を削除
        chunks = split_on_silence(no_music_audio, min_silence_len = MIN_SILENCE_LEN, silence_thresh = SILENCE_THRESH)
        os.makedirs(output_path + '/check', exist_ok = True)
        os.makedirs(output_path + '/sub', exist_ok = True)
        file_id = 0
        for i in range(len(chunks)-1):
            # merge.wavがあればcurrentに保存
            if os.path.isfile(output_path + '/check/merge.wav'):
                current = AudioSegment.from_file(output_path + '/check/merge.wav')
                os.remove(output_path + '/check/merge.wav')
            else:
                current = chunks[i]

            # 無音部分の前後5秒を保存
            next = chunks[i+1]
            current[len(current) - 5000:].export(output_path + '/check/current.wav', format = 'wav')
            next[:5000].export(output_path + '/check/next.wav', format = 'wav')

            # faster-whisperで前後5秒を文字起こしする
            seg_prev, _ = whs_model.transcribe(output_path + '/check/current.wav')
            seg_next, _ = whs_model.transcribe(output_path + '/check/next.wav')
            script_prev, script_next = '', ''
            for s in seg_prev:
                script_prev += str(s.text) + '.'
            for s in seg_next:
                script_next += str(s.text) + '.'

            # Next Sentence Prediction
            input_tensor = tokenizer(script_prev, script_next, return_tensors = 'pt')
            result = torch.argmax(nsp_model(**input_tensor).logits)
            if result == 0:
                # 前後のファイルを合成
                merge = current + next
                merge.export(output_path + '/check/merge.wav', format = 'wav')
            else:
                # currentを保存
                current.export(output_path + f'/sub/{str(file_id) + str(file_id)}', format = 'wav')
                file_id += 1

            # 前後のファイルを削除
            os.remove(output_path + '/check/current.wav')
            os.remove(output_path + '/check/next.wav')

    # __dirnameと__dirname/checkのファイルを削除、__dirname/subのファイルを__dirnameに保存して削除
    for file in os.listdir(output_path):
        if os.path.isfile(file):
            os.remove(file)
    shutil.rmtree(output_path + '/check')
    for file in os.listdir(output_path + '/sub'):
        shutil.move(file, output_path + f'/{paths[-1] + '_' + str(rename_id)}.wav')
        rename_id += 1
    shutil.rmtree(output_path + '/sub')


if __name__ == '__main__':
    audio_path = './././Data/ANN_hoshinogen/20240703_hoshinogen_ann.wav'
    output_path = './././Data/generate_data/'
    spectral.THRESHOLD = 0.0025
    spectral.DISTANCE = 30
    MIN_SILENCE_LEN = 1000
    SILENCE_THRESH = -42
    ss_split(audio_path, output_path)
    print("done")