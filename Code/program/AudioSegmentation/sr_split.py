import pandas as pd
from pyannote.audio import Pipeline
from pydub import AudioSegment
import os
import torch
from dotenv import load_dotenv

if torch.cuda.is_available():
    device = torch.device('cuda')
    # print("CUDA is available. Using GPU.", torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')
    # print("CUDA is not available. Using CPU")

load_dotenv()
USE_AUTH_TOKEN = os.getenv("USE_AUTH_TOKEN")

# ハイパーパラメータ
MAIN_SPEAKER_TIME = 60
MAIN_SPEAKER_SILENCE = 30.0

def Diarization(audio_path):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.0", use_auth_token = USE_AUTH_TOKEN)
    diarization = pipeline(audio_path)

    speakers = {}
    current_speaker = ""
    current_speaker_start = 0
    current_speaker_end = 0
    for turn, _, speaker in diarization.itertracks(yield_label = True):
        if turn.end - turn.start < 5.0:
            continue

        if current_speaker == "":
            current_speaker = speaker
            current_speaker_start = turn.start
            current_speaker_end = turn.end
        if current_speaker == speaker:
            current_speaker_end = turn.end
        else:
            print(f"発話開始時間: {current_speaker_start:.1f}s, 発話終了時間: {current_speaker_end:.1f}s, 発話者: {current_speaker}")
            current_speaker = speaker
            current_speaker_start = turn.start
            current_speaker_end = turn.end

        if speaker in speakers:
            speakers[speaker] += turn.end - turn.start
        else:
            speakers[speaker] = turn.end - turn.start
    print('\n')
    speakers = sorted(speakers.items(), key = lambda x:x[1])
    for t in speakers:
        print(f"発話者: {t[0]}, 合計発話時間: {t[1]}")

def SpeechRecognition(audio_path):
    # 話者認識モデルのロード
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.0", use_auth_token = USE_AUTH_TOKEN)
    diarization = pipeline(audio_path)

    # 発話データフレームの作成
    speakers = {}
    data = {'start': [], 'stop': [], 'speaker': []}
    for turn, _, speaker in diarization.itertracks(yield_label = True):
        # print(f"発話開始時間: {turn.start:.1f}s, 発話終了時間: {turn.end:.1f}s, 話者: {speaker}")
        data['start'].append(turn.start)
        data['stop'].append(turn.end)
        data['speaker'].append(speaker)
        if speaker in speakers:
            speakers[speaker] += turn.end - turn.start
        else:
            speakers[speaker] = turn.end - turn.start
    df = pd.DataFrame(data)

    # 60秒以上話している話者をメインスピーカーとして定義
    main_speakers = [key for key, value in speakers.items() if value >= MAIN_SPEAKER_TIME]

    # メインスピーカー以外の話者の行データを除去
    df_main = df[df['speaker'].isin(main_speakers)].reset_index(drop = True)
    threshold = MAIN_SPEAKER_SILENCE
    segments = []
    current_segment = []
    previous_stop = None

    # メインスピーカーのデータ取得
    # start_times = df_main['start'].tolist()
    # end_times = df_main['end'].tolist()
    # speaker_names = df['speaker'].tolist()
    # for start_time, end_time, speaker_name in zip(start_times, end_times, speaker_names):
    #     print(f"発話開始時間: {start_time:.1f}s, 発話終了時間: {end_time:.1f}s, 話者: {speaker_name}")
    # df_main.to_excel('./speaker.csv', index = False, encoding = 'utf-8')
    # return

    # データフレームの分割
    for index, row in df_main.iterrows():
        if previous_stop is not None:
            silence_duration = row['start'] - previous_stop
            if silence_duration >= threshold:
                segments.append(current_segment)
                current_segment = []
        current_segment.append(row)
        previous_stop = row['stop']
    if current_segment:
        segments.append(current_segment)


    # 音声データの分割
    audio = AudioSegment.from_file(audio_path, format = 'wav')
    for i, segment in enumerate(segments):
        # 分割したデータフレームのstartの最小値(発話者の話始め), stopの最大値(発話者の話終わり)を取得
        data = {'start': [], 'stop': [], 'speaker': []}
        for s in segment:
            data['start'].append(s['start'])
            data['stop'].append(s['stop'])
            data['speaker'].append(s['speaker'])
        seg = pd.DataFrame(data)
        seg_start = min(seg['start'].values.tolist())
        seg_stop = max(seg['stop'].values.tolist())

        # セグメントの開始時間と終了時間の調整
        if seg_start - 5.0 >= 0:
            seg_start -= 5.0
        else:
            seg_start = 0
        if seg_stop + 5.0 <= audio.duration_seconds:
            seg_stop += 5.0
        else:
            seg_stop = audio.duration_seconds

        # データの切り出し
        split_path = '/'.join(audio_path.split('/')[:-1]) + f'/split_audio_{audio_path.split('_')[-1].replace('.wav', '')}_{i+1}.wav'
        split_audio = audio[seg_start*1000 : seg_stop*1000]
        split_audio.export(split_path, format = "wav")
        print(f'Segment {i+1}: start = {seg_start}, stop = {seg_stop}')

if __name__ == '__main__':
    mid_path = './././Data/generate_data/hoshinogen/20240703/audio_chunk_5.wav'
    # Diarization(mid_path)
    SpeechRecognition(mid_path)