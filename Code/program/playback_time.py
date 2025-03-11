from pydub import AudioSegment

def get_time(path):
    audio = AudioSegment.from_wav(path)

    # 再生時間をミリ秒で取得し、秒に変換
    duration_in_seconds = len(audio) / 1000
    return duration_in_seconds
