from pydub import AudioSegment
import os
import sys

def convert(file):
    # 入力ファイルがwavファイルかどうか判定
    if file.endswith('.wav'):
        newfile = file.replace('.wav', '.mp3')

        # wavをmp3に変換
        sound = AudioSegment.from_wav(file)
        sound.export(newfile, format = "mp3")

        # 元のwavファイルを削除
        os.remove(file)

def restore(file):
    if file.endswith('.mp3'):
        newfile = file.replace('.mp3', '.wav')
        sound = AudioSegment.from_mp3(file)
        sound.export(newfile, format = 'wav')
        os.remove(file)

if __name__ == '__main__':
    file = sys.argv[1]
    convert(file)