import librosa
import librosa.display
import matplotlib.pyplot as plt

# 音声ファイルのスペクトル解析と波形の描画モジュール
def plot(audio_path):
    audio, sr = librosa.load(audio_path)
    spectrogram = librosa.stft(audio) # 短時間フーリエ変換（STFT）を使用してスペクトログラムを計算
    spectrogram_db = librosa.amplitude_to_db(abs(spectrogram))
    plt.figure(figsize=(12, 6)) # スペクトログラムをプロット
    librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.show()

if __name__ == "__main__":
    audio_path = '/Users/takuno125m/Documents/Research/Data/radiowave/RadioWave#1-master.wav'
    audio_path_hosino_chunk3 = '././Data/generate_data/hoshinogen/20240625/audio_chunk_hoshino3.wav'
    audio_path_hosino_chunk4 = '././Data/generate_data/hoshinogen/20240625/audio_chunk_hoshino4.wav'
    # plot(audio_path_hosino_chunk3)
    plot(audio_path_hosino_chunk4)
