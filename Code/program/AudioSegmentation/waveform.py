import librosa
import librosa.display
import matplotlib.pyplot as plt

# オーディオファイルを読み込む
audio_path = '/Users/takuno125m/Documents/Research/Data/radiowave/RadioWave#1-master.wav'
audio, sr = librosa.load(audio_path)

# 波形をプロット
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1) # 2行1列のグリッドの上部
librosa.display.waveshow(audio, sr=sr)
plt.title('Audio Waveform')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')

# スペクトログラムを計算
spectrogram = librosa.stft(audio)
spectrogram_db = librosa.amplitude_to_db(abs(spectrogram))

# スペクトログラムをプロット
plt.subplot(2, 1, 2) # 2行1列のグリッドの下部
librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency (Hz)')

plt.tight_layout()
plt.show()