import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# オーディオファイルを読み込む
audio_path = '/Users/takuno125m/Documents/Research/Data/radiowave/RadioWave#1-master.wav'
audio, sr = librosa.load(audio_path)

# ピッチを抽出
pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
# タイムフレームごとに最も大きなマグニチュードのピッチを抽出
pitch = []
for t in range(pitches.shape[1]):
    index = magnitudes[:, t].argmax()
    pitch.append(pitches[index, t])

pitch = np.array(pitch)

# ピッチをプロット
plt.figure(figsize=(12, 6))
plt.plot(pitch)
plt.title('Pitch Tracking')
plt.xlabel('Time')
plt.ylabel('Pitch (Hz)')
plt.show()