import librosa
import librosa.display
import numpy as np
import scipy
from pydub import AudioSegment
import matplotlib.pyplot as plt

# ハイパーパラメータ
# THRESHOLD = 0.0025
# DISTANCE = 30

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

# 時間取得モジュール
def get_times(audio_path):
    y, sr = librosa.load(audio_path) # 音声ファイルの読み込み
    D = np.abs(librosa.stft(y)) # STFTを使用してスペクトルを計算
    spectral_amplitude = np.mean(D, axis = 0) #スペクトルの振幅を取得　各時間フレームの振幅の平均を取る
    for i in range(0, len(spectral_amplitude), 100):
        print(spectral_amplitude[i])
    threshold = THRESHOLD  # 適切な閾値を設定
    low_amplitude_times = np.where(spectral_amplitude > threshold)[0] # 振幅が閾値以下の時間を特定
    times_ = librosa.frames_to_time(low_amplitude_times, sr = sr) # STFTのフレームを時間に変換 <class: numpy.ndarray>
    # print(f"スペクトルが小さい時間: \n{times_}\n")
    # for value in times_:
    #     print(value)
    return times_ # print(f"スペクトルが小さい時間: {times_}")

# クラスタリングモジュール
def clustering(times):
    ctimes = times[:, None]
    D = scipy.spatial.distance.pdist(ctimes, 'cityblock')
    Z = scipy.cluster.hierarchy.linkage(D, 'single')
    cluster_labels = scipy.cluster.hierarchy.fcluster(Z, t = DISTANCE, criterion = 'distance') # <class: numpy.ndarray>
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(ctimes[i][0]) # times[i][0]で[0]を加えたのは、times[i]だとndarrayでクラスタの最大・最小値を取得できないから
    times = []
    for cluster, points in clusters.items(): # 結果を表示
        min_time = min(points)
        max_time = max(points)
        if (max_time - min_time) < 30:
            times.extend([min(points), max(points)])
        # times.extend([min(points), max(points)])
        # print(f"Cluster {cluster}: {points}\n")
    # print(f"{times}\n")
    return times

def split(audio_path, path):
    times = clustering(get_times(audio_path))
    audio_files = []
    audio_segment = AudioSegment.from_wav(audio_path)
    # print(f"分割数: {len(times)//2 - 1}")
    for i in range(len(times)//2 - 1):
        start_time = times[2*i+1]
        end_time = times[2*i+2]
        print(f"開始時間: {start_time}, 終了時間: {end_time}")
        chunk = audio_segment[start_time*1000 : end_time*1000] # pydubはミリ秒を使用するため、元の時間を1000倍する
        chunk_path = f"./././Data/generate_data/{path}/audio_chunk_hoshino{i+1}.wav"
        chunk.export(chunk_path, format = "wav")
        audio_files.append(chunk_path)

if __name__ == "__main__":
    # audio_path = '/Users/takuno125m/Documents/Research/Data/radiowave/RadioWave#1-master.wav'
    THRESHOLD = 2
    DISTANCE = 30
    audio_path_hoshino_chunk = './././Data/generate_data/hoshinogen/20240625/audio_chunk_hoshino3.wav'
    path = 'hoshinogen/20240625/03'
    split(audio_path_hoshino_chunk, path)


# 参考サイト
# [Python]動画ファイルを15分以上なら5分毎に音声ファイルに分割させる: https://qiita.com/yukiaprogramming/items/a26836626453d5716767
# [LIBROSAのコード]オーディオスペクトル解析&周波数解析のコード: https://curanzsounds.com/audio-spectrum-analysis/
# Librosaで扱える音響特徴量まとめ: https://zenn.dev/kthrlab_blog/articles/4e69b7d87a2538
# 階層的クラスタリング: https://chokkan.github.io/mlnote/unsupervised/02hac.html