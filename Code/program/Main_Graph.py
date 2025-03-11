import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
import numpy as np

if __name__ == '__main__':
    if os.name == 'posix':
        font_path = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"
    elif os.name == 'nt':
        font_path = "C:/Windows/Fonts/meiryo.ttc"
    jp_font = fm.FontProperties(fname = font_path)

    clustersizes = [2, 3, 4]
    similarities = [0.7, 0.8]
    for similarity in similarities:
        for clustersize in clustersizes:
            print(f"clustersize = {clustersize}, similarity = {similarity}")
            id_clustersize_similarity = str(clustersize) + '_' + str(similarity).replace('.', '')

            raw_df = pd.read_csv(f'./././Data/test_data/final_presentation/df_raw_{id_clustersize_similarity}/raw_results.csv')
            adjusted_df = pd.read_csv(f'./././Data/test_data/final_presentation/df_adjusted_{id_clustersize_similarity}/adjusted_results.csv')
            df = pd.merge(raw_df, adjusted_df, on = 'threshold', how = 'inner')

            thresholds = df['threshold'].tolist()
            # raw_represents = df['raw_represent'].tolist()
            raw_averages = df['raw_average'].tolist()
            # adjusted_represents = df['adjusted_represent'].tolist()
            adjusted_averages = df['adjusted_average'].tolist()


            # 生成されたトピック数
            raw_topics = pd.read_csv(f'./././Data/test_data/final_presentation/df_raw_{id_clustersize_similarity}/word_distribution_clusters.csv')
            adjusted_topics = pd.read_csv(f'./././Data/test_data/final_presentation/df_adjusted_{id_clustersize_similarity}/word_distribution_clusters.csv')
            print(f"調整しないトピック辞書のトピック数: {len(raw_topics)}")
            print(f"調整したトピック辞書のトピック数: {len(adjusted_topics)}\n")


            # グラフの作成
            plt.figure(figsize=(10, 6))

            plt.rcParams["font.size"] = 18

            # raw_ のプロット（赤色）
            # plt.plot(thresholds, raw_represents, label="Raw Represent", color='red', linestyle='--')
            plt.plot(thresholds, raw_averages, label = "トピック辞書(調整なし)", color = 'red', linestyle = '--', lw = 5)

            # adjusted_ のプロット（青色）
            # plt.plot(thresholds, adjusted_represents, label="Adjusted Represent", color='blue', linestyle='--')
            plt.plot(thresholds, adjusted_averages, label = "トピック辞書(調整あり)", color = 'blue', linestyle = '-', lw = 3)

            # 軸範囲を指定
            plt.xlim(0, 1)  # X軸の範囲を0から1に設定
            plt.ylim(0, 1)  # Y軸の範囲を0から1に設定

            # 軸目盛りを0.1ずつ増加
            plt.xticks(np.arange(0, 1.1, 0.1))  # X軸目盛りを0.1刻みに設定
            plt.yticks(np.arange(0, 1.2, 0.1))  # Y軸目盛りを0.1刻みに設定

            # 軸ラベルとタイトルの設定
            plt.xlabel('正解と見なすcos類似度の閾値', fontsize = 18, fontproperties = jp_font)
            plt.ylabel('正解率', fontsize = 18, fontproperties = jp_font)

            # グリッドの追加
            plt.grid(True, linestyle='--', alpha=0.7)
            # plt.title(f"Similarity = {similarity}, Cluster size = {clustersize}", fontproperties = jp_font, fontsize = 12)
            font_prop = FontProperties(fname = font_path, size = 20)
            plt.legend(prop = font_prop, loc = "lower left")

            # グラフの表示
            plt.tight_layout()

            os.makedirs(f"./././Data/test_data/final_presentation/img", exist_ok = True)
            plt.savefig(f"./././Data/test_data/final_presentation/img/graph_{id_clustersize_similarity}.png")

            plt.show()
        print("\n")


# clustersize = 2, similarity = 0.7
# 調整しないトピック辞書のトピック数: 100
# 調整したトピック辞書のトピック数: 140

# clustersize = 3, similarity = 0.7
# 調整しないトピック辞書のトピック数: 100
# 調整したトピック辞書のトピック数: 102

# clustersize = 4, similarity = 0.7
# 調整しないトピック辞書のトピック数: 100
# 調整したトピック辞書のトピック数: 78



# clustersize = 2, similarity = 0.8
# 調整しないトピック辞書のトピック数: 100
# 調整したトピック辞書のトピック数: 60

# clustersize = 3, similarity = 0.8
# 調整しないトピック辞書のトピック数: 100
# 調整したトピック辞書のトピック数: 13

# clustersize = 4, similarity = 0.8
# 調整しないトピック辞書のトピック数: 100
# 調整したトピック辞書のトピック数: 3