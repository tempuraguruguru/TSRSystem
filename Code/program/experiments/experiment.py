import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertJapaneseTokenizer, BertModel
import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
    # print("CUDA is available. Using GPU.", torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')
    # print("CUDA is not available. Using CPU")

tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
bert_model = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
bert_model = bert_model.to(device)

def embed2(word):
    inputs = tokenizer(word, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = bert_model(**inputs)
    last_hidden_state = outputs.last_hidden_state
    sample_vector = last_hidden_state[0][1].detach().cpu().numpy()
    return sample_vector

def experiment_topic_probaility_threshold():
    for current_dir, dirs, files in os.walk('././././Data/experiments5/decade'):
        for file in files:
            if file == 'システム_100.csv':
                os.makedirs(os.path.join(current_dir, 'experiment'), exist_ok=True)
                df_system = pd.read_csv(os.path.join(current_dir, file))

                # 四分位数を計算
                q1 = np.percentile(df_system['topic_probability'], 25)
                q2 = np.percentile(df_system['topic_probability'], 50)
                q3 = np.percentile(df_system['topic_probability'], 75)

                # それぞれの範囲で3つずつサンプリング
                # 最小値以下の外れ値
                range_below_min = df_system[df_system['topic_probability'] < q1 - 1.5 * (q3 - q1)]
                # 最大値以上の外れ値
                range_above_max = df_system[df_system['topic_probability'] > q3 + 1.5 * (q3 - q1)]

                # 各範囲を設定し、サンプル数が少ない場合はそのままの数をサンプリング
                range1 = df_system[df_system['topic_probability'] <= q1]
                range2 = df_system[(df_system['topic_probability'] > q1) & (df_system['topic_probability'] <= q2)]
                range3 = df_system[(df_system['topic_probability'] > q2) & (df_system['topic_probability'] <= q3)]
                range4 = df_system[df_system['topic_probability'] > q3]

                # サンプル数が少ない場合はそのままのデータを取得
                range_below_min_sampled = range_below_min.head(n=min(3, len(range_below_min))) if len(range_below_min) > 0 else pd.DataFrame()
                range_above_max_sampled = range_above_max.head(n=min(3, len(range_above_max))) if len(range_above_max) > 0 else pd.DataFrame()
                range1_sampled = range1.head(n=min(3, len(range1))) if len(range1) > 0 else pd.DataFrame()
                range2_sampled = range2.head(n=min(3, len(range2))) if len(range2) > 0 else pd.DataFrame()
                range3_sampled = range3.head(n=min(3, len(range3))) if len(range3) > 0 else pd.DataFrame()
                range4_sampled = range4.head(n=min(3, len(range4))) if len(range4) > 0 else pd.DataFrame()

                df_system = pd.concat([range_below_min_sampled, range_above_max_sampled, range1_sampled, range2_sampled, range3_sampled, range4_sampled], ignore_index=True)
                df_system.to_excel(os.path.join(current_dir, 'experiment', 'システム.xlsx'), index=False)

            elif file == 'ユーザー_100.csv':
                os.makedirs(os.path.join(current_dir, 'experiment'), exist_ok=True)
                df_user = pd.read_csv(os.path.join(current_dir, file))

                # 四分位数を計算
                q1 = np.percentile(df_user['topic_probability'], 25)
                q2 = np.percentile(df_user['topic_probability'], 50)
                q3 = np.percentile(df_user['topic_probability'], 75)

                # それぞれの範囲で3つずつサンプリング
                # 最小値以下の外れ値
                range_below_min = df_user[df_user['topic_probability'] < q1 - 1.5 * (q3 - q1)]
                # 最大値以上の外れ値
                range_above_max = df_user[df_user['topic_probability'] > q3 + 1.5 * (q3 - q1)]

                # 各範囲を設定し、サンプル数が少ない場合はそのままの数をサンプリング
                range1 = df_user[df_user['topic_probability'] <= q1]
                range2 = df_user[(df_user['topic_probability'] > q1) & (df_user['topic_probability'] <= q2)]
                range3 = df_user[(df_user['topic_probability'] > q2) & (df_user['topic_probability'] <= q3)]
                range4 = df_user[df_user['topic_probability'] > q3]

                # サンプル数が少ない場合はそのままのデータを取得
                range_below_min_sampled = range_below_min.head(n=min(3, len(range_below_min))) if len(range_below_min) > 0 else pd.DataFrame()
                range_above_max_sampled = range_above_max.head(n=min(3, len(range_above_max))) if len(range_above_max) > 0 else pd.DataFrame()
                range1_sampled = range1.head(n=min(3, len(range1))) if len(range1) > 0 else pd.DataFrame()
                range2_sampled = range2.head(n=min(3, len(range2))) if len(range2) > 0 else pd.DataFrame()
                range3_sampled = range3.head(n=min(3, len(range3))) if len(range3) > 0 else pd.DataFrame()
                range4_sampled = range4.head(n=min(3, len(range4))) if len(range4) > 0 else pd.DataFrame()

                df_user = pd.concat([range_below_min_sampled, range_above_max_sampled, range1_sampled, range2_sampled, range3_sampled, range4_sampled], ignore_index=True)

                # データを更新
                df_user['テキストを読んで何について話しているかtopicsの10単語からいくつか推定'] = np.nan
                df_user.to_excel(os.path.join(current_dir, 'experiment', 'ユーザー.xlsx'), index=False)


def experiment_ngram():
    Threshold = 0.383
    for current_dir, dirs, files in os.walk('././././Data/experiments5/decade'):
        for file in files:
            if file == 'システム_100.csv':
                os.makedirs(os.path.join(current_dir, 'experiment2'), exist_ok=True)
                df_system = pd.read_csv(os.path.join(current_dir, file))
                df_system = df_system[df_system['topic_probability'] >= Threshold]
                df_system = df_system.head(10)
                df_system.to_excel(os.path.join(current_dir, 'experiment2', 'システム.xlsx'), index=False)

            elif file == 'ユーザー_100.csv':
                os.makedirs(os.path.join(current_dir, 'experiment2'), exist_ok=True)
                df_user = pd.read_csv(os.path.join(current_dir, file))
                df_user = df_user[df_user['topic_probability'] >= Threshold]
                df_user['テキストを読んで何について話しているかtopicsの10単語からいくつか推定'] = np.nan
                df_user = df_user.head(10)
                df_user.to_excel(os.path.join(current_dir, 'experiment2', 'ユーザー.xlsx'), index=False)

def experiment_ngram2(root):
    Threshold = 0.383
    for current_dir, dirs, files in os.walk(root):
        for file in files:
            if file == '正解_100.csv':
                os.makedirs(os.path.join(current_dir, 'experiment'), exist_ok = True)
                df = pd.read_csv(os.path.join(current_dir, file))

                df_user = df[['file_path', 'transcription', 'topic', 'topic_probability', 'nouns', 'probabilities']]
                data_user = {'file_path': [], 'transcription': [], 'topic': [], 'topic_probability': [], 'nouns': [], 'probabilities': []}
                file_paths = df_user['file_path'].tolist()
                transcriptions = df_user['transcription'].tolist()
                topics = df_user['topic'].tolist()
                topics_probabilities = df_user['topic_probability'].tolist()
                nounses = df_user['nouns'].tolist()
                probabilitieses = df_user['probabilities'].tolist()
                for fp, ts, t, tp, ns, ps in zip(file_paths, transcriptions, topics, topics_probabilities, nounses, probabilitieses):
                    new_ns = []
                    new_ps = []
                    for noun, probability in zip(eval(ns), eval(ps)):
                        if probability > 1e-4:
                            new_ns.append(noun)
                            new_ps.append(probability)
                    data_user['file_path'].append(fp)
                    data_user['transcription'].append(ts)
                    data_user['topic'].append(t)
                    data_user['topic_probability'].append(tp)
                    data_user['nouns'].append(new_ns)
                    data_user['probabilities'].append(new_ps)
                df_user = pd.DataFrame(data_user)
                df_user = df_user[df_user['topic_probability'] >= Threshold]

                # サンプリング
                # 四分位数を計算
                q1 = np.percentile(df_user['topic_probability'], 25)
                q2 = np.percentile(df_user['topic_probability'], 50)
                q3 = np.percentile(df_user['topic_probability'], 75)

                # それぞれの範囲で3つずつサンプリング
                # 最小値以下の外れ値
                range_below_min = df_user[df_user['topic_probability'] < q1 - 1.5 * (q3 - q1)]
                # 最大値以上の外れ値
                range_above_max = df_user[df_user['topic_probability'] > q3 + 1.5 * (q3 - q1)]

                # 各範囲を設定し、サンプル数が少ない場合はそのままの数をサンプリング
                range1 = df_user[df_user['topic_probability'] <= q1]
                range2 = df_user[(df_user['topic_probability'] > q1) & (df_user['topic_probability'] <= q2)]
                range3 = df_user[(df_user['topic_probability'] > q2) & (df_user['topic_probability'] <= q3)]
                range4 = df_user[df_user['topic_probability'] > q3]

                # サンプル数が少ない場合はそのままのデータを取得
                range_below_min_sampled = range_below_min.head(n=min(3, len(range_below_min))) if len(range_below_min) > 0 else pd.DataFrame()
                range_above_max_sampled = range_above_max.head(n=min(3, len(range_above_max))) if len(range_above_max) > 0 else pd.DataFrame()
                range1_sampled = range1.head(n = min(3, len(range1))) if len(range1) > 0 else pd.DataFrame()
                range2_sampled = range2.head(n = min(3, len(range2))) if len(range2) > 0 else pd.DataFrame()
                range3_sampled = range3.head(n = min(3, len(range3))) if len(range3) > 0 else pd.DataFrame()
                range4_sampled = range4.head(n = min(3, len(range4))) if len(range4) > 0 else pd.DataFrame()

                df_user = pd.concat([range_below_min_sampled, range_above_max_sampled, range1_sampled, range2_sampled, range3_sampled, range4_sampled], ignore_index=True)

                df_user.to_excel(os.path.join(current_dir, 'experiment', 'ユーザー修正_100.xlsx'), index = False)

def TopicSegmentsDistribution(root):
    for current_dir, dirs, files in os.walk(root):
        for file in files:
            if file == '正解_100.csv':
                df = pd.DataFrame(os.path.join(current_dir, file))
                topic_probabilities = df['topic_probability'].tolist()


def TopicSegmentsDistribution_hist_and_box(root):
    topic_probabilities = []

    for current_dir, dirs, files in os.walk(root):
        for file in files:
            if file == '正解_100.csv':
                df = pd.read_csv(os.path.join(current_dir, file))
                topic_probabilities.extend(df['topic_probability'].tolist())

                # プロットの準備
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))  # 横に並べるために (1, 2)

                # ヒストグラムの描画
                sns.histplot(topic_probabilities, bins = 20, kde=True, ax=axes[0])
                axes[0].set_title(f'Distribution of Topic Probabilities (Histogram) by {current_dir.split('/')[-1]}-gram')
                axes[0].set_xlabel('Topic Probability')
                axes[0].set_ylabel('Frequency')

                # 箱ひげ図の描画
                sns.boxplot(y=topic_probabilities, ax=axes[1])
                axes[1].set_title(f'Boxplot of Topic Probabilities by {current_dir.split('/')[-1]}-gram')
                axes[1].set_ylabel('Topic Probability')

                # プロットの表示
                plt.tight_layout()
                plt.show()

def AverageRows(root):
    rows = []
    for current_dir, dirs, files in os.walk(root):
        for file in files:
            if file == 'info.csv':
                document = []
                df = pd.read_csv(os.path.join(current_dir, file))
                transcriptions = df['transcription'].tolist()
                for transcription in transcriptions:
                    if not isinstance(transcription, str):
                        continue
                    text = transcription.split('\n')
                    rows.append(len(text))
                    document.append(len(text))
                print(f"{document}")
    print(f"\n\n行数の平均: {sum(rows) / len(rows)}")


if __name__ == '__main__':
    df = pd.read_csv('././././Data/test_data/final_presentation/正解データ_25_200.csv')
    lyric_and_music = df['lyric and music'].tolist()
    raw_count = 0
    count = 0
    for ele in lyric_and_music:
        raw_count += 1
        if ele:
            count += 1
    raw_count2 = 0
    count2 = 0
    live = df['live'].tolist()
    for ele in live:
        raw_count2 += 1
        if ele:
            count2 += 1
    print(f"正解データ(lyric and music): {count}")
    print(f"正解データ(live): {count2}")

    # AverageRows('././././Data/test_data/final_presentation')
    # experiment_ngram2('././././Data/experiments6/decade')
    # TopicSegmentsDistribution_hist_and_box('././././Data/experiments6/decade')