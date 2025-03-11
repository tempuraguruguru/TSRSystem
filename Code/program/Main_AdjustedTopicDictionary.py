import os
import pandas as pd
import numpy as np
import json
from transformers import BertJapaneseTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from pydub import AudioSegment
from pydub.playback import play
import sys
import ast
import torch
import random
import shutil
import MeCab
from gensim import corpora
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools

MODEL_GRADE = "large"
SIMILARITY_TOPIC_THRESHOLD = 0.80 # 類似トピックと見なすときの閾値
NOUN_THRESHOLD = 0.01 # 名詞の埋め込み表現に生起確率を乗算したもの

if torch.cuda.is_available():
    device = torch.device('cuda')
    # print("CUDA is available. Using GPU.", torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')
    # print("CUDA is not available. Using CPU")

# OSの確認
if os.name == 'posix':  # macOSやLinuxの場合
    neologd_path = '-d /opt/homebrew/lib/mecab/dic/mecab-ipadic-neologd'
    normal_path = '-d /opt/homebrew/lib/mecab/dic/ipadic'
    stopwords_path = "/Users/takuno125m/Documents/Research/Japanese.txt"
elif os.name == 'nt':  # Windowsの場合
    neologd_path = '-d "C:/Program Files (x86)/MeCab/dic/ipadic" -u "C:/Program Files (x86)/MeCab/dic/NEologd/NEologd.20200910-u.dic"'
    normal_path = '-d "C:/Program Files (x86)/Mecab/etc/../dic/ipadic"'
    stopwords_path = r"C:\Users\takun\Documents\laboratory\Japanese.txt"

mecab = MeCab.Tagger(neologd_path)
mecab_filter = MeCab.Tagger(normal_path)
persons = {'星野': '星野源', '源': '星野源'}

nltk.download('stopwords')
stopwords_english = stopwords.words('english')
stopwords = pd.read_csv(stopwords_path, header = None)[0].to_list()
stopwords.append('笑')
lowercase_letters = [chr(i) for i in range(ord('a'), ord('z') + 1)]
uppercase_letters = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
hiragana_list = [chr(i) for i in range(ord('あ'), ord('ん') + 1)]
alphabets = lowercase_letters + uppercase_letters + hiragana_list

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

def average_embedding(words):
    vectors = [embed2(word) for word in words]
    average_vector = np.mean(vectors, axis=0)
    return average_vector

def extract_nouns2(text):
    if not isinstance(text, str):
        raise TypeError(f"Expected a string but got {type(text)}: {text}")

    parsed = mecab.parse(text)
    nouns = []
    for line in parsed.splitlines():
        if line == "EOS" or not line.strip():
            continue  # EOSや空行をスキップ
        parts = line.split("\t")
        if len(parts) < 2:
            continue  # 分割後に予期しない行はスキップ
        word, feature = parts
        features = feature.split(",")
        # 名詞または複合名詞を抽出
        if features[0] == "名詞" and (features[1] == "一般" or features[1] == "固有名詞" or features[1] == "サ変接続"):
            if len(word) == 1 and (word in alphabets):
                continue
            nouns.append(word)

    # フィルタリング
    nouns = [noun for noun in nouns if noun not in stopwords]
    nouns = [noun for noun in nouns if noun not in stopwords_english]
    nouns = [word for word in nouns if not (word.isdigit() or "ー" in word)]

    nouns2 = []
    for noun in nouns:
        filtered = mecab_filter.parse(noun)
        # 助詞・助動詞・名詞をカウント
        count_noun = 0
        count = 0
        for line2 in filtered.splitlines():
            if line2 == "EOS" or not line2.strip():
                continue  # EOSや空行をスキップ
            parts2 = line2.split("\t")
            if len(parts2) < 2:
                continue  # 分割後に予期しない行はスキップ
            word2, feature2 = parts2
            features2 = feature2.split(",")
            if features2[0] == "名詞" and (features2[1] == "一般" or features2[1] == "固有名詞" or features2[1] == "サ変接続"):
                count_noun += 1
            if features2[0] == "名詞" and features2[1] == "接尾":
                noun = noun.replace(word2, "")
            if (features2[0] == "助詞") or (features2[0] == "助動詞"):
                count += 1
            # 登場する人名の一覧を列挙
            # if features2[0] == "名詞" and features2[2] == "人名":
            #     if word2 not in persons:
            #         persons[word2] = 1
            #     else:
            #         persons[word2] += 1

        # 助詞・助動詞が含まれていた場合、名詞が含まれていない場合
        if (count >= 1) or (count_noun == 0):
            continue
        nouns2.append(noun)

    nouns3 = []
    for noun in nouns2:
        if noun in persons:
            nouns3.append(persons[noun])
        else:
            nouns3.append(noun)

    filter_str = ['!', '?', '！', '？']
    nouns3 = [noun for noun in nouns3 if noun not in filter_str]

    return nouns3

def nouns_similarity(nouns1, nouns2):
    if not nouns1 or not nouns2:
        return 0
    # 各名詞を埋め込みベクトルに変換
    nouns1_vector = [embed2(noun) for noun in nouns1]
    nouns2_vector = [embed2(noun) for noun in nouns2]
    # 各ペアのコサイン類似度を計算
    similarity_sum = 0
    count = 0
    for vec1 in nouns1_vector:
        for vec2 in nouns2_vector:
            similarity = cosine_similarity([vec1], [vec2])[0][0]
            similarity_sum += similarity
            count += 1

    # 平均類似度を計算
    average_similarity = similarity_sum / count
    return average_similarity

def clustering_topic_dictionary(nouns, probabilities, probability_threshold, similarity_threshold):
    removed_nouns = [noun for noun, prob in zip(nouns, probabilities) if prob > probability_threshold]
    if not removed_nouns:
        return []

    embeds = np.array([embed2(noun) for noun in removed_nouns])
    similarity_matrix = cosine_similarity(embeds)
    clusters = []
    used_indices = set()
    for i, noun in enumerate(removed_nouns):
        if i in used_indices:
            continue
        cluster = [noun]
        used_indices.add(i)
        for j in range(i + 1, len(removed_nouns)):
            if j not in used_indices and similarity_matrix[i, j] > similarity_threshold:
                cluster.append(removed_nouns[j])
                used_indices.add(j)
        clusters.append(cluster)
    return clusters

def clustering_topic_dictionary2(nouns, probabilities, probability_threshold, similarity_threshold):
    filtered_nouns_probs = [(noun, prob) for noun, prob in zip(nouns, probabilities) if prob > probability_threshold]
    if not filtered_nouns_probs:
        return []

    removed_nouns = [noun for noun, _ in filtered_nouns_probs]
    embeds = np.array([embed2(noun) for noun in removed_nouns])
    similarity_matrix = cosine_similarity(embeds)

    clusters = []
    used_indices = set()
    for i, (noun, prob) in enumerate(filtered_nouns_probs):
        if i in used_indices:
            continue
        cluster = [(noun, prob)]
        used_indices.add(i)
        for j in range(i + 1, len(filtered_nouns_probs)):
            if j not in used_indices and similarity_matrix[i, j] > similarity_threshold:
                cluster.append(filtered_nouns_probs[j])
                used_indices.add(j)
        clusters.append(cluster)

    return clusters

def cluster_average_similarity(nouns):
    embeddings = [embed2(noun) for noun in nouns]
    similarity_matrix = cosine_similarity(embeddings)
    upper_triangle_indices = np.triu_indices_from(similarity_matrix, k=1)
    upper_triangle_values = similarity_matrix[upper_triangle_indices]
    average_similarity = np.mean(upper_triangle_values)
    return average_similarity

def CountNouns(texts):
    dict = {}
    nouns_counter = Counter()
    for text in texts.split('\n'):
        nouns = extract_nouns2(text)
        nouns_counter.update(nouns)
    for noun, count in nouns_counter.items():
        dict[noun] = count
    return dict

def extract_utterances(transcription, transcriptions_timestamp, timestamp_list):
    if len(timestamp_list) == 1:
        return transcription, transcriptions_timestamp

    # 文字列を辞書に変換
    timestamps_and_texts = {}
    for line in transcriptions_timestamp.split('\n'):
        if len(line) == 0:
            continue
        lines = line.split('::')
        timestamp = lines[0]
        text = lines[1]
        start = eval(timestamp)[0]
        timestamps_and_texts[float(start)] = text

    # 発話内容の抽出
    utterances = []
    utterances_timestamp = []
    last_end = 0.0
    for start, end in zip(timestamp_list[:-1], timestamp_list[1:]):
        utterance = ""
        utterance_timestamp = ""
        for timestamp, text in timestamps_and_texts.items():
            if start <= timestamp < end:
                utterance += text + '\n'
                utterance_timestamp += f'[{timestamp}]::' + text + '\n'
        utterances.append(utterance)
        utterances_timestamp.append(utterance_timestamp)
        last_end = end
    utterance = ""
    utterance_timestamp = ""
    for timestamp, text in timestamps_and_texts.items():
        if last_end <= timestamp:
            utterance += text + '\n'
            utterance_timestamp += f'[{timestamp}]::' + text + '\n'
    utterance.rstrip('\n')
    utterance_timestamp.rstrip('\n')
    utterances.append(utterance)
    utterances_timestamp.append(utterance_timestamp)
    return utterances, utterances_timestamp

def calculate_topic_distribution_for_text(text, topic_data):
    """
    文書のトピック分布を計算する関数
    :param text: 対象の文書テキスト
    :param topic_data: 各トピックの名詞とその確率を含むデータ
    :return: トピック分布（確率）
    """
    nouns_in_text = extract_nouns2(text)

    # トピック分布を初期化
    num_topics = len(topic_data)
    topic_distribution = np.zeros(num_topics)

    # 各トピックに関連する名詞とその確率を取得
    for idx, row in topic_data.iterrows():
        topic_nouns = row['nouns']  # トピックの関連名詞
        topic_probabilities = row['probabilities']  # トピックの関連名詞の生起確率
        # 文書内の各名詞が関連するトピックに対して寄与する確率を加算
        for noun, prob in zip(topic_nouns, topic_probabilities):
            if noun in nouns_in_text:
                topic_distribution[idx] += prob

    # 正規化して確率分布にする
    total_prob = topic_distribution.sum()
    if total_prob > 0:
        topic_distribution /= total_prob

    # すべてのトピックを出力
    topic_distribution = [(i, val) for i, val in enumerate(topic_distribution)]

    return topic_distribution


if __name__ == '__main__':
    # n_gram = 38, num_topics = 64, probability_threshold = 0.0407, similarity_threshold = 0.6685, clustersize_threshold = 2
    n_gram_list = [7]
    num_topics_list = [100]
    probability_thresholds = [1e-4]
    similarity_thresholds = [0.8]
    clustersize_thresholds = [4]

    X = similarity_thresholds
    X_label = "Similarity"
    sample_size = len(average_embedding(['サンプル']))

    average_similarities = []
    average_cluster_size = []
    average_ratio_td = []
    topic_dictionnary_size = []
    id_clustersize_similarity = str(clustersize_thresholds[0]) + '_' + str(similarity_thresholds[0]).replace('.', '')
    os.makedirs(f'./././Data/test_data/final_presentation/df_adjusted_{id_clustersize_similarity}', exist_ok = True)
    for n_gram_ele in n_gram_list:
        for num_topics_ele in num_topics_list:
            Number_of_Topics = num_topics_ele
            N_gram = n_gram_ele

            data = {'file_path': [], 'transcription': [], 'transcription_timestamp': [], 'time': []}
            for current_dir, dirs, files in os.walk('./././Data/test_data/final_presentation'):
                for file in files:
                    if file == 'info.csv':
                        segment = pd.read_csv(os.path.join(current_dir, file))
                        data['file_path'] += segment['file_path'].tolist()
                        data['transcription'] += segment['transcription'].tolist()
                        data['transcription_timestamp'] += segment['transcription_timestamp'].tolist()
                        data['time'] += segment['time'].tolist()
                        break
            df = pd.DataFrame(data)
            df = df.dropna(how = 'any')
            fpaths = df['file_path'].tolist()
            documents = df['transcription'].tolist()
            documents_timestamp = df['transcription_timestamp'].tolist()

            subdocuments = []
            startTimes = []
            for i in range(len(documents)):
                tdocs = documents_timestamp[i].split('\n') # documents_timestampを1行ずつ取り出しリストに格納

                # tdocsにおかしな部分がないか探索
                a = []
                for tdoc in tdocs:
                    if len(tdoc) == 0:
                        continue
                    b = tdoc.split('::')
                    if b[1] == '':
                        continue
                    a.append(tdoc)
                tdocs = a

                # 発話音声データの文字起こしテキストを行単位のn-gramで取り出す
                n_grams = []
                n_grams_time = []
                if len(tdocs) - N_gram <= 0:
                    doc = ''
                    for l in range(len(tdocs)):
                        docs = tdocs[l].split('::')
                        if l == 0:
                            doc += docs[1]
                            start = eval(docs[0])[0]
                            n_grams_time.append(start)
                    n_grams.append(doc)
                else:
                    for j in range(len(tdocs) - N_gram + 1):
                        doc = ''
                        for k in range(N_gram):
                            docs = tdocs[j+k].split('::')
                            if k == 0:
                                doc += docs[1]
                                start = eval(docs[0])[0]
                                n_grams_time.append(start)
                            else:
                                doc += docs[1]
                        n_grams.append(doc)

                subdocuments.append(n_grams)
                startTimes.append(n_grams_time)

            flattened_subdocuments = [doc for sublist in subdocuments for doc in sublist]
            tokenized_documents = [extract_nouns2(doc) for doc in flattened_subdocuments]
            dictionary = corpora.Dictionary(tokenized_documents)
            corpus = [dictionary.doc2bow(doc) for doc in tokenized_documents]

            # LDAモデルの構築
            lda_model = LdaModel(
                corpus = corpus,
                id2word = dictionary,
                num_topics = Number_of_Topics,
                random_state = 42,
                passes = 2,
                alpha = 'auto',
            )
            print(f"生成されたトピックの数: {lda_model.num_topics}")



            # トピック辞書を調整 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            topics_previous = lda_model.show_topics(num_topics = -1, formatted = False)
            topics_new = []
            probability_threshold = 1e-4
            topics_raw = {'id': [], 'nouns': [], 'probabilities': []}
            topics_clustered = {'id': [], 'nouns': [], 'probabilities': []}
            for topic_id, topic_terms in topics_previous:
                nouns = [word for word, probability in topic_terms if probability >= probability_threshold]
                probabilities = [probability for word, probability in topic_terms if probability >= probability_threshold]
                topics_raw['id'].append(topic_id)
                topics_raw['nouns'].append(nouns)
                topics_raw['probabilities'].append(probabilities)
                clusters = clustering_topic_dictionary2(nouns, probabilities, probability_threshold, similarity_thresholds[0])
                clusters2 = [cluster for cluster in clusters if len(cluster) >= clustersize_thresholds[0]]
                for cluster in clusters2:
                    topics_new.append(cluster)
            for i, topic_new in enumerate(topics_new):
                nouns = [word for word, probability in topic_new]
                probabilities = [probability for word, probability in topic_new]
                topics_clustered['id'].append(i)
                topics_clustered['nouns'].append(nouns)
                topics_clustered['probabilities'].append(probabilities)

            # トピック辞書を作成
            df_clustered = pd.DataFrame(topics_clustered)
            df_clustered.to_csv(f'./././Data/test_data/final_presentation/df_adjusted_{id_clustersize_similarity}/df_clustered.csv', index = False, encoding = 'utf-8')
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



            # トピック辞書を調整した場合のトピック分布計算
            doc_topic_distributions = []
            for subdocs in subdocuments:
                sub_doc_topic_distributions = []
                for subdoc in subdocs: # subdocはn-gram
                    sub_doc_topics = calculate_topic_distribution_for_text(subdoc, df_clustered)
                    sub_doc_topic_distributions.append(sub_doc_topics)
                doc_topic_distributions.append(sub_doc_topic_distributions)



            # 分割文書ごとのトピック分布を表示
            data = []
            Segments = []
            i = 0

            # doc_topic_distributionは発話音声データの数  :  sub_doc_topic_distributionはn-gramにした文のトピック分布の情報
            for fp, sub_doc_topic_distributions in zip(fpaths, doc_topic_distributions):
                Segment_Dic = {'file_path': fp ,'topics': None, 'start_time': None}

                topic_segment_topic = []
                topic_segment_start_time = []

                # 現在の窓のトピック情報
                if len(sub_doc_topic_distributions) == 0:
                    print(f"sub_doc_topic_distribution: size = 0 !!!!!")
                    continue
                topic_sequence = []
                similarities = []
                program_times = []
                for j in range(len(sub_doc_topic_distributions) - N_gram):
                    if len(sub_doc_topic_distributions[j+N_gram]) == 0:
                        continue
                    # 現在の窓のトピック情報
                    current_topic_id = (max(sub_doc_topic_distributions[j], key = lambda x : x[1]))[0]
                    current_topic_vec = np.array([prob for _, prob in sub_doc_topic_distributions[j]]).reshape(1, -1)
                    current_topic_start = startTimes[i][j]

                    # 次の窓のトピック情報
                    next_topic_id = (max(sub_doc_topic_distributions[j+N_gram], key = lambda x : x[1]))[0]
                    next_topic_vec = np.array([prob for _, prob in sub_doc_topic_distributions[j+N_gram]]).reshape(1, -1)
                    next_topic_start = startTimes[i][j+N_gram]

                    # 類似度計算
                    similarity = cosine_similarity(current_topic_vec, next_topic_vec)

                    # 情報の保持
                    topic_sequence.append(current_topic_id)
                    similarities.append(similarity[0][0])
                    program_times.append(current_topic_start)

                mean_similarity = np.mean(similarities)
                # std_similarity = np.std(similarities)
                above_mean = False
                for topic_id, similarity, start_topic_time in zip(topic_sequence, similarities, program_times):
                    if similarity >= mean_similarity:
                        if not above_mean:
                            topic_segment_topic.append(topic_id)
                            topic_segment_start_time.append(start_topic_time)
                            above_mean = True
                    else:
                        if above_mean:
                            topic_segment_topic.append(topic_id)
                            topic_segment_start_time.append(start_topic_time)
                            above_mean = False

                i += 1
                Segment_Dic['topics'] = topic_segment_topic
                Segment_Dic['start_time'] = topic_segment_start_time
                Segments.append(Segment_Dic)

            # 発話音声データの情報を格納
            All_Segments_Info = {'file_path': [], 'topics': [], 'start_time': []}
            All_Topics_Info = {'topic': [], 'nouns': [], 'probabilities': []}
            has_topics = []
            segment_count = 0
            for seg in Segments:
                if len(seg['topics']) == 0 and len(seg['start_time']) == 0:
                    continue
                for s in seg['topics']:
                    has_topics.append(s)
                    segment_count += 1
                All_Segments_Info['file_path'].append(seg['file_path'])
                All_Segments_Info['topics'].append(seg['topics'])
                All_Segments_Info['start_time'].append(seg['start_time'])



            # トピック辞書を調整して用いる場合
            topic_dictionary = pd.read_csv(f'./././Data/test_data/final_presentation/df_adjusted_{id_clustersize_similarity}/df_clustered.csv')
            ids = topic_dictionary['id'].tolist()
            nounses = topic_dictionary['nouns'].tolist()
            probabilitieses = topic_dictionary['probabilities'].tolist()
            for probability_threshold in probability_thresholds:
                for similarity_threshold in similarity_thresholds:
                    for clustersize_threshold in clustersize_thresholds:
                        removed_nouns = []
                        clusterses = []
                        most_clusterses = []
                        clusterses_size = []
                        clusterses_average_vector = []
                        clusters_ratio = []

                        # 平均ベクトルと代表ベクトルの算出
                        average_vectors = []
                        average_most_vectors = []

                        for nouns, probabilities in zip(nounses, probabilitieses):
                            # 生起確率が閾値より小さい名詞を除去
                            removed = []
                            for noun, probability in zip(eval(nouns), eval(probabilities)):
                                if probability > probability_threshold:
                                    removed.append(noun)
                            removed_nouns.append(removed)

                            # 関連名詞の平均ベクトルの計算
                            average_vector = average_embedding(removed)
                            average_vectors.append(average_vector)

                            # 名詞をクラスタリング
                            clusters = clustering_topic_dictionary(eval(nouns), eval(probabilities), probability_threshold, similarity_threshold)
                            clusterses.append(clusters)
                            clusters = [cluster for cluster in clusters if len(cluster) >= clustersize_threshold]

                            # クラスタがない場合、このループをスキップ
                            if len(clusters) == 0:
                                most_clusterses.append([])
                                clusterses_size.append(0)
                                clusterses_average_vector.append(0)
                                clusters_ratio.append(0)
                                average_most_vectors.append(np.zeros(sample_size))
                                continue

                            # 代表クラスタの特定
                            cluster_max_size = clusters[0]
                            for cluster in clusters:
                                if len(cluster_max_size) < len(cluster):
                                    cluster_max_size = cluster
                            most_clusterses.append(cluster_max_size)
                            clusterses_size.append(len(cluster_max_size))

                            # 代表クラスタの平均ベクトル
                            average_most_vector = average_embedding(cluster_max_size)

                            # 代表クラスタの名詞の全体的なコサイン類似度
                            if len(cluster_max_size) == 1:
                                clusterses_average_vector.append(1)
                                clusters_ratio.append(1)
                            else:
                                clusterses_average_vector.append(cluster_average_similarity(cluster_max_size))
                                clusters_ratio.append(len(cluster_max_size) / len(removed))
                            average_most_vectors.append(average_most_vector)

                        if len(clusterses_size) == 0:
                            average_cluster_size.append(0)
                        else:
                            average_cluster_size.append(sum(clusterses_size) / len(clusterses_size))
                        if len(clusterses_average_vector) == 0:
                            average_similarities.append(0)
                        else:
                            average_similarities.append(sum(clusterses_average_vector) / len(clusterses_average_vector))

                        topic_dictionnary_size.append(len(clusterses_average_vector))
                        if len(clusters_ratio) == 0:
                            average_ratio_td.append(0)
                        else:
                            average_ratio_td.append(sum(clusters_ratio) / len(clusters_ratio))

    # fig, ax1 = plt.subplots(figsize=(15, 9))
    # # 左軸にプロット
    # ax1.plot(X, average_ratio_td, label='Average Ratio', marker='o', color='b')
    # ax1.set_xlabel(f'{X_label}')
    # ax1.set_ylabel('Average Ratio', color='b')
    # ax1.tick_params(axis='y', labelcolor='b')

    # # 右軸を作成
    # # ax2 = ax1.twinx()
    # # ax2.plot(X, topic_dictionnary_size, label='Topic Dictionary Size', marker='x', color='r')
    # # ax2.set_ylabel('Topic Dictionary Size', color='r')
    # # ax2.tick_params(axis='y', labelcolor='r')

    # ax2 = ax1.twinx()
    # ax2.plot(X, average_similarities, label='Nouns Similarity', marker='x', color='r')
    # ax2.set_ylabel('Nouns Similarity', color='r')
    # ax2.tick_params(axis='y', labelcolor='r')

    # # グリッドとレイアウト調整
    # plt.grid(True)
    # fig.tight_layout()

    # plt.show()

    # print(f"クラスタが１つ: {count_one}")
    # print(f"クラスタが複数: {count_many}")
    # print(f"クラスタの類似度の平均: {sum(clusterses_average_vector) / len(clusterses_average_vector)}")
    # print(f"クラスタの平均サイズ: {sum(clusterses_size) / len(clusterses_size)}")
    # print("processing end!")

    # print(len(ids))
    # print(len(nounses))
    # print(len(probabilitieses))
    # print(len(removed_nouns))
    # print(len(most_clusterses))
    # print(len(clusters_ratio))
    # print(len(clusterses))
    # print(len(clusterses_average_vector))
    # print(f"size: {len(average_similarities)}, {average_similarities}")
    # print(f"size: {len(average_cluster_size)}, {average_cluster_size}")
    # print(f"size: {len(topic_dictionnary_size)}, {topic_dictionnary_size}")

    data = {
        'topic': ids,
        'nouns': nounses,
        'probabilities': probabilitieses,
        'removed_nouns': removed_nouns,
        'most_clusters': most_clusterses,
        'ratio': clusters_ratio,
        'clusterses_average_similarity': clusterses_average_vector,
        'clusters': clusterses,
        'average_vector': average_vectors,
        'average_most_vector': average_most_vectors
    }
    clusters_df = pd.DataFrame(data)
    clusters_df.to_csv(f'./././Data/test_data/final_presentation/df_adjusted_{id_clustersize_similarity}/word_distribution_clusters.csv', index = False, encoding = 'utf-8')

    df2 = pd.DataFrame(All_Segments_Info)
    df_merge = pd.merge(df, df2, on = 'file_path', how = 'inner')
    # df_merge.to_csv("./././Data/test_data/final_presentation/df_merge.csv", index = False, encoding = "utf-8")

    # 各トピックセグメントのトピック生起確率を求める
    file_paths = df_merge['file_path'].tolist()
    transcriptions = df_merge['transcription'].tolist()
    transcriptions_timestamp = df_merge['transcription_timestamp'].tolist()
    topicses = df_merge['topics'].tolist()
    times = df_merge['time'].tolist()
    start_times = df_merge['start_time'].tolist()
    topics_probabilitieses = []
    topicses_new = []
    start_times_new = []
    topic_segment_data = {
        'file_path': [], 
        'transcription': [], 
        'transcription_timestamp': [], 
        'transcription_nouns': [], 
        'transcription_nouns_frequency': [], 
        'start': [], 
        'end': [], 
        'topic_distribution': [], 
        'topic': [], 
        'topic_probability': []
    }
    for file_path, transcription, transcription_timestamp, topics, start_time in zip(file_paths, transcriptions, transcriptions_timestamp, topicses, start_times):
        topics_probabilities = []

        L1_new = []
        L2_new = []
        # print(topics)
        current_value = topics[0]
        L1_new.append(current_value)
        L2_new.append(start_time[0])
        for i in range(1, len(topics)):
            if topics[i] != current_value:
                current_value = topics[i]
                L1_new.append(current_value)
                L2_new.append(start_time[i])
        topics_new = L1_new
        start_time_new = L2_new

        topicses_new.append(topics_new)
        start_times_new.append(start_time_new)

        utterances, utterances_timestamp = extract_utterances(transcription, transcription_timestamp, start_time_new)
        for topic, utterance, utterance_timestamp, start, end in zip(topics_new, utterances, utterances_timestamp, start_time_new[:-1], start_time_new[1:]):
            utterance_nouns = extract_nouns2(utterance)
            utterance_bow = dictionary.doc2bow(utterance_nouns)
            # topic_distribution = lda_model.get_document_topics(utterance_bow, minimum_probability = 0.0)
            topic_distribution = calculate_topic_distribution_for_text(utterance, df_clustered)
            topic_distribution2 = []
            total = 0
            for tt in topic_distribution:
                total += tt[1]
            t_average = total / len(topic_distribution)
            # print(t_average) # 平均は0.0049999 ~ 0.0050000
            for tt in topic_distribution:
                if tt[1] > t_average:
                    topic_distribution2.append((tt[0], tt[1]))

            # トピックセグメントの情報(パス、テキスト、トピック、そのトピックの生起確率)
            most_topic = max(topic_distribution, key = lambda x : x[1])
            topic_segment_data['file_path'].append(file_path)
            topic_segment_data['transcription'].append(utterance)
            topic_segment_data['transcription_timestamp'].append(utterance_timestamp)
            topic_segment_data['transcription_nouns'].append(set(utterance_nouns))
            topic_segment_data['transcription_nouns_frequency'].append(CountNouns(utterance))
            topic_segment_data['topic'].append(most_topic[0])
            topic_segment_data['topic_probability'].append(most_topic[1])
            topic_segment_data['topic_distribution'].append(topic_distribution2)
            topic_segment_data['start'].append(start)
            topic_segment_data['end'].append(end)

        # 1つのtranscriptionごとにトピック確率を追加
        topics_probabilitieses.append(topics_probabilities)

    data = {'file_path': file_paths, 'transcription': transcriptions, 'transcription_timestamp': transcriptions_timestamp,
            'time': times, 'topics': topicses_new, 'start_time': start_times_new, 'topic_probabilities': topics_probabilitieses}
    df = pd.DataFrame(data)
    df.to_csv(f'./././Data/test_data/final_presentation/df_adjusted_{id_clustersize_similarity}/topic_segments_info.csv', index = False, encoding = 'utf-8')

    # トピックセグメントのcsv
    segment_df = pd.DataFrame(topic_segment_data)
    ids = [id for id in range(len(segment_df))]
    ids = {'id': ids}
    ids = pd.DataFrame(ids)
    segment_df = pd.concat([segment_df, ids], axis = 1)
    # segment_df.to_csv('./././Data/test_data/final_presentation/segment_df.csv', index = False, encoding = 'utf-8')
    df = pd.merge(segment_df, clusters_df, on = 'topic', how = 'inner')
    df.to_csv(f'./././Data/test_data/final_presentation/df_adjusted_{id_clustersize_similarity}/topic_segments_df.csv', index = False, encoding = 'utf-8')


    # similarity_threshold = 0.6685, clustersize_threhold = 2
    # similarity_thresholds = [0.8]
    # clustersize_thresholds = [4]

    Y = similarity_thresholds
    Y_label = "Similarity"

    average_similarities = []
    average_cluster_size = []

    segment_score = []

    ids = df['id'].tolist()
    file_paths = df['file_path'].tolist()
    transcriptions = df['transcription'].tolist()
    td_most_clusters = df['most_clusters'].tolist()
    td_average_vectors = df['average_vector'].tolist()
    td_most_vectors = df['average_most_vector'].tolist()
    for similarity_threshold in similarity_thresholds:
        for clustersize_threshold in clustersize_thresholds:
            probability_threshold = 1e-4

            clusterses = []
            most_clusterses = []
            clusterses_size = []
            most_clusters_ratio = [] #ratio
            clusters_size = [] # cluster_size
            clusterses_average_vector = []
            topic_segment_data = {
                'file_path': file_paths,
                'id': ids,
                'transcription': transcriptions,
                'transcription_nouns': [],
                'clusters': [],
                'most_cluster': [],
                'most_cluster_ratio': [],
                'clusters_size': [],
                'represent_matching_degree': [],
                'average_matching_degree': []
            }

            for transcription, td_most_vector, td_average_vector in zip(transcriptions, td_most_vectors, td_average_vectors):
                nouns2 = extract_nouns2(transcription)
                nouns = list(set(nouns2))

                # トピックセグメントの平均ベクトル
                ts_avector = average_embedding(nouns2)
                topic_segment_data['average_matching_degree'].append(cosine_similarity([ts_avector], [td_average_vector])[0][0])

                topic_segment_data['transcription_nouns'].append(nouns2)
                probabilities = [0 for _ in range(len(nouns2))]
                probability_threshold = -1

                clusters = clustering_topic_dictionary(nouns2, probabilities, probability_threshold, similarity_threshold)
                clusterses.append(clusters)
                topic_segment_data['clusters'].append(clusters)
                clusters = [cluster for cluster in clusters if len(cluster) >= clustersize_threshold]

                # クラスタがない
                if len(clusters) == 0:
                    clusterses_average_vector.append(0)
                    topic_segment_data['most_cluster'].append(-1)
                    topic_segment_data['most_cluster_ratio'].append(-1)
                    topic_segment_data['clusters_size'].append(-1)
                    topic_segment_data['represent_matching_degree'].append(-1)
                    continue

                # 代表クラスタの特定
                cluster_max_size = clusters[0]
                for cluster in clusters:
                    if len(cluster_max_size) < len(cluster):
                        cluster_max_size = cluster
                most_clusterses.append(cluster_max_size)
                clusterses_size.append(len(cluster_max_size))
                most_clusters_ratio.append(len(cluster_max_size) / len(nouns2))
                clusters_size.append(len(clusters))
                topic_segment_data['most_cluster'].append(cluster_max_size)
                topic_segment_data['most_cluster_ratio'].append(len(cluster_max_size) / len(nouns2))
                topic_segment_data['clusters_size'].append(len(clusters))

                # 代表クラスタの名詞の全体的なコサイン類似度
                if len(cluster_max_size) == 0 or len(cluster_max_size) == 1:
                    clusterses_average_vector.append(0)
                else:
                    clusterses_average_vector.append(cluster_average_similarity(cluster_max_size))
                    # clusterses_average_vector.append(cluster_average_similarity(removed))
                
                ts_most_cluster_avector = average_embedding(cluster_max_size)

                topic_segment_data['represent_matching_degree'].append(cosine_similarity([ts_most_cluster_avector], [td_most_vector])[0][0])
            


            if len(clusterses_size) == 0:
                average_cluster_size.append(0)
            else:
                average_cluster_size.append(sum(clusterses_size) / len(clusterses_size))
            if len(clusterses_average_vector) == 0:
                average_similarities.append(0)
            else:
                average_similarities.append(sum(clusterses_average_vector) / len(clusterses_average_vector))
            
            if len(most_clusters_ratio) == 0:
                average_ratio = 0
            else:
                average_ratio = sum(most_clusters_ratio) / len(most_clusters_ratio)
            if len(clusters_size) == 0:
                average_size = 0
                segment_score.append(0)
            else:
                average_size = sum(clusters_size) / len(clusters_size)
                segment_score.append(average_ratio / average_size)


    # fig, ax1 = plt.subplots(figsize=(15, 9))
    # # 左軸にプロット
    # ax1.plot(Y, segment_score, label='Segment Score', marker='o', color='b')
    # ax1.set_xlabel(f'{Y_label}')
    # ax1.set_ylabel('Segment Score', color='b')
    # ax1.tick_params(axis='y', labelcolor='b')

    # # 右軸を作成
    # # ax2 = ax1.twinx()
    # # ax2.plot(clustersize_thresholds, average_cluster_size, label='Average Cluster Size', marker='x', color='r')
    # # ax2.set_ylabel('Average Cluster Size', color='r')
    # # ax2.tick_params(axis='y', labelcolor='r')

    # # グリッドとレイアウト調整
    # plt.grid(True)
    # fig.tight_layout()

    # plt.show()

    topic_segment_df = pd.DataFrame(topic_segment_data)
    topic_segment_df.to_csv(f'./././Data/test_data/final_presentation/df_adjusted_{id_clustersize_similarity}/topic_segments_df_new.csv', index = False, encoding = 'utf-8')

    topic_segment_df = pd.read_csv(f'./././Data/test_data/final_presentation/df_adjusted_{id_clustersize_similarity}/topic_segments_df_new.csv')
    print(f"トピックセグメントの数: {len(topic_segment_df)}")
    topic_segments_size = 25
    adjusted_data = {'threshold': [], 'adjusted_represent': [], 'adjusted_average': []}
    print("トピックセグメントで現れる名詞をクラスタリングして取得した代表的なクラスタのベクトルと")
    print("トピックセグメントに付与されたトピックの代表的なベクトルの類似度が、\n")

    print("トピックセグメントの代表ベクトルとトピック辞書の代表ベクトルのコサイン類似度")
    rdegrees = list(map(float, topic_segment_df['represent_matching_degree'].tolist()))
    rdegrees.sort()
    q1 = np.percentile(rdegrees, 25)
    q2 = np.percentile(rdegrees, 50)
    q3 = np.percentile(rdegrees, 75)
    range_1 = [x for x in rdegrees if x < q1][:topic_segments_size]
    range_2 = [x for x in rdegrees if q1 <= x < q2][:topic_segments_size]
    range_3 = [x for x in rdegrees if q2 <= x < q3][:topic_segments_size]
    range_4 = [x for x in rdegrees if x >= q3][:topic_segments_size]
    rdegrees = range_1 + range_2 + range_3 + range_4
    for i in range(1, 21):
        count = 0
        adjusted_data['threshold'].append(i/20)
        for degree in rdegrees:
            if float(degree) >= i/20:
                count += 1
        adjusted_data['adjusted_represent'].append(count / len(rdegrees))
        print(f"{i/20}以上の割合: {count / len(rdegrees):.05f}")
    print("\n")
    
    print("トピックセグメントの平均ベクトルのトピック辞書の平均ベクトルのコサイン類似度")
    adegrees = list(map(float, topic_segment_df['average_matching_degree'].tolist()))
    adegrees.sort()
    q1 = np.percentile(adegrees, 25)
    q2 = np.percentile(adegrees, 50)
    q3 = np.percentile(adegrees, 75)
    range_1 = [x for x in adegrees if x < q1][:topic_segments_size]
    range_2 = [x for x in adegrees if q1 <= x < q2][:topic_segments_size]
    range_3 = [x for x in adegrees if q2 <= x < q3][:topic_segments_size]
    range_4 = [x for x in adegrees if x >= q3][:topic_segments_size]
    adegrees = range_1 + range_2 + range_3 + range_4
    for i in range(1, 21):
        count = 0
        for degree in adegrees:
            if float(degree) >= i/20:
                count += 1
        adjusted_data['adjusted_average'].append(count / len(adegrees))
        print(f"{i/20}以上の割合: {count / len(adegrees):.05f}")
    print("\n")

    adjusted_df = pd.DataFrame(adjusted_data)
    adjusted_df.to_csv(f'./././Data/test_data/final_presentation/df_adjusted_{id_clustersize_similarity}/adjusted_results.csv', index = False, encoding = 'utf-8')
    print("Adjusted")
    print(f"Similaritiy Threshold: {similarity_threshold}")
    print(f"Cluster size: {clustersize_threshold}")

    sys.exit()