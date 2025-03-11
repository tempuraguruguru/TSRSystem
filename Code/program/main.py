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

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def embed(text): # 384次元
    word_embeddings = model.encode(text)
    return word_embeddings

def embed2(word):
    inputs = tokenizer(word, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = bert_model(**inputs)
    last_hidden_state = outputs.last_hidden_state
    sample_vector = last_hidden_state[0][1].detach().cpu().numpy()
    return sample_vector

def average_embedding(words):
    # 各単語をベクトル化してリストに追加
    vectors = [embed2(word) for word in words]
    
    # ベクトルの平均を計算
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

def jaccard(nouns1, nouns2):
    if not nouns1 or not nouns2:
        return 0
    nouns1_set = set(nouns1)
    nouns2_set = set(nouns2)
    intersection = nouns1_set & nouns2_set
    union = nouns1_set | nouns2_set
    similarity = len(intersection) / len(union)
    return similarity

def jaccard_nouns_similarity(nouns1, nouns2, threshold):
    if not nouns1 or not nouns2:
        return 0
    nouns1 = set(nouns1)
    nouns2 = set(nouns2)
    total = 0
    count = 0
    for noun1 in nouns1:
        for noun2 in nouns2:
            total += 1
            if cosine_similarity([embed2(noun1)], [embed2(noun2)])[0][0] >= threshold: # 名詞間の類似度がどれくらいの時に名詞が類似しているとするか
                count += 1
    similarity = count / total
    return similarity

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

# def calculate_topic_distribution_for_text(text, topic_data):
#     """
#     文書のトピック分布を計算する関数
#     :param text: 対象の文書テキスト
#     :param topic_data: 各トピックの名詞とその確率を含むデータ
#     :return: トピック分布（確率）
#     """
#     nouns_in_text = extract_nouns2(text)
#     print(nouns_in_text)

#     # トピック分布を初期化
#     num_topics = len(topic_data)
#     topic_distribution = np.zeros(num_topics)

#     # 各トピックに関連する名詞とその確率を取得
#     for idx, row in topic_data.iterrows():
#         topic_nouns = row['nouns'] # トピックの関連名詞
#         topic_probabilities = row['probabilities'] # トピックの関連名詞の生起確率
#         # 文書内の各名詞が関連するトピックに対して寄与する確率を加算
#         for noun, prob in zip(topic_nouns, topic_probabilities):
#             # トピック分布を割り当てる文書にトピックの関連名詞が含まれているか？
#             if noun in nouns_in_text: # 文書に関連名詞が含まれていたら、そのトピックの生起確率にその関連名詞の生起確率を加算
#                 topic_distribution[idx] += prob

#     # 正規化して確率分布にする
#     if topic_distribution.sum() > 0:
#         topic_distribution /= topic_distribution.sum()
#     topic_distribution = [(i, val) for i, val in enumerate(topic_distribution) if val != 0]

#     return topic_distribution

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
    n_gram_list = [38]
    num_topics_list = [200]
    probability_thresholds = [0.0407]
    similarity_thresholds = [0.6685]
    clustersize_thresholds = [2]

    X = similarity_thresholds
    X_label = "Similarity"

    average_similarities = []
    average_cluster_size = []
    average_ratio_td = []
    topic_dictionnary_size = []
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
            similarity_threshold = 0.8
            clustersize_threshold = 2
            topics_raw = {'id': [], 'nouns': [], 'probabilities': []}
            topics_clustered = {'id': [], 'nouns': [], 'probabilities': []}
            for topic_id, topic_terms in topics_previous:
                nouns = [word for word, probability in topic_terms if probability >= probability_threshold]
                probabilities = [probability for word, probability in topic_terms if probability >= probability_threshold]
                topics_raw['id'].append(topic_id)
                topics_raw['nouns'].append(nouns)
                topics_raw['probabilities'].append(probabilities)
                clusters = clustering_topic_dictionary2(nouns, probabilities, probability_threshold, similarity_threshold)
                clusters2 = [cluster for cluster in clusters if len(cluster) >= clustersize_threshold]
                for cluster in clusters2:
                    topics_new.append(cluster)
            for i, topic_new in enumerate(topics_new):
                nouns = [word for word, probability in topic_new]
                probabilities = [probability for word, probability in topic_new]
                topics_clustered['id'].append(i)
                topics_clustered['nouns'].append(nouns)
                topics_clustered['probabilities'].append(probabilities)

            # トピック辞書を作成
            df_raw = pd.DataFrame(topics_raw)
            df_raw.to_csv('./././Data/test_data/final_presentation/df_raw.csv', index = False, encoding = 'utf-8')
            df_clustered = pd.DataFrame(topics_clustered)
            df_clustered.to_csv('./././Data/test_data/final_presentation/df_clustered.csv', index = False, encoding = 'utf-8')

            # sys.exit()
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



            # 分割文書に対してトピック分布を計算
            # sub_texts = []
            # for subdocs in subdocuments:
            #     sub_nouns = []
            #     for subdoc in subdocs: # subdocはn-gram
            #         sub_nouns.append(extract_nouns2(subdoc))
            #     sub_texts.append(sub_nouns)

            # sub_corpuses = []
            # for sub_text in sub_texts:
            #     sub_corpus = []
            #     for sub_nouns in sub_text:
            #         sub_corpus.append(dictionary.doc2bow(sub_nouns))
            #     sub_corpuses.append(sub_corpus)

            # doc_topic_distributions = []
            # for sub_corpus in sub_corpuses:
            #     sub_doc_topic_distributions = []
            #     for sub_doc_bows in sub_corpus:
            #         sub_doc_topics = lda_model.get_document_topics(sub_doc_bows, minimum_probability = 0)
            #         sub_doc_topic_distributions.append(sub_doc_topics)
            #     doc_topic_distributions.append(sub_doc_topic_distributions)



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



            # 存在トピック分布を取得
            # has_topics = set(has_topics)
            # has_topics = list(has_topics)
            # topic_num = 0

            # for t in has_topics:
            #     topic_num += 1
            #     topics = lda_model.show_topic(t, topn = 10)
            #     nouns = []
            #     probabilities = []
            #     for word, prob in topics:
            #         nouns.append(word)
            #         probabilities.append(prob)
            #     All_Topics_Info['topic'].append(t)
            #     All_Topics_Info['nouns'].append(nouns)
            #     All_Topics_Info['probabilities'].append(probabilities)

            # word_distribution = pd.DataFrame(All_Topics_Info)
            # word_distribution.to_csv(f'./././Data/test_data/final_presentation/word_distribution.csv', index = False, encoding = 'utf-8')





            # トピック辞書をそのまま用いる場合
            # topic_dictionary = pd.read_csv('./././Data/test_data/final_presentation/word_distribution.csv')
            # ids = topic_dictionary['topic'].tolist()
            # nounses = topic_dictionary['nouns'].tolist()
            # probabilitieses = topic_dictionary['probabilities'].tolist()
            # for probability_threshold in probability_thresholds:
            #     for similarity_threshold in similarity_thresholds:
            #         for clustersize_threshold in clustersize_thresholds:
            #             clusterses = []
            #             most_clusterses = []
            #             clusterses_size = []
            #             clusterses_average_vector = []
            #             clusters_ratio = []
            #             removed_nouns = []

            #             for nouns, probabilities in zip(nounses, probabilitieses):
            #                 removed = []
            #                 for noun, probability in zip(eval(nouns), eval(probabilities)):
            #                     if probability > probability_threshold:
            #                         removed.append(noun)
            #                 removed_nouns.append(removed)

            #                 clusters = clustering_topic_dictionary(eval(nouns), eval(probabilities), probability_threshold, similarity_threshold)
            #                 clusterses.append(clusters)
            #                 clusters = [cluster for cluster in clusters if len(cluster) >= clustersize_threshold]

            #                 # クラスタがない
            #                 if len(clusters) == 0:
            #                     clusterses_average_vector.append(0)
            #                     most_clusterses.append([])
            #                     clusters_ratio.append(0)
            #                     continue

            #                 # 代表クラスタの特定
            #                 cluster_max_size = clusters[0]
            #                 for cluster in clusters:
            #                     if len(cluster_max_size) < len(cluster):
            #                         cluster_max_size = cluster
            #                 most_clusterses.append(cluster_max_size)
            #                 clusterses_size.append(len(cluster_max_size))

            #                 # 代表クラスタの名詞の全体的なコサイン類似度
            #                 if len(cluster_max_size) == 1:
            #                     clusterses_average_vector.append(1)
            #                     clusters_ratio.append(1)
            #                 else:
            #                     clusterses_average_vector.append(cluster_average_similarity(cluster_max_size))
            #                     # clusterses_average_vector.append(cluster_average_similarity(removed))
            #                     clusters_ratio.append(len(cluster_max_size) / len(removed))

            #             if len(clusterses_size) == 0:
            #                 average_cluster_size.append(0)
            #             else:
            #                 average_cluster_size.append(sum(clusterses_size) / len(clusterses_size))
            #             if len(clusterses_average_vector) == 0:
            #                 average_similarities.append(0)
            #             else:
            #                 average_similarities.append(sum(clusterses_average_vector) / len(clusterses_average_vector))

            #             topic_dictionnary_size.append(len(clusterses_average_vector))
            #             average_ratio_td.append(sum(clusters_ratio) / len(clusters_ratio))


            # トピック辞書を調整して用いる場合
            topic_dictionary = pd.read_csv('./././Data/test_data/final_presentation/df_clustered.csv')
            ids = topic_dictionary['id'].tolist()
            nounses = topic_dictionary['nouns'].tolist()
            probabilitieses = topic_dictionary['probabilities'].tolist()
            for probability_threshold in probability_thresholds:
                for similarity_threshold in similarity_thresholds:
                    for clustersize_threshold in clustersize_thresholds:
                        probability_threshold = 1e-4
                        similarity_threshold = 0.8
                        clustersize_threshold = 2

                        clusterses = []
                        most_clusterses = []
                        clusterses_size = []
                        clusterses_average_vector = []
                        clusters_ratio = []
                        removed_nouns = []

                        for nouns, probabilities in zip(nounses, probabilitieses):
                            removed = []
                            for noun, probability in zip(eval(nouns), eval(probabilities)):
                                if probability > probability_threshold:
                                    removed.append(noun)
                            removed_nouns.append(removed)

                            clusters = clustering_topic_dictionary(eval(nouns), eval(probabilities), probability_threshold, similarity_threshold)
                            clusterses.append(clusters)
                            clusters = [cluster for cluster in clusters if len(cluster) >= clustersize_threshold]

                            # クラスタがない
                            if len(clusters) == 0:
                                clusterses_average_vector.append(0)
                                most_clusterses.append([])
                                clusters_ratio.append(0)
                                continue

                            # 代表クラスタの特定
                            cluster_max_size = clusters[0]
                            for cluster in clusters:
                                if len(cluster_max_size) < len(cluster):
                                    cluster_max_size = cluster
                            most_clusterses.append(cluster_max_size)
                            clusterses_size.append(len(cluster_max_size))

                            # 代表クラスタの名詞の全体的なコサイン類似度
                            if len(cluster_max_size) == 1:
                                clusterses_average_vector.append(1)
                                clusters_ratio.append(1)
                            else:
                                clusterses_average_vector.append(cluster_average_similarity(cluster_max_size))
                                # clusterses_average_vector.append(cluster_average_similarity(removed))
                                clusters_ratio.append(len(cluster_max_size) / len(removed))

                        if len(clusterses_size) == 0:
                            average_cluster_size.append(0)
                        else:
                            average_cluster_size.append(sum(clusterses_size) / len(clusterses_size))
                        if len(clusterses_average_vector) == 0:
                            average_similarities.append(0)
                        else:
                            average_similarities.append(sum(clusterses_average_vector) / len(clusterses_average_vector))

                        topic_dictionnary_size.append(len(clusterses_average_vector))
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

    data = {'topic': ids, 'nouns': nounses, 'probabilities': probabilitieses, 'removed_nouns': removed_nouns, 'most_clusters': most_clusterses, 'ratio': clusters_ratio, 'clusterses_average_similarity': clusterses_average_vector, 'clusters': clusterses}
    clusters_df = pd.DataFrame(data)
    clusters_df.to_csv('./././Data/test_data/final_presentation/word_distribution_clusters.csv', index = False, encoding = 'utf-8')

    # sys.exit()

    ratios = clusters_df['ratio'].tolist()
    for i in range(1, 11):
        count = 0
        for ratio in ratios:
            if ratio >= i/10:
                count += 1
        print(f"代表クラスタが占める割合が{i/10}以上: {count / len(ratios)}")


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
    # トピックセグメントの情報
    topic_segment_data = {'file_path': [], 'transcription': [], 'transcription_timestamp': [], 'transcription_nouns': [], 'transcription_nouns_frequency': [], 'start': [], 'end': [], 'topic_distribution': [], 'topic': [], 'topic_probability': []}
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
    df.to_csv(f'./././Data/test_data/final_presentation/topic_segments_info.csv', index = False, encoding = 'utf-8')

    # トピックセグメントのcsv
    segment_df = pd.DataFrame(topic_segment_data)
    ids = [id for id in range(len(segment_df))]
    ids = {'id': ids}
    ids = pd.DataFrame(ids)
    segment_df = pd.concat([segment_df, ids], axis = 1)
    # segment_df.to_csv('./././Data/test_data/final_presentation/segment_df.csv', index = False, encoding = 'utf-8')
    df = pd.merge(segment_df, clusters_df, on = 'topic', how = 'inner')
    df.to_csv(f'./././Data/test_data/final_presentation/topic_segments_df.csv', index = False, encoding = 'utf-8')


    # similarity_threshold = 0.6685, clustersize_threhold = 2
    similarity_thresholds = [0.8]
    clustersize_thresholds = [2]

    Y = similarity_thresholds
    Y_label = "Similarity"

    average_similarities = []
    average_cluster_size = []

    segment_score = []

    ids = df['id'].tolist()
    file_paths = df['file_path'].tolist()
    transcriptions = df['transcription'].tolist()
    td_most_clusters = df['most_clusters'].tolist()
    for similarity_threshold in similarity_thresholds:
        for clustersize_threshold in clustersize_thresholds:
            probability_threshold = 1e-4
            similarity_threshold = 0.8
            clustersize_threshold = 2

            clusterses = []
            most_clusterses = []
            clusterses_size = []
            most_clusters_ratio = [] #ratio
            clusters_size = [] # cluster_size
            clusterses_average_vector = []
            topic_segment_data = {'file_path': file_paths, 'id': ids, 'transcription': transcription, 'transcription_nouns': [], 'clusters': [], 'most_cluster': [], 'most_cluster_ratio': [], 'clusters_size': [], 'matching_degree': []}

            for transcription, td_most_cluster in zip(transcriptions, td_most_clusters):
                nouns = extract_nouns2(transcription)
                nouns = list(set(nouns))
                topic_segment_data['transcription_nouns'].append(nouns)
                probabilities = [0 for _ in range(len(nouns))]
                probability_threshold = -1

                clusters = clustering_topic_dictionary(nouns, probabilities, probability_threshold, similarity_threshold)
                clusterses.append(clusters)
                topic_segment_data['clusters'].append(clusters)
                clusters = [cluster for cluster in clusters if len(cluster) >= clustersize_threshold]

                # クラスタがない
                if len(clusters) == 0:
                    clusterses_average_vector.append(0)
                    topic_segment_data['most_cluster'].append(0)
                    topic_segment_data['most_cluster_ratio'].append(0)
                    topic_segment_data['clusters_size'].append(0)
                    topic_segment_data['matching_degree'].append(0)
                    continue
                            
                # 代表クラスタの特定
                cluster_max_size = clusters[0]
                for cluster in clusters:
                    if len(cluster_max_size) < len(cluster):
                        cluster_max_size = cluster
                most_clusterses.append(cluster_max_size)
                clusterses_size.append(len(cluster_max_size))
                most_clusters_ratio.append(len(cluster_max_size) / len(nouns))
                clusters_size.append(len(clusters))
                topic_segment_data['most_cluster'].append(cluster_max_size)
                topic_segment_data['most_cluster_ratio'].append(len(cluster_max_size) / len(nouns))
                topic_segment_data['clusters_size'].append(len(clusters))

                # 代表クラスタの名詞の全体的なコサイン類似度
                if len(cluster_max_size) == 0 or len(cluster_max_size) == 1:
                    clusterses_average_vector.append(0)
                else:
                    clusterses_average_vector.append(cluster_average_similarity(cluster_max_size))
                    # clusterses_average_vector.append(cluster_average_similarity(removed))
                ts_most_cluster_avector = average_embedding(cluster_max_size)
                td_most_cluster_avector = average_embedding(td_most_cluster)
                topic_segment_data['matching_degree'].append(cosine_similarity([ts_most_cluster_avector], [td_most_cluster_avector])[0][0])


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


    fig, ax1 = plt.subplots(figsize=(15, 9))
    # 左軸にプロット
    ax1.plot(Y, segment_score, label='Segment Score', marker='o', color='b')
    ax1.set_xlabel(f'{Y_label}')
    ax1.set_ylabel('Segment Score', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # 右軸を作成
    # ax2 = ax1.twinx()
    # ax2.plot(clustersize_thresholds, average_cluster_size, label='Average Cluster Size', marker='x', color='r')
    # ax2.set_ylabel('Average Cluster Size', color='r')
    # ax2.tick_params(axis='y', labelcolor='r')

    # グリッドとレイアウト調整
    plt.grid(True)
    fig.tight_layout()

    plt.show()

    topic_segment_df = pd.DataFrame(topic_segment_data)
    topic_segment_df.to_csv(f'./././Data/test_data/final_presentation/topic_segments_df_new.csv', index = False, encoding = 'utf-8')

    degrees = topic_segment_df['matching_degree'].tolist()
    count = 0
    for degree in degrees:
        if degree >= 0.8:
            count += 1
    print(f"トピックセグメントで現れる名詞をクラスタリングして取得した代表的なクラスタのベクトル、トピックセグメントに付与されたトピックの代表的なベクトルの類似度が0.8以上の割合: {count / len(degrees)}")


    sys.exit()































    print(os.getcwd())
    os.chdir("../../")
    print(os.getcwd())

    mid_path = './././Data/test_data/final_presentation'
    mid_path = os.path.abspath(mid_path)
    print(mid_path)

    alpha = 1
    # トピック辞書
    topic_and_words = pd.read_csv(f'./././Data/test_data/final_presentation/word_distribution_{N_gram}_{Number_of_Topics}.csv')
    topics = topic_and_words['topic'].tolist()
    nounses = topic_and_words['nouns'].tolist()
    probabilitieses2 = topic_and_words['probabilities'].tolist()

    # トピックセグメントデータベース
    utterance_audio_df = pd.read_csv(f'./././Data/test_data/final_presentation/segment_df_{N_gram}_{Number_of_Topics}.csv')
    # similarities = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.1]
    similarities = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

    # similarities = [0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085]
    # similarities = [0.05, 0.051, 0.052, 0.053, 0.054, 0.055, 0.056, 0.057, 0.058, 0.059, 0.06, 0.061, 0.062, 0.063, 0.064, 0.065, 0.066, 0.067, 0.068, 0.069, 0.07, 0.071, 0.072, 0.073, 0.074, 0.075, 0.076, 0.077, 0.078, 0.079, 0.08, 0.081, 0.082, 0.083, 0.084, 0.085]

    # similarities = [0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08]
    similarities = [0.055, 0.056, 0.057, 0.058, 0.059, 0.06, 0.061, 0.062, 0.063, 0.064, 0.065, 0.066, 0.067, 0.068, 0.069, 0.07]

    nouns_similarities = []
    similarity_topics = []

    for similarity in similarities:
        SIMILARITY_TOPIC_THRESHOLD = similarity # 類似トピックと見なすときの閾値
        WORD_SIMILARITY_THRESHOLD = 0.8 # 名詞間の類似度がどれくらいの時に、2つの名詞が似ているとするか

        similarity_s = str(similarities[0])
        similarity_e = str(similarities[-1])
        similarity_s = similarity_s.replace(".", "")
        similarity_e = similarity_e.replace(".", "")

        print(similarity)
        user_nouns = extract_nouns2(sys.argv[1])
        user_interests = [embed2(noun) for noun in user_nouns] # ユーザーが聴きたい内容について明示的に表した文章から名詞を抽出
        user_interest = embed2(sys.argv[1])
        history = eval(sys.argv[2])
        index = int(sys.argv[3])

        CurrentDir = f'experiment_{N_gram}_{Number_of_Topics}'
        if os.path.exists(f'./././Data/experiments7/{CurrentDir}'):
            shutil.rmtree(f'./././Data/experiments7/{CurrentDir}')
        os.makedirs(f'./././Data/experiments7/{CurrentDir}')

        # 入力キーワードと最も類似しているトピックを特定、そのトピックについて話している発話音声データの一覧を取得
        similarity_df = {'topic': [], 'similarity': []}
        for topic, nouns, probabilities2 in zip(topics, nounses, probabilitieses2):
            t_similarity = 0
            t_similarities = []

            for user_interest in user_interests: # それぞれの名詞を取得
                for noun, probabilitity2 in zip(eval(nouns), eval(probabilities2)):
                    t_s = cosine_similarity([user_interest], [embed2(noun)])[0][0] # 名詞とトピック辞書の各関連名詞とのコサイン類似度を計算
                    t_similarities.append(t_s * probabilitity2) # コサイン類似度に関連名詞の生起確率を乗算
                if len(t_similarities) == 0:
                    t_similarity += 0
                else:
                    t_similarity += sum(t_similarities) / len(t_similarities) # この名詞に対するトピック類似度を計算
            t_similarity = t_similarity / len(user_interests) # 全ての名詞に対するトピック類似度の平均を計算


            # # Jaccard_nouns_similarity関数
            # total = 0
            # count = 0
            # for noun1 in user_interests:
            #     for noun, probabilitity2 in zip(eval(nouns), eval(probabilities2)):
            #         total += 1
            #         t_s = cosine_similarity([noun1], [embed2(noun)])
            #         if t_s >= WORD_SIMILARITY_THRESHOLD: # 2つの名詞が似ていて
            #             if probabilitity2 >= 1e-4: # その名詞の生起確率が1e-4より高い
            #                 count += 1
            # t_similarity = count / total


            # 入力文に対するトピックの類似度
            similarity_df['topic'].append(topic)
            similarity_df['similarity'].append(t_similarity)

        # ユーザーが入力した文とトピック辞書からユーザーの興味に類似したトピックを特定
        topic_similarities = pd.DataFrame(similarity_df)
        topic_similarities = topic_similarities.sort_values(by = 'similarity', ascending = False)
        topic_similarities = pd.merge(topic_similarities, topic_and_words, on = 'topic', how = 'inner')
        if os.path.exists(f'./././Data/experiments7/{CurrentDir}/topic_similarities.csv'):
            os.remove(f'./././Data/experiments7/{CurrentDir}/topic_similarities.csv')
        # topic_similarities.to_csv(f'./././Data/experiments7/{CurrentDir}/topic_similarities.csv', index = False, encoding = 'utf-8')

        # 類似度の閾値がどのような値のときに、トピックを類似トピックとするか
        # TOPIC_THRESHOLD = similarity
        # similarity_topic_df = topic_similarities[topic_similarities['similarity'] >= TOPIC_THRESHOLD]
        similarity_topic_df = topic_similarities[topic_similarities['similarity'] >= SIMILARITY_TOPIC_THRESHOLD]
        selected_topics = similarity_topic_df
        # selected_topics.to_csv(f'./././Data/experiments7/{CurrentDir}/selected_topics.csv', index = False, encoding = 'utf-8')
        selected_topic_list = selected_topics['topic'].tolist()
        transcriptions = utterance_audio_df['transcription'].tolist()
        transcriptions_timestamp = utterance_audio_df['transcription_timestamp'].tolist()

        topic_sequences = []
        topic_start_times = []
        topic_file_paths = utterance_audio_df['file_path'].tolist()
        topic_ids = utterance_audio_df['id'].tolist()
        for transcription_timestamp in transcriptions_timestamp:
            # 文字起こしをtri-gramにする
            texts = transcription_timestamp.split('\n')
            trigrams = []
            trigrams_timestamp = []
            for i in range(len(texts) - 2):
                text = ''
                text_time = 0
                for j in range(3):
                    text_timestamp = texts[i + j].split('::')
                    if j == 0:
                        text_time = eval(text_timestamp[0])[0]
                    text += texts[i + j] + ' '
                trigrams.append(text.strip())
                trigrams_timestamp.append(text_time)

            # トピックの流れを推定
            topic_sequence = []
            topic_start_time = []
            for trigram, trigram_timestamp in zip(trigrams, trigrams_timestamp):
                trigram_nouns = extract_nouns2(trigram)
                trigram_bow = dictionary.doc2bow(trigram_nouns)
                topic_distribution = lda_model.get_document_topics(trigram_bow, minimum_probability=0.0)
                most_topic = max(topic_distribution, key=lambda x: x[1])
                topic_sequence.append(most_topic[0])
                topic_start_time.append(trigram_timestamp)

            if topic_sequence:
                L1_new = [topic_sequence[0]]
                L2_new = [topic_start_time[0]]
                current_value = topic_sequence[0]
                for i in range(1, len(topic_sequence)):
                    if topic_sequence[i] != current_value:
                        current_value = topic_sequence[i]
                        L1_new.append(current_value)
                        L2_new.append(topic_start_time[i])
                topic_sequence = L1_new
                topic_start_time = L2_new

            # トピックの流れと発話開始時間を取得
            topic_sequences.append(topic_sequence)
            topic_start_times.append(topic_start_time)
        topic_sequence_data = {'id': topic_ids, 'file_path': topic_file_paths, 'topic_sequence': topic_sequences, 'topic_start_times': topic_start_times}
        topic_sequence_df = pd.DataFrame(topic_sequence_data)
        utterance_audio_df2 = pd.merge(utterance_audio_df, topic_sequence_df, on = ['id', 'file_path'], how = 'inner')
        utterance_audio_df2.to_csv('./././Data/test_data/final_presentation/utterance_audio_df.csv', index = False, encoding = 'utf-8')

        # 類似トピックを含んでいるトピックセグメントのみを取得
        df = utterance_audio_df2[utterance_audio_df2['topic_sequence'].apply(lambda lst: any(topic in lst for topic in selected_topic_list))]
        df['time'] = df['end'] - df['start']
        rows = ['file_path', 'transcription', 'transcription_timestamp',
                'transcription_nouns', 'transcription_nouns_frequency', 'start', 'end',
                'topic_distribution', 'entropy', 'max_entropy', 'topic',
                'topic_probability', 'id', 'nouns', 'probabilities', 'topic_sequence',
                'topic_start_times']

        # 類似トピック以外の部分を除去したトピックセグメントの作成
        topic_ids = df['id'].tolist()
        file_paths = df['file_path'].tolist()
        transcriptions = df['transcription'].tolist()
        transcriptions_timestamp = df['transcription_timestamp'].tolist()
        topic_sequenceses = df['topic_sequence'].tolist()
        topic_start_timeses = df['topic_start_times'].tolist()
        topic_segment_data = {
            'id': [], 'file_path': [],
            'transcription': [], 'transcription_timestamp': [],
            'transcription_nouns': [], 'user_nouns_and_transcription_nouns_similarity': [],
            'topic_sequence': [], 'topic_start_times': [],
        }
        for topic_id, file_path, transcription, transcription_timestamp, topic_sequences, topic_start_times in zip(topic_ids, file_paths, transcriptions, transcriptions_timestamp, topic_sequenceses, topic_start_timeses):
            # 文字列を辞書に変換
            timestamps_and_texts = {}
            for line in transcription_timestamp.split('\n'):
                if len(line) == 0:
                    continue
                lines = line.split('::')
                timestamp = lines[0]
                text = lines[1]
                start = eval(timestamp)[0]
                timestamps_and_texts[float(start)] = text

            # トピックセグメントのトピックの流れと発話時間を取得
            utterance = ''
            for topic_sequence, start_time, end_time in zip(topic_sequences, topic_start_times[:-1], topic_start_times[1:]):
                if topic_sequence in selected_topic_list:
                    # 該当する時間帯のテキストを追加
                    for timestamp, text in timestamps_and_texts.items():
                        if start_time <= timestamp < end_time:
                            utterance += text + '\n'
            utterance.rstrip('\n')

            # トピックセグメントの文字起こしテキストデータ
            topic_segment_nouns = extract_nouns2(utterance)

            # トピックセグメントデータに追加
            topic_segment_data['id'].append(topic_id)
            topic_segment_data['file_path'].append(file_path)
            topic_segment_data['transcription'].append(utterance.strip())
            topic_segment_data['transcription_timestamp'].append(f'{start_time}::{end_time}')
            topic_segment_data['transcription_nouns'].append(topic_segment_nouns)
            # topic_segment_data['user_nouns_and_transcription_nouns_similarity'].append(nouns_similarity(user_nouns, topic_segment_nouns))
            topic_segment_data['user_nouns_and_transcription_nouns_similarity'].append(jaccard_nouns_similarity(user_nouns, topic_segment_nouns, WORD_SIMILARITY_THRESHOLD))
            topic_segment_data['topic_sequence'].append(topic_sequence)
            topic_segment_data['topic_start_times'].append(start_time)

        df = pd.DataFrame(topic_segment_data)
        df['file_path'] = df['file_path'].astype(str).str.replace('\\', '/', regex = False)
        df.to_csv('./././Data/test_data/final_presentation/topic_segments_df.csv', index = False, encoding = 'utf-8')

        user_nouns_and_transcription_nouns_similarity = topic_segment_data['user_nouns_and_transcription_nouns_similarity']
        if len(user_nouns_and_transcription_nouns_similarity) != 0:
            nouns_similarities.append(sum(user_nouns_and_transcription_nouns_similarity) / len(user_nouns_and_transcription_nouns_similarity))
        else:
            nouns_similarities.append(0)
        similarity_topics.append(len(selected_topic_list))

    plot_data = {'similarity': similarities, 'nouns_similarities': nouns_similarities, 'similarity_topics': similarity_topics}
    plot_df = pd.DataFrame(plot_data)
    os.makedirs("./././Data/test_data/final_presentation/img2", exist_ok = True)
    os.makedirs("./././Data/test_data/final_presentation/img2/plot_data", exist_ok = True)
    plot_df.to_csv(f"./././Data/test_data/final_presentation/img2/plot_data/plot_data_{similarity_s}_to_{similarity_e}.csv", index = False, encoding = 'utf-8')

    # グラフの描画
    fig, ax1 = plt.subplots(figsize=(15, 9))

    # 左側の縦軸 (nouns_similarities)
    ax1.plot(similarities, nouns_similarities, label='Nouns Similarities', marker='o', color='b')
    ax1.set_xlabel('Similarities')
    ax1.set_ylabel('Nouns Similarities', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # 右側の縦軸 (similarity_topics)
    ax2 = ax1.twinx()
    ax2.plot(similarities, similarity_topics, label='Similarity Topics', marker='x', color='r')
    ax2.set_ylabel('Similarity Topics', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # グラフの装飾
    fig.tight_layout()
    plt.title('Similarities vs Nouns Similarities and Similarity Topics')
    plt.grid(True)

    # グラフの保存
    save_path = f"./././Data/test_data/final_presentation/img2/plot_data_{similarity_s}_to_{similarity_e}.png"  # 保存するフォルダとファイル名を指定
    plt.savefig(save_path)

    # グラフの表示
    plt.show()

        # for remove_path in history:
        #     df = df[df['file_path'] != remove_path]

        # os.makedirs(f'./././Data/experiments7/{CurrentDir}/high_probability', exist_ok=True)
        # # os.makedirs(f'./././Data/experiments3/{sys.argv[1]}/short_time', exist_ok=True)

        # for current_dir, dirs, files in os.walk('./././Django_Server/myproject/static/audio/'):
        #     dir_name = os.path.basename(current_dir)
        #     if dir_name == 'high_probability':# or dir_name == 'short_time':
        #         for file in files:
        #             if file.endswith('.mp3'):
        #                 os.remove(os.path.join(current_dir, file))


        # トピックセグメントのトピック生起確率が高い順
        # high_probability_df = df.sort_values(by = ['topic_probability', 'play_time'], ascending = [False, True])
        # if os.path.exists('./././Data/test_data/final_presentation/high_probability_df_original.csv'):
        #     os.remove('./././Data/test_data/final_presentation/high_probability_df_original.csv')
        # high_probability_df.to_csv('./././Data/test_data/final_presentation/high_probability_df_original.csv', index = False, encoding = 'utf-8')

        # file_paths = high_probability_df['file_path'].tolist()
        # stimes = high_probability_df['stime'].tolist()
        # etimes = high_probability_df['etime'].tolist()
        # for i in range(len(file_paths)):
        #     history.append(file_paths[i])
        #     audio = AudioSegment.from_file(file_paths[i])
        #     # audio = audio[float(stimes[i])*1000 : float(etimes[i])*1000]
        #     audio = audio[float(stimes[i])*1000:]
        #     audio.export(f'./././Django_Server/myproject/static/audio/high_probability/segment_{i+1}.mp3', format = 'mp3')
        #     audio.export(f'./././Data/experiments7/{CurrentDir}/high_probability/segment_{i+1}.mp3', format = 'mp3')
        # high_probability_df2 = pd.merge(high_probability_df, topic_and_words, left_on = 'topics', right_on = 'topic', how = 'inner')

        # if os.path.exists('./././Data/test_data/final_presentation/high_probability_df.csv'):
        #     os.remove('./././Data/test_data/final_presentation/high_probability_df.csv')
        # high_probability_df.to_csv('./././Data/test_data/final_presentation/high_probability_df.csv', index = False, encoding = 'utf-8')
        # if os.path.exists(f'./././Data/experiments7/{CurrentDir}/high_probability_df.csv'):
        #     os.remove(f'./././Data/experiments7/{CurrentDir}/high_probability_df.csv')
        # high_probability_df2.to_csv(f'./././Data/experiments7/{CurrentDir}/high_probability_df.csv', index = False, encoding = 'utf-8')


        # トピック生起確率が閾値以上のトピックセグメントを再生時間が短い順
        # short_time_df = df[df['topic_probability'] >= TOPIC_THRESHOLD]
        # short_time_df = short_time_df.sort_values(by = 'play_time', ascending = True)
        # file_paths = short_time_df['file_path'].tolist()
        # stimes = short_time_df['stime'].tolist()
        # etimes = short_time_df['etime'].tolist()
        # for i in range(len(file_paths)):
        #     history.append(file_paths[i])
        #     audio = AudioSegment.from_file(file_paths[i])
        #     audio = audio[float(stimes[i])*1000 : float(etimes[i])*1000]
        #     audio.export(f'./././Django_Server/myproject/static/audio/short_time/segment_{i+1}.mp3', format = 'mp3')
        # short_time_df2 = pd.merge(short_time_df, topic_and_words, left_on = 'topics', right_on = 'topic', how = 'inner')
        # if os.path.exists('./././Data/test_data/final_presentation/short_time_df.csv'):
        #     os.remove('./././Data/test_data/final_presentation/short_time_df.csv')
        # short_time_df.to_csv('./././Data/test_data/final_presentation/short_time_df.csv', index = False, encoding = 'utf-8')
        # if os.path.exists(f'./././Data/experiments3/{sys.argv[1]}/short_time_df_r.csv'):
        #     os.remove(f'./././Data/experiments3/{sys.argv[1]}/short_time_df_r.csv')
        # short_time_df2.to_csv(f'./././Data/experiments3/{sys.argv[1]}/short_time_df_r.csv', index = False, encoding = 'utf-8')


        # 実験用
        # for current_dir, dirs, files in os.walk('./././Django_Server/myproject/static/audio/'):
        #     dir_name = os.path.basename(current_dir)
        #     if dir_name == 'high_probability':# or dir_name == 'short_time':
        #         save_folder = f'./././Data/experiments7/{CurrentDir}/{dir_name}/'
        #         os.makedirs(save_folder, exist_ok = True)
        #         for file in files:
        #             if file.endswith('.mp3'):
        #                 shutil.copy2(os.path.join(current_dir, file), save_folder)
                        # os.remove(os.path.join(current_dir, file))
        # high_probability_df2.to_csv('./././Data/test_data/final_presentation/high_probability_df.csv', index = False, encoding = 'utf-8')
        # short_time_df2.to_csv('./././Data/test_data/final_presentation/short_time_df.csv', index = False, encoding = 'utf-8')

        # print(f"{history}")

        # break