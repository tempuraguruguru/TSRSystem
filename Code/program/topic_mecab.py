from Transcription import transcription
import playback_time

import MeCab
from gensim import corpora
from gensim.models.ldamodel import LdaModel
import pandas as pd
import re
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pylab as plt
from gensim.models import CoherenceModel
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt2
from collections import Counter
import seaborn as sns
from pydub import AudioSegment

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

# トピックセグメント分割におけるハイパーパラメータ
ROW_COUNT = 25
TOPIC_SIMILARITY = 0.8
# TOPIC_BREAKCOUNT_THRESHOLD = 10
# print(f"ROW_COUNT = {ROW_COUNT}, TOPIC_PROBABILITY_THRESHOLD = {TOPIC_PROBABILITY_THRESHOLD}, TOPIC_CONTINUATION_THRESHOLD = {TOPIC_CONTINUATION_THRESHOLD}")

stopwords = pd.read_csv(stopwords_path, header = None)[0].to_list()
lowercase_letters = [chr(i) for i in range(ord('a'), ord('z') + 1)]
uppercase_letters = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
hiragana_list = [chr(i) for i in range(ord('あ'), ord('ん') + 1)]
alphabets = lowercase_letters + uppercase_letters + hiragana_list

row_counts = []
topics_num_list = []

exist_nouns = []
exist_topics = []
coherence_scores = []
perplexity_list = []

def extract_nouns(text):
    if not isinstance(text, str):
        raise TypeError(f"Expected a string but got {type(text)}: {text}")

    parsed = mecab.parse(text)
    nouns = []
    for line in parsed.splitlines():
        if line == "EOS":
            break
        word, feature = line.split("\t")
        features = feature.split(",")
        # 名詞または複合名詞を抽出
        if features[0] == "名詞" and (features[1] == "一般" or features[1] == "固有名詞"):
            word = re.sub(r'\d', '', word)
            if len(word) == 1 and word.isalpha():
                continue
            nouns.append(word)
        nouns = [noun for noun in nouns if noun not in stopwords]
        nouns = [word for word in nouns if not (word.isdigit() or "ー" in word)]

    return nouns

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
            if (features2[0] == "助詞") or (features2[0] == "助動詞"):
                count += 1

        # 助詞・助動詞が含まれていた場合、名詞が含まれていない場合
        if (count >= 1) or (count_noun == 0):
            continue
        nouns2.append(noun)

    return nouns2

def analyze(documents, num_topics, num_words):
    texts = [extract_nouns(doc) for doc in documents]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    if len(texts[0]) == 0 or len(corpus[0]) == 0:
        return 0

    lda_model = LdaModel(corpus = corpus, id2word = dictionary, num_topics = num_topics, random_state = 42)
    topics = lda_model.print_topics(num_words = num_words)
    matches = re.findall(r'\"(.*?)\"', topics[0][1])
    return ', '.join(matches)

def Estimate_Word_Distribution(documents, num_topics, num_words):
    texts = [extract_nouns(doc) for doc in documents]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    if len(texts[0]) == 0 or len(corpus[0]) == 0:
        return 0

    lda_model = LdaModel(corpus = corpus, id2word = dictionary, num_topics = num_topics, random_state = 42)
    topics = lda_model.print_topics(num_words = num_words)
    matches = re.findall(r'\"(.*?)\"', topics[0][1])
    for topic in topics:
        print(topic)

def Estimate_Topic_Distribution(documents, num_topics, num_words):
    texts = [extract_nouns(doc) for doc in documents]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    if len(texts[0]) == 0 or len(corpus[0]) == 0:
        return 0

    lda_model = LdaModel(corpus = corpus, id2word = dictionary, num_topics = num_topics, random_state = 42)
    topics = lda_model.print_topics(num_words = num_words)
    matches = re.findall(r'\"(.*?)\"', topics[0][1])

    doc_topic_distributions = []
    for doc_bow in corpus:
        doc_topics = lda_model.get_document_topics(doc_bow)
        doc_topic_distributions.append(doc_topics)

    # 各文書のトピック分布を表示
    for i, doc_topics in enumerate(doc_topic_distributions):
        print(f"Document {i} topic distribution: {doc_topics}")
        for doc_topic in doc_topics:
            topic = lda_model.show_topic(doc_topic[0], topn = num_words)
            print(f"Topic {doc_topic[0]}: {topic}")
        print("\n")

    return topics, doc_topic_distributions

def calculate_entropy(distribution):
    # 確率の部分だけを抽出
    probabilities = np.array([prob for _, prob in distribution])
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def calculate_max_entropy(distribution):
    num_topics = len(distribution)
    return np.log2(num_topics)

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

def Estimate_Topic_Distribution_Overall(root, num_topics, num_words):
    global row_counts, exist_nouns, coherence_scores, perplexity_list, topics_num_list, exist_topics

    # 全ての発話音声データの情報を一つのデータフレームとして保持
    data = {'file_path': [], 'transcription': [], 'transcription_timestamp': [], 'time': []}
    for current_dir, dirs, files in os.walk(root):
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
    df.to_csv(f'./././Data/test_data/final_presentation/all_data_{num_topics}.csv', index = False, encoding = 'utf-8')
    fpaths = df['file_path'].tolist()
    documents = df['transcription'].tolist()
    documents_timestamp = df['transcription_timestamp'].tolist()

    # 各文書をN-gramにする
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
        if len(tdocs) - ROW_COUNT <= 0:
            doc = ''
            for l in range(len(tdocs)):
                docs = tdocs[l].split('::')
                if l == 0:
                    doc += docs[1]
                    start = eval(docs[0])[0]
                    n_grams_time.append(start)
            n_grams.append(doc)
        else:
            for j in range(len(tdocs) - ROW_COUNT + 1):
                doc = ''
                for k in range(ROW_COUNT):
                    docs = tdocs[j+k].split('::')
                    if k == 0:
                        doc += docs[1]
                        start = eval(docs[0])[0]
                        n_grams_time.append(start)
                    else:
                        doc += docs[1]
                n_grams.append(doc)

        # documentsをb-gramにしたリストをsubdocuments、n-gramしたdocumentsの先頭が現れる時間を格納したn_grams_timeをstartTimesに格納
        subdocuments.append(n_grams)
        startTimes.append(n_grams_time)

    # LDAモデルを構築するための文書整形
    flattened_subdocuments = [doc for sublist in subdocuments for doc in sublist]
    tokenized_documents = [extract_nouns2(doc) for doc in flattened_subdocuments]
    dictionary = corpora.Dictionary(tokenized_documents)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_documents]
    lda_model = LdaModel(
        corpus = corpus,
        id2word = dictionary,
        num_topics = num_topics, # トピック数
        random_state = 42,
        passes = 2, # 学習エポック数
        alpha = 'auto', # ハイパーパラメータ
    )

    # 分割文書に対してトピック分布を計算
    sub_texts = []
    for subdocs in subdocuments:
        sub_nouns = []
        for subdoc in subdocs: # subdocはn-gram
            sub_nouns.append(extract_nouns2(subdoc))
        sub_texts.append(sub_nouns)

    sub_corpuses = []
    for sub_text in sub_texts:
        sub_corpus = []
        for sub_nouns in sub_text:
            sub_corpus.append(dictionary.doc2bow(sub_nouns))
        sub_corpuses.append(sub_corpus)

    doc_topic_distributions = []
    for sub_corpus in sub_corpuses:
        sub_doc_topic_distributions = []
        for sub_doc_bows in sub_corpus:
            sub_doc_topics = lda_model.get_document_topics(sub_doc_bows, minimum_probability = 0)
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
        for j in range(len(sub_doc_topic_distributions) - ROW_COUNT):
            if len(sub_doc_topic_distributions[j+ROW_COUNT]) == 0:
                continue
            # 現在の窓のトピック情報
            current_topic_id = (max(sub_doc_topic_distributions[j], key = lambda x : x[1]))[0]
            current_topic_vec = np.array([prob for _, prob in sub_doc_topic_distributions[j]]).reshape(1, -1)
            current_topic_start = startTimes[i][j]

            # 次の窓のトピック情報
            next_topic_id = (max(sub_doc_topic_distributions[j+ROW_COUNT], key = lambda x : x[1]))[0]
            next_topic_vec = np.array([prob for _, prob in sub_doc_topic_distributions[j+ROW_COUNT]]).reshape(1, -1)
            next_topic_start = startTimes[i][j+ROW_COUNT]

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
    has_topics = set(has_topics)
    has_topics = list(has_topics)
    topic_num = 0

    for t in has_topics:
        topic_num += 1
        topics = lda_model.show_topic(t, topn=num_words)
        nouns = []
        probabilities = []
        for word, prob in topics:
            nouns.append(word)
            probabilities.append(prob)
        All_Topics_Info['topic'].append(t)
        All_Topics_Info['nouns'].append(nouns)
        All_Topics_Info['probabilities'].append(probabilities)

    word_distribution = pd.DataFrame(All_Topics_Info)
    word_distribution.to_csv(f'./././Data/test_data/final_presentation/word_distribution_{ROW_COUNT}_{num_topics}.csv', index = False, encoding = 'utf-8')

    # スコア関連計算^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    CoherenceScores = []
    Perplexity = []
    coherence_score_model = CoherenceModel(model = lda_model, corpus = corpus, dictionary = dictionary, coherence='c_v', texts = tokenized_documents)
    coherence_score = coherence_score_model.get_coherence()

    log_perplexity = lda_model.log_perplexity(corpus)
    perplexity = np.exp2(-log_perplexity)

    row_counts.append(ROW_COUNT)
    topics_num_list.append(num_topics) # 入力したトピック数
    exist_topics.append(topic_num) # トピックセグメントとして存在するトピック数

    exist_nouns.append(topic_num)
    coherence_scores.append(coherence_score)
    perplexity_list.append(perplexity)
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    df2 = pd.DataFrame(All_Segments_Info)
    df_merge = pd.merge(df, df2, on = 'file_path', how = 'inner')
    df_merge.to_csv("./././Data/test_data/final_presentation/df_merge.csv", index = False, encoding = "utf-8")

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
    topic_segment_data = {'file_path': [], 'transcription': [], 'transcription_timestamp': [], 'transcription_nouns': [], 'transcription_nouns_frequency': [], 'start': [], 'end': [], 'topic_distribution': [], 'entropy': [], 'max_entropy': [], 'topic': [], 'topic_probability': []}
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
            topic_distribution = lda_model.get_document_topics(utterance_bow, minimum_probability = 0.0)
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
            topic_segment_data['entropy'].append(calculate_entropy(topic_distribution2))
            topic_segment_data['max_entropy'].append(calculate_max_entropy(topic_distribution2))
            topic_segment_data['start'].append(start)
            topic_segment_data['end'].append(end)

            # for t in topic_distribution:
            #     if t[0] == topic:
            #         topic_segment_data['file_path'].append(file_path)
            #         topic_segment_data['transcription'].append(utterance)
            #         topic_segment_data['topic'].append(t[0])
            #         topic_segment_data['topic_probability'].append(t[1])
            #         # トピックセグメントの発話テキストからトピック分布を推定、最も生起確率が高いトピック
            #         topics_probabilities.append(t[1])

        # 1つのtranscriptionごとにトピック確率を追加
        topics_probabilitieses.append(topics_probabilities)

    data = {'file_path': file_paths, 'transcription': transcriptions, 'transcription_timestamp': transcriptions_timestamp,
            'time': times, 'topics': topicses_new, 'start_time': start_times_new, 'topic_probabilities': topics_probabilitieses}
    df = pd.DataFrame(data)
    df.to_csv(f'./././Data/test_data/final_presentation/topic_segments_info_{ROW_COUNT}_{num_topics}.csv', index = False, encoding = 'utf-8')

    # いろいろスコア
    # print(f"N-gram: {ROW_COUNT}")
    # print(f"トピック数: {num_topics}")
    # print(f"存在トピック数: {topic_num}")
    # print(f"perplexity: {perplexity}")
    # print(f"coherence score: {coherence_score}")
    # print(f"トピックセグメントの数: {segment_count}")
    # print()

    # トピックセグメントのcsv
    os.makedirs(f"./././Data/experiments6/decade/{ROW_COUNT}", exist_ok=True)
    segment_df = pd.DataFrame(topic_segment_data)
    ids = [id for id in range(len(segment_df))]
    ids = {'id': ids}
    ids = pd.DataFrame(ids)
    segment_df = pd.concat([segment_df, ids], axis = 1)
    # segment_df = segment_df[segment_df['entropy'] <= (segment_df['max_entropy'] * 0.80)]
    segment_df.to_csv('./././Data/test_data/final_presentation/segment_df.csv', index = False, encoding = 'utf-8')

    df = pd.merge(segment_df, word_distribution, on = 'topic', how = 'inner')
    df.to_csv(f'./././Data/test_data/final_presentation/segment_df_{ROW_COUNT}_{num_topics}.csv', index = False, encoding = 'utf-8')
    df_correct = df[['id', 'file_path', 'transcription', 'transcription_timestamp', 'transcription_nouns']]
    df_correct['select_user_nouns'] = np.nan
    df_correct.to_csv(f'./././Data/test_data/final_presentation/df_correct_{ROW_COUNT}_{num_topics}.csv', index = False, encoding = 'utf-8')
    df_correct.to_excel(f'./././Data/test_data/final_presentation/df_correct_{ROW_COUNT}_{num_topics}.xlsx')

    # トピックセグメントの時間が1分以上
    segment_df['time'] = segment_df['end'] - segment_df['start']
    segment_df = segment_df[segment_df['time'] >= 60]

    # サンプリング
    # 四分位数の計算
    min_value = segment_df['topic_probability'].min()
    Q1 = segment_df['topic_probability'].quantile(0.25)
    Q2 = segment_df['topic_probability'].quantile(0.50)
    Q3 = segment_df['topic_probability'].quantile(0.75)
    max_value = segment_df['topic_probability'].max()

    print(f"最小値: {min_value}")
    print(f"第一四分位数 (Q1): {Q1}")
    print(f"中央値 (Q2): {Q2}")
    print(f"第三四分位数 (Q3): {Q3}")
    print(f"最大値: {max_value}")

    # 範囲に基づくデータの選択とサンプリング
    min_to_Q1 = segment_df[segment_df['topic_probability'] < Q1].sample(25, random_state=42)
    Q1_to_Q2 = segment_df[(segment_df['topic_probability'] >= Q1) & (segment_df['topic_probability'] < Q2)].sample(25, random_state=42)
    Q2_to_Q3 = segment_df[(segment_df['topic_probability'] >= Q2) & (segment_df['topic_probability'] < Q3)].sample(25, random_state=42)
    Q3_to_max = segment_df[segment_df['topic_probability'] >= Q3].sample(25, random_state=42)

    select_df = pd.concat([min_to_Q1, Q1_to_Q2, Q2_to_Q3, Q3_to_max], ignore_index = True)
    select_df = pd.merge(select_df, word_distribution, on = 'topic', how = 'inner')
    new_columns = ['id', 'file_path', 'transcription', 'transcription_timestamp', 'transcription_nouns', 'transcription_nouns_frequency', 'start', 'end', 'topic_distribution', 'entropy', 'max_entropy', 'topic', 'topic_probability', 'time', 'nouns', 'probabilities']
    select_df = select_df[new_columns]
    select_df.to_csv(f'./././Data/test_data/final_presentation/sampling_{ROW_COUNT}_{num_topics}.csv', index = False, encoding = 'utf-8')

    # トピックセグメントの音声ファイルの作成
    os.makedirs('./././Data/test_data/topic_segments', exist_ok = True)
    # topic_segment_raw_ids = select_df['id'].tolist()
    # topic_segment_raw_file_paths = select_df['file_path'].tolist()
    # topic_segment_raw_start_times = select_df['start'].tolist()
    # topic_segment_raw_end_times = select_df['end'].tolist()
    # for rid, rfp, rst, red in zip(topic_segment_raw_ids, topic_segment_raw_file_paths, topic_segment_raw_start_times, topic_segment_raw_end_times):
    #     audio = AudioSegment.from_file(rfp)
    #     audio = audio[rst*1000 : red*1000]
    #     audio.export(f'./././Data/test_data/topic_segments/{rid}.mp3', format = 'mp3')

    # ユーザーに正解データを作成してもらうためのファイル
    select_df_user = select_df[['id', 'file_path', 'transcription', 'transcription_timestamp', 'transcription_nouns']]
    select_df_user['select_user_noun'] = np.nan
    select_df_user.to_csv(f'./././Data/test_data/final_presentation/sampling_user_{ROW_COUNT}_{num_topics}.csv', index = False, encoding = 'utf-8')
    select_df_user.to_excel(f'./././Data/test_data/final_presentation/sampling_user_{ROW_COUNT}_{num_topics}.xlsx', index = False)
    select_df_user.to_excel(f'./././Data/test_data/topic_segments/sampling_user_{ROW_COUNT}_{num_topics}.xlsx', index = False)


    # df_user = df[['file_path', 'transcription', 'topic', 'topic_probability', 'nouns']]

    # df_system = df
    # data_system = {'file_path': [], 'transcription': [], 'topic': [], 'topic_probability': [], 'nouns': [], 'probabilities': []}
    # file_paths = df_system['file_path'].tolist()
    # transcriptions = df_system['transcription'].tolist()
    # topics = df_system['topic'].tolist()
    # topics_probabilities = df_system['topic_probability'].tolist()
    # nounses = df_system['nouns'].tolist()
    # probabilitieses = df_system['probabilities'].tolist()
    # for fp, ts, t, tp, ns, ps in zip(file_paths, transcriptions, topics, topics_probabilities, nounses, probabilitieses):
    #     new_ns = []
    #     new_ps = []
    #     for noun, probability in zip(ns, ps):
    #         if probability > 1e-5:
    #             new_ns.append(noun)
    #             new_ps.append(probability)
    #     data_system['file_path'].append(fp)
    #     data_system['transcription'].append(ts)
    #     data_system['topic'].append(t)
    #     data_system['topic_probability'].append(tp)
    #     data_system['nouns'].append(new_ns)
    #     data_system['probabilities'].append(new_ps)
    # df_system = pd.DataFrame(data_system)

    # df.to_csv(f'./././Data/experiments6/decade/{ROW_COUNT}/正解_100.csv', index = False, encoding = 'utf-8')
    # df_user.to_csv(f'./././Data/experiments6/decade/{ROW_COUNT}/ユーザー_100.csv', index = False, encoding = 'utf-8')
    # df_system.to_csv(f'./././Data/experiments6/decade/{ROW_COUNT}/システム_100.csv', index = False, encoding = 'utf-8')

def plot_ngram_2score(X, Ys1, Ys2, labels, ylabel1, ylabel2):
    fig, ax1 = plt.subplots(figsize = (8, 5))
    colors = ["#FF0000", "#0000FF", "#00FF00", "#FFA500", "#800080", "#00FFFF", "#FF00FF", "#FFFF00", "#A52A2A", "#FFC0CB", "#808080", "#006400", "#FFD700"]
    exclude = []

    index = 0
    for y in Ys1:
        if labels[index] in exclude:
            index += 1
            continue
        y = [ele * 1e20 for ele in y]
        ax1.plot(X, y, label = f"{labels[index]}_Score", color = colors[index], marker = 'o')
        ax1.set_xlabel('Number of Topics')
        ax1.set_ylabel(ylabel1)
        ax1.tick_params(axis = 'y')
        ax1.grid()
        ax1.set_yscale('log')
        index += 1

    index = 0
    ax2 = ax1.twinx()
    for y in Ys2:
        if labels[index] in exclude:
            index += 1
            continue
        ax2.plot(X, y, label = f"{labels[index]}_CoherenceScore", color = colors[index], marker = 'x', linestyle = '--')
        ax2.set_ylabel(ylabel2)
        ax2.tick_params(axis = 'y')
        index += 1

    # ax1.legend() # 線のキャプションをつける
    ax1.set_xticks(X)
    plt.title(f"{ylabel1} & {ylabel2}")
    fig.tight_layout()
    plt.show()

def plot_ngram_2score_top5(X, Ys1, Ys2, labels, ylabel1, ylabel2):
    fig, ax1 = plt.subplots(figsize = (15, 7))
    colors = ["#FF0000", "#0000FF", "#00FF00", "#FFA500", "#800080", "#00FFFF", "#FF00FF", "#FFFF00", "#A52A2A", "#FFC0CB", "#808080", "#006400", "#FFD700"]

    # Ys1 の最大値でスコアをソートし、上位5つを取得
    scores = [(max(y), idx) for idx, y in enumerate(Ys1)]
    top5_indices = [idx for _, idx in sorted(scores, key=lambda x: x[0], reverse=True)[:5]]

    # ax1 に関連するプロット
    lines1 = []
    for idx in top5_indices:
        y = [ele * 1e20 for ele in Ys1[idx]]
        line, = ax1.plot(X, y, label=f"{labels[idx]}_Score", color=colors[idx % len(colors)], marker='o')
        lines1.append(line)
        ax1.set_xlabel('Number of Topics')
        ax1.set_ylabel(ylabel1)
        ax1.tick_params(axis='y')
        ax1.grid()
        ax1.set_yscale('log')

    # ax2 に関連するプロット
    ax2 = ax1.twinx()
    lines2 = []
    for idx in top5_indices:
        line, = ax2.plot(X, Ys2[idx], label=f"{labels[idx]}_CoherenceScore", color=colors[idx % len(colors)], marker='x', linestyle='--')
        lines2.append(line)
        ax2.set_ylabel(ylabel2)
        ax2.tick_params(axis='y')
        ax2.grid()

    # 両方の凡例を統合
    lines = lines1 + lines2
    labels_combined = [line.get_label() for line in lines]
    ax1.legend(lines, labels_combined, loc='upper left', bbox_to_anchor=(1.05, 1))

    # x 軸の設定とタイトル
    ax1.set_xticks(X)
    plt.title(f"Top 5 {ylabel1} & {ylabel2}")
    fig.tight_layout()
    plt.show()

def plot_ngram_score(X, Ys, labels, ylabel, toplabel):
    fig, ax = plt.subplots(figsize = (8, 5))
    colors = ["#FF0000", "#0000FF", "#00FF00", "#FFA500", "#800080", "#00FFFF", "#FF00FF", "#FFFF00", "#A52A2A", "#FFC0CB", "#808080", "#006400", "#FFD700"]
    exclude = []
    data = []

    index = 0
    for y in Ys:
        data.append(y)
        if labels[index] in exclude:
            index += 1
            continue
        y = [ele * 1e20 for ele in y]
        ax.plot(X, y, label = labels[index], color = colors[index])
        ax.set_xlabel('Number of Topics')
        ax.set_ylabel(ylabel)
        ax.tick_params(axis = 'y')
        ax.grid()
        ax.set_yscale('log')
        index += 1

    data_array = np.array(data)
    max_indices = np.argmax(data_array, axis = 0)
    max_labels = [labels[i] for i in max_indices]
    for col, label in zip(X, max_labels):
        print(f"Column {col} has maximum value at row: {label}")

    ax.legend()
    ax.set_xticks(X)
    plt.title(toplabel)
    fig.tight_layout()
    plt.show()

def plot_ngram_2score(X, Ys1, Ys2, labels, ylabel1, ylabel2):
    fig, ax1 = plt.subplots(figsize = (8, 5))
    colors = ["#FF0000", "#0000FF", "#00FF00", "#FFA500", "#800080", "#00FFFF", "#FF00FF", "#FFFF00", "#A52A2A", "#FFC0CB", "#808080", "#006400", "#FFD700"]
    exclude = []

    index = 0
    for y in Ys1:
        if labels[index] in exclude:
            index += 1
            continue
        y = [ele * 1e20 for ele in y]
        ax1.plot(X, y, label = f"{labels[index]}_{ylabel1}", color = colors[index], marker = 'o')
        ax1.set_xlabel('Number of Topics')
        ax1.set_ylabel(ylabel1)
        ax1.tick_params(axis = 'y')
        ax1.grid()
        ax1.set_yscale('log')
        index += 1

    index = 0
    ax2 = ax1.twinx()
    for y in Ys2:
        if labels[index] in exclude:
            index += 1
            continue
        ax2.plot(X, y, label = f"{labels[index]}_{ylabel2}", color = colors[index], marker = 'x', linestyle = '--')
        ax2.set_ylabel(ylabel2)
        ax2.tick_params(axis = 'y')
        index += 1

    # ax1.legend() # 線のキャプションをつける
    ax1.set_xticks(X)
    plt.title(f"{ylabel1} & {ylabel2}")
    fig.tight_layout()
    plt.show()

def plot_ngram_2score_top5(X, Ys1, Ys2, labels, ylabel1, ylabel2):
    fig, ax1 = plt.subplots(figsize = (15, 7))
    colors = ["#FF0000", "#0000FF", "#00FF00", "#FFA500", "#800080", "#00FFFF", "#FF00FF", "#FFFF00", "#A52A2A", "#FFC0CB", "#808080", "#006400", "#FFD700"]

    # Ys1 の最大値でスコアをソートし、上位5つを取得
    scores = [(max(y), idx) for idx, y in enumerate(Ys1)]
    top5_indices = [idx for _, idx in sorted(scores, key=lambda x: x[0], reverse=True)[:5]]

    # ax1 に関連するプロット
    lines1 = []
    for idx in top5_indices:
        y = [ele * 1 for ele in Ys1[idx]]
        line, = ax1.plot(X, y, label=f"{labels[idx]}_{ylabel1}", color=colors[idx % len(colors)], marker='o')
        lines1.append(line)
        ax1.set_xlabel('Number of Topics')
        ax1.set_ylabel(ylabel1)
        ax1.tick_params(axis='y')
        ax1.grid()
        ax1.set_yscale('log')

    # ax2 に関連するプロット
    ax2 = ax1.twinx()
    lines2 = []
    for idx in top5_indices:
        y = [ele * 1 for ele in Ys2[idx]]
        line, = ax2.plot(X, y, label=f"{labels[idx]}_{ylabel2}", color=colors[idx % len(colors)], marker='x', linestyle='--')
        lines2.append(line)
        ax2.set_ylabel(ylabel2)
        ax2.tick_params(axis='y')
        ax2.grid()

    # 両方の凡例を統合
    lines = lines1 + lines2
    labels_combined = [line.get_label() for line in lines]
    ax1.legend(lines, labels_combined, loc='upper left', bbox_to_anchor=(1.05, 1))

    # x 軸の設定とタイトル
    ax1.set_xticks(X)
    plt.title(f"Top 5 {ylabel1} & {ylabel2}")
    fig.tight_layout()
    plt.show()

def plot_ngram_score(X, Ys, labels, ylabel, toplabel):
    fig, ax = plt.subplots(figsize = (8, 5))
    colors = ["#FF0000", "#0000FF", "#00FF00", "#FFA500", "#800080", "#00FFFF", "#FF00FF", "#FFFF00", "#A52A2A", "#FFC0CB", "#808080", "#006400", "#FFD700"]
    exclude = []
    data = []

    index = 0
    for y in Ys:
        data.append(y)
        if labels[index] in exclude:
            index += 1
            continue
        y = [ele * 1 for ele in y]
        ax.plot(X, y, label = labels[index], color = colors[index])
        ax.set_xlabel('Number of Topics')
        ax.set_ylabel(ylabel)
        ax.tick_params(axis = 'y')
        ax.grid()
        ax.set_yscale('log')
        index += 1

    data_array = np.array(data)
    max_indices = np.argmax(data_array, axis = 0)
    max_labels = [labels[i] for i in max_indices]
    for col, label in zip(X, max_labels):
        print(f"Column {col} has maximum value at row: {label}")

    ax.legend()
    ax.set_xticks(X)
    plt.title(toplabel)
    fig.tight_layout()
    plt.show()



if __name__ == '__main__':
    test_path = ('./././Data/test_data/final_presentation/audrey/20240907/split_audio_15.wav')
    csv_file = '/'.join(test_path.split('/')[:-1]) + '/info.csv'

    # text = "好きなんだけどさ"
    # print(extract_nouns(text))
    # print(extract_nouns2(text))
    text = "音楽について聴きたい。特に作詞や作曲について聴きたい"
    nouns = extract_nouns2(text)
    print(nouns)

    text2 = "音楽について聴きたい。特にライブについて聴きたい"
    nouns2 = extract_nouns2(text2)
    print(nouns2)

    ROW_COUNT = 25
    num_topics = 200
    Estimate_Topic_Distribution_Overall('./././Data/test_data/final_presentation/', num_topics, 10)

    # tri_gram_tn = [50, 100, 150, 200]
    # tri_gram_etn = [50, 21, 9, 1]
    # tri_gram_p = [5.292521756446607e+16,  1.484983963982315e+34,  9.362214433049772e+49, 2.4018847765464195e+120]
    # tri_gram_cs = [0.5603337235937978, 0.47269924208406183, 0.4691389391105712, 0.467804196359114]
    # tri_gram_score = []
    # for p, cs in zip(tri_gram_p, tri_gram_cs):
    #     score = cs / p
    #     tri_gram_score.append(score)

    # fifth_gram_tn = [50, 100, 150, 200]
    # fifth_gram_etn = [50, 79, 20, 11]
    # fifth_gram_p = [201707075045143.72, 1.7111550696927105e+30, 2.317153869819191e+48, 3.4693982937463864e+64]
    # fifth_gram_cs = [0.5488076541635166, 0.36939561502824697, 0.32218704771280715, 0.31700213055672644]
    # fifth_gram_score = []
    # for p, cs in zip(fifth_gram_p, fifth_gram_cs):
    #     score = cs / p
    #     fifth_gram_score.append(score)

    # seventh_gram_tn = [50, 100, 150, 200]
    # seventh_gram_etn = [50, 100, 67, 28]
    # seventh_gram_p = [47846521312998.19, 8.436545647873174e+25, 5.210693444151099e+43, 1.413639320780789e+62]
    # seventh_gram_cs = [0.543888436618775, 0.4040586520318024, 0.2837893154063015, 0.2497257475729783]
    # seventh_gram_score = []
    # for p, cs in zip(seventh_gram_p, seventh_gram_cs):
    #     score = cs / p
    #     seventh_gram_score.append(score)

    # nineth_gram_tn = [50, 100, 150, 200]
    # nineth_gram_etn = [50, 100, 120, 53]
    # nineth_gram_p = [34821160660355.53, 3.8899644349935514e+24, 6.286831831766454e+40, 2.060247316821634e+57]
    # nineth_gram_cs = [0.5397149293848622, 0.4434689213330453, 0.35180096433757063, 0.30773769362483316]
    # nineth_gram_score = []
    # for p, cs in zip(nineth_gram_p, nineth_gram_cs):
    #     score = cs / p
    #     nineth_gram_score.append(score)

    # eleventh_gram_tn = [50, 100, 150, 200]
    # eleventh_gram_etn = [50, 100, 143, 103]
    # eleventh_gram_p = [31231635171242.242, 1.6215369394139614e+24, 5.556386939870381e+36, 3.339307475398505e+53]
    # eleventh_gram_cs = [0.541464698958703, 0.42830699860064037, 0.35619764655939046, 0.2770342541240703]
    # eleventh_gram_score = []
    # for p, cs in zip(eleventh_gram_p, eleventh_gram_cs):
    #     score = cs / p
    #     eleventh_gram_score.append(score)

    # thirteenth_gram_tn = [50, 100, 150, 200]
    # thirteenth_gram_etn = [50, 100, 149, 160]
    # thirteenth_gram_p = [27427202650012.586, 1.0484309652473477e+24, 1.919339414385143e+35, 6.943442939035111e+49]
    # thirteenth_gram_cs = [0.5224633928226141, 0.42751486710370723, 0.3682875106973854, 0.3029185362972094]
    # thirteenth_gram_score = []
    # for p, cs in zip(thirteenth_gram_p, thirteenth_gram_cs):
    #     score = cs / p
    #     thirteenth_gram_score.append(score)

    # fifteenth_gram_tn = [50, 100, 150, 200]
    # fifteenth_gram_etn = [50, 100, 149, 189]
    # fifteenth_gram_p = [24240233211809.81, 6.532366655528776e+23, 5.5211322191144e+34, 7.250753634286758e+46]
    # fifteenth_gram_cs = [0.4973699432900046, 0.3910855764110377, 0.34088953154811263, 0.31161150901167817]
    # fifteenth_gram_score = []
    # for p, cs in zip(fifteenth_gram_p, fifteenth_gram_cs):
    #     score = cs / p
    #     fifteenth_gram_score.append(score)

    # seventeenth_gram_tn = [50, 100, 150, 200]
    # seventeenth_gram_etn = [50, 100, 149, 189]
    # seventeenth_gram_p = [22700656322495.285, 5.970908436300293e+23, 2.2882397323501114e+34, 6.667952884638911e+45]
    # seventeenth_gram_cs = [0.4842648252348392, 0.40324145587608656, 0.3701472398510226, 0.3350007087551114]
    # seventeenth_gram_score = []
    # for p, cs in zip(seventeenth_gram_p, seventeenth_gram_cs):
    #     score = cs / p
    #     seventeenth_gram_score.append(score)

    # nineteenth_gram_tn = [50, 100, 150, 200]
    # nineteenth_gram_etn = [50, 100, 148, 193]
    # nineteenth_gram_p = [20724382725265.18, 5.0840766132152155e+23, 1.3722675180470308e+34, 9.350604845149909e+44]
    # nineteenth_gram_cs = [0.4884342862626416, 0.41061166545693895, 0.3612081543542624, 0.3232105219944688]
    # nineteenth_gram_score = []
    # for p, cs in zip(nineteenth_gram_p, nineteenth_gram_cs):
    #     score = cs / p
    #     nineteenth_gram_score.append(score)

    # twenty_first_gram_tn = [50, 100, 150, 200]
    # twenty_first_gram_etn = [50, 100, 150, 195]
    # twenty_first_gram_p = [20967131572318.477,  4.424223725399572e+23, 1.3751299163156526e+34, 7.696545661651735e+44]
    # twenty_first_gram_cs = [0.4667367883655346, 0.3829476866954479, 0.3634455778403577, 0.34832411844708866]
    # twenty_first_gram_score = []
    # for p, cs in zip(twenty_first_gram_p, twenty_first_gram_cs):
    #     score = cs / p
    #     twenty_first_gram_score.append(score)

    # twenty_third_gram_tn = [50, 100, 150, 200]
    # twenty_third_gram_etn = [50, 100, 149, 199]
    # twenty_third_gram_p = [18784244433790.957, 3.8891700105505275e+23, 9.938684267032315e+33, 2.7451153547399394e+44]
    # twenty_third_gram_cs = [0.48641775279235133, 0.4152326348368382, 0.3719841850722733, 0.34412836809903474]
    # twenty_third_gram_score = []
    # for p, cs in zip(twenty_third_gram_p, twenty_third_gram_cs):
    #     score = cs / p
    #     twenty_third_gram_score.append(score)

    # twenty_fifth_gram_tn = [50, 100, 150, 200]
    # twenty_fifth_gram_etn = [50, 100, 149, 193]
    # twenty_fifth_gram_p = [15633774945960.693, 2.6724701659699272e+23, 5.586343351609163e+33, 1.1506151524190418e+44]
    # twenty_fifth_gram_cs = [0.507759646983791, 0.4531768932686136, 0.40186103369783505, 0.37467684613395263]
    # twenty_fifth_gram_score = []
    # for p, cs in zip(twenty_fifth_gram_p, twenty_fifth_gram_cs):
    #     score = cs / p
    #     twenty_fifth_gram_score.append(score)

    # twenty_seventh_gram_tn = [50, 100, 150, 200]
    # twenty_seventh_gram_etn = [50, 100, 150, 196]
    # twenty_seventh_gram_p = [15567444700636.139, 2.4988698976495246e+23, 5.0986312205247507e+33, 9.643885424956931e+43]
    # twenty_seventh_gram_cs = [0.4805783796133804, 0.39952874477140216, 0.37143184391935674, 0.3524314435465412]
    # twenty_seventh_gram_score = []
    # for p, cs in zip(twenty_seventh_gram_p, twenty_seventh_gram_cs):
    #     score = cs / p
    #     twenty_seventh_gram_score.append(score)

    # X = [50, 100, 150, 200]
    # Perplexities = [tri_gram_p, fifth_gram_p, seventh_gram_p, nineth_gram_p, eleventh_gram_p, thirteenth_gram_p, fifteenth_gram_p,
    #         seventeenth_gram_p, nineteenth_gram_p, twenty_first_gram_p, twenty_third_gram_p, twenty_fifth_gram_p, twenty_seventh_gram_p]
    # CoherenceScores = [tri_gram_cs, fifth_gram_cs, seventh_gram_cs, nineth_gram_cs, eleventh_gram_cs, thirteenth_gram_cs, fifteenth_gram_cs,
    #         seventeenth_gram_cs, nineteenth_gram_cs, twenty_first_gram_cs, twenty_third_gram_cs, twenty_fifth_gram_cs, twenty_seventh_gram_cs]
    # Scores = [tri_gram_score, fifth_gram_score, seventh_gram_score, nineth_gram_score, eleventh_gram_score, thirteenth_gram_score, fifteenth_gram_score,
    #         seventeenth_gram_score, nineteenth_gram_score, twenty_first_gram_score, twenty_third_gram_score, twenty_fifth_gram_score, twenty_seventh_gram_score]
    # ExistTopics = [tri_gram_etn, fifth_gram_etn, seventh_gram_etn, nineth_gram_etn, eleventh_gram_etn, thirteenth_gram_etn, fifteenth_gram_etn,
    #         seventeenth_gram_etn, nineteenth_gram_etn, twenty_first_gram_etn, twenty_third_gram_etn, twenty_fifth_gram_etn, twenty_seventh_gram_etn]
    # labels = ['3-gram', '5-gram', '7-gram', '9-gram', '11-gram', '13-gram', '15-gram', '17-gram', '19-gram', '21-gram', '23-gram', '25-gram', '27-gram']


    # # plot_ngram_score(X, Perplexities, 'Perplexity', 'Perplexity')
    # # plot_ngram_score(X, CoherenceScores, 'Coherence Score', 'Coherence Score')
    # plot_ngram_score(X, Scores, labels, 'Score', 'Score (Coherence Score / Perplexity)')
    # # plot_ngram_score(X, ExistTopics, 'ExistTopics', 'Number of Exist Topics')

    # # plot_ngram_2score(X, Scores, CoherenceScores, labels, 'Score', 'Coherence Score')a
    # plot_ngram_2score_top5(X, Scores, CoherenceScores, labels, 'Score', 'Coherence Score')
