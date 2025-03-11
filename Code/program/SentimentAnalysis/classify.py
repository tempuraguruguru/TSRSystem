from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

emotion_names = ['Joy', 'Sadness', 'Anticipation', 'Surprise', 'Anger', 'Fear', 'Disgust', 'Trust']
emotion_names_jp = ['喜', '悲', '期待', '驚', '怒', '怖', '嫌', '信頼']

# モデルとトークナイザーをロード
model_path = "./././Code/Model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

def np_softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

def analyze_emotion(text, show_fig=False):
    model.eval() # 推論モードを有効化
    # 入力データ変換 + 推論
    tokens = tokenizer(text, truncation=True, return_tensors="pt")
    tokens.to(model.device)
    preds = model(**tokens)
    prob = np_softmax(preds.logits.cpu().detach().numpy()[0])
    out_dict = {n: p for n, p in zip(emotion_names_jp, prob)}
    return out_dict

def analynize_average(text):
    texts = text.split("。")
    vectors = [] # 各文の感性ベクトルを格納するためのリスト
    for line in texts:
        dic = analyze_emotion(line)
        vector = []
        for i in range(8):
            # print(dic)
            vector.append(dic[emotion_names_jp[i]])
        vectors.append(vector)
    emotion = {}
    for i in range(len(emotion_names_jp)):
        emotion_name = emotion_names_jp[i]
        total = 0
        for vector in vectors:
            total += vector[i]
        emotion[emotion_name] = total/len(vectors)
    return emotion

def analynize_next(text):
    texts = text.split("。")
    texts2 = []
    for k in range(len(texts)-1):
        l = texts[k] + texts[k+1]
        texts2.append(l)
    texts = texts2
    vectors = [] # 各文の感性ベクトルを格納するためのリスト
    for line in texts:
        dic = analyze_emotion(line)
        vector = []
        for i in range(8):
            # print(dic)
            vector.append(dic[emotion_names_jp[i]])
        vectors.append(vector)
    emotion = {}
    for i in range(len(emotion_names_jp)):
        emotion_name = emotion_names_jp[i]
        total = 0
        for vector in vectors:
            total += vector[i]
        emotion[emotion_name] = total/len(vectors)
    return emotion

def plot(T, E0, E1, E2, E3, E4, E5, E6, E7):
    plt.figure(figsize=(12, 6))
    plt.plot(T, E0, label='Joy')
    plt.plot(T, E1, label='Sadness')
    plt.plot(T, E2, label='Anticipation')
    plt.plot(T, E3, label='Surprise')
    plt.plot(T, E4, label='Anger')
    plt.plot(T, E5, label='Fear')
    plt.plot(T, E6, label='Disgust')
    plt.plot(T, E7, label='Trust')
    plt.legend()
    plt.grid()
    plt.show()

def emotion_plot(text):
    texts = text.split("。")
    h = 0
    T = []
    E0 = []
    E1 = []
    E2 = []
    E3 = []
    E4 = []
    E5 = []
    E6 = []
    E7 = []
    for line in texts:
        dic = analyze_emotion(line)
        T.append(h)
        E0.append(dic[emotion_names_jp[0]])
        E1.append(dic[emotion_names_jp[1]])
        E2.append(dic[emotion_names_jp[2]])
        E3.append(dic[emotion_names_jp[3]])
        E4.append(dic[emotion_names_jp[4]])
        E5.append(dic[emotion_names_jp[5]])
        E6.append(dic[emotion_names_jp[6]])
        E7.append(dic[emotion_names_jp[7]])
        h += 1
    # plot(T, E0, E1, E2, E3, E4, E5, E6, E7)
    plot(T[:20], E0[:20], E1[:20], E2[:20], E3[:20], E4[:20], E5[:20], E6[:20], E7[:20])

def emotion_next_plot(text):
    texts = text.split("。")
    h = 0
    T = []
    E0 = []
    E1 = []
    E2 = []
    E3 = []
    E4 = []
    E5 = []
    E6 = []
    E7 = []
    for i in range(1, len(texts)):
        pre = analyze_emotion(texts[i-1])
        current = analyze_emotion(texts[i])
        E = []
        for j in range(8):
            E.append(current[emotion_names_jp[j]] - pre[emotion_names_jp[j]])
        T.append(h)
        E0.append(E[0])
        E1.append(E[1])
        E2.append(E[2])
        E3.append(E[3])
        E4.append(E[4])
        E5.append(E[5])
        E6.append(E[6])
        E7.append(E[7])
        h += 1
    plot(T[:20], E0[:20], E1[:20], E2[:20], E3[:20], E4[:20], E5[:20], E6[:20], E7[:20])

if __name__ == "__main__":
    example = "わずかに寂しい"
    emo = analyze_emotion(example)
    print(emo)