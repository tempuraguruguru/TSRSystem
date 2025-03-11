from transformers import BertJapaneseTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import numpy as np
import torch
import time

tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
bert_model = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def embed(text): # 384次元
    word_embeddings = model.encode(text)
    return word_embeddings

def embed2(word): # 768次元
    input = tokenizer(word, return_tensors = "pt")
    outputs = bert_model(**input)
    last_hidden_state = outputs.last_hidden_state
    sample_vector = last_hidden_state[0][1].detach().numpy()
    return sample_vector

def make_list(str):
    list = []
    str = str[2:-2].split(',')
    for s in str:
        list.append(s.replace(' ', ''))
    return list

def make_dic(str):
    dic = {}
    str = str[2:-2].split(',')
    for s in str:
        s = s.split(':')
        dic[s[0].replace(' ', '')] = np.float64((s[1])[1:-1].replace(' ', ''))
    return dic

def dic_to_list(dic):
    res = []
    for key, value in dic.items():
        res.append(value)
    return res

if __name__ == '__main__':
    input = "音楽"
    test = "チケット"

    start = time.time()
    sample_vector = embed(input)
    test_vector = embed(test)
    cs = cosine_similarity([sample_vector], [test_vector])[0][0]
    end = time.time()
    print(f"{input}, {test}のcos類似度: {cs}")
    print(f"Processing time: {end - start}")


    strl = "['./././Data/test_data/mid_presentation/yamaguchiichiro/20240914/split_audio_72.wav','./././Data/test_data/mid_presentation/yamaguchiichiro/20240914/split_audio_72.wav',]"
    print(eval(strl))