from transformers import BertForNextSentencePrediction, BertJapaneseTokenizer
import torch

# 東北大の日本語学習済みBERTモデルをダウンロード
nsp_model = BertForNextSentencePrediction.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
bert_tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
nsp_model.eval()

# promptとnext_sentenceを繋げたときに自然な文章になるかどうか
prompt = 'ケーキは'
next_sentence = 'ショートケーキの一種だ。'

# 入力のpromptとnext_sentenceをトークン化
input_tensor = bert_tokenizer(prompt, next_sentence, return_tensors='pt')
# print(input_tensor)
# list(bert_tokenizer.get_vocab().items())[:5]

output = nsp_model(**input_tensor)
# print(output)

print(torch.argmax(output.logits))
if(torch.argmax(output.logits) == 0):
    print(True)