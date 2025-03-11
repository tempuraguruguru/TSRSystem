import Code.program.Transcription.transcription as transcription
from transformers import pipeline
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("CUDA is available. Using GPU.", torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')
    print("CUDA is not available. Using CPU")

# 使用するモデルを指定して、トークナイザとモデルを読み込む
# checkpoint = 'cl-tohoku/bert-base-japanese-char-v3'
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

nlp_base = pipeline('fill-mask', model='cl-tohoku/bert-base-japanese-char-v3', device = "mps") # device = "cuda"
nlp_large = pipeline('fill-mask', model='cl-tohoku/bert-large-japanese-char-v2', device = "mps") # device = "cuda"

def insert_char_to_text(i, char, text):
    l = list(text)
    l.insert(i, char)
    inserted_text = ''.join(l)
    return inserted_text

def text_correction(text, thresh, model, chars_count):
    text = text.replace(' ', '')  # なぜか元の文章に半角スペースが入っていたので除外
    nlp = model
    punctuations = ['、', '。', '?', '？', '!', '！']

    i = 0
    corrected_text = text
    while i < len(corrected_text):
        i += 1
        if corrected_text[i-1] in punctuations: continue

        masked_text = insert_char_to_text(i, nlp.tokenizer.mask_token, corrected_text)
        pre_text = masked_text.split("。")[-1].split(nlp.tokenizer.mask_token)[0][-chars_count:]
        post_text = masked_text.split(nlp.tokenizer.mask_token)[1][:chars_count]
        res = nlp(f'{pre_text}{nlp.tokenizer.mask_token}{post_text}')[0]

        if res['token_str'] not in punctuations: continue
        if res['score'] < thresh: continue

        punctuation = res['token_str'] if res['token_str'] not in ['?', '？'] else '。'
        corrected_text = insert_char_to_text(i, punctuation, corrected_text)
    return corrected_text


def insert_priod(audio_path, model_level):
    if (model_level != "tiny") and (model_level != "base") and (model_level != "small") and (model_level != "medium") and (model_level != "large"):
        return 0
    raw_text = transcription.transcribe_github(audio_path, model_level)
    text = text_correction(raw_text, 0.6, nlp_base, 50)
    return text

if __name__ == "__main__":
    test_m1_01 = './Data/test_data/radiowave#1-01.wav'
    models_name = ["tiny", "base", "small", "medium"] # largeは計算時間が長くなるので省略
    for name in models_name:
        start_time = time.time()
        text = insert_priod(test_m1_01, name)
        end_time = time.time()
        print(f"model: {name}\n文字起こしにかかった時間: {end_time - start_time}\n文字起こしの結果:\n{text}")
        time.sleep(30)


# 参考サイト
# PodcastをWhisperで文字起こしして、BERTで句読点抜きの文章に句読点を付与する（その２）: https://qiita.com/SoySoySoyB/items/8bbc6f698a82804acb4c