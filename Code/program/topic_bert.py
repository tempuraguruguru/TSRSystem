from bertopic import BERTopic
from Transcription import transcription
from hdbscan import HDBSCAN

model = BERTopic(embedding_model = "paraphrase-multilingual-MiniLM-L12-v2")

hoshino_test01 = './././Data/test_data/topic/hoshinogen01.wav'
hoshino_test02 = './././Data/test_data/topic/hoshinogen02.wav'
hoshino_test03 = './././Data/test_data/topic/hoshinogen03.wav'
hoshino_test04 = './././Data/test_data/topic/hoshinogen04.wav'
hoshino_test05 = './././Data/test_data/topic/hoshinogen05.wav'
hoshino_test06 = './././Data/test_data/topic/hoshinogen06.wav'
hoshino_test07 = './././Data/test_data/topic/hoshinogen07.wav'
paths = [hoshino_test01, hoshino_test02, hoshino_test03, hoshino_test04, hoshino_test05, hoshino_test06, hoshino_test07]
csv_file = '/'.join(hoshino_test01.split('/')[:-1]) + '/info.csv'

news_data: list[str] = []
for path in paths:
    text = transcription.transcribe_faster(path, 'small', csv_file)
    text = text.replace('\n', '')
    news_data.append(text)
print(len(news_data))

topics, probs = model.fit_transform(news_data)

model.visualize_barchart()