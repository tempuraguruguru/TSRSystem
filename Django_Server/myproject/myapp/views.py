from django.shortcuts import render
from django.http import HttpResponse
import subprocess
import os

from django.views.decorators.csrf import csrf_exempt
import json

import pandas as pd
import re

# Create your views here.
# def index(request):
#     return render(request, 'index2.html')

idx = 0
history = []
sub_history = []
high_probability_paths = []
short_time_paths = []
input_value = None

high_path = None
short_path = None
data_high = {'path': [], 'related keyword?': []}

def index(request):
    global history, sub_history, high_probability_paths, short_time_paths, data_high, input_value, high_path, short_path
    audio_path = None
    high_probability_path = None
    short_time_path = None
    phistory = None
    rec_s = None

    if os.name == 'posix':  # macOSやLinuxの場合
        high_probability_root = '/Users/takuno125m/Documents/Research/Django_Server/myproject/static/audio/high_probability'
        short_time_root = '/Users/takuno125m/Documents/Research/Django_Server/myproject/static/audio/short_time'
    elif os.name == 'nt':  # Windowsの場合
        high_probability_root = r'C:\Users\takun\Documents\laboratory\Django_Server\myproject\static\audio\high_probability'
        short_time_root = r'C:\Users\takun\Documents\laboratory\Django_Server\myproject\static\audio\short_time'

    if "yes" in request.POST:
        print("yes")
        data_high["path"].append(high_path)
        data_high["related keyword?"].append("Yes")
    elif "no" in request.POST:
        print("no")
        data_high["path"].append(high_path)
        data_high["related keyword?"].append("No")

    # current dir: /Users/takuno125m/Documents/Research/Django_Server/myproject
    if request.method == "POST":
        global history, sub_history, idx
        idx += 1

        if "next" in request.POST:
            # 推薦するトピックセグメントがまだ残っている場合
            print("next!!!")
            if len(high_probability_paths) != 0:
                high_probability_path = high_probability_paths.pop()
            if len(short_time_paths) != 0:
                short_time_path = short_time_paths.pop()

            # カレントディレクトリに対するhigh_probability_pathの相対パスを取得
            # print("high_probability_path:", high_probability_path)  # デバッグ用
            # print("Current working directory:", os.getcwd())  # デバッグ用
            print()
            high_probability_path = os.path.relpath(high_probability_path, os.getcwd())
            sub_history.append(high_probability_path)

            # 番組履歴
            phistory = {}
            for program_name in sub_history:
                names = program_name.split('/')[6:]
                pname = '/'.join(names)
                phistory[pname] = program_name

            high_path = high_probability_path

            return render(request, "index.html", {"high_probability_path": high_probability_path, "short_time_path": short_time_path, "history": history, "programs": phistory})

        input_value = request.POST.get("input_value")

        # historyをstringに変換
        rec_s = "["
        for item in history:
            rec_s += "'" + item + "'" + ","
        rec_s += "]"

        try:
            print("first!!")
            # wavファイルとトピック区間を取得
            result = subprocess.run(
                ["python", "../../Code/program/main.py", input_value, rec_s, f"{idx}"],
                capture_output = True,
                text = True,
                check = True
            )
            audio_path = result.stdout.strip()
            # print("Audio path:", audio_path)  # デバッグ用

            audio_paths = audio_path.split("\n")[-1]
            history = eval(audio_paths)

        except subprocess.CalledProcessError:
            audio_path = None

        # トピックの生起確率が高い順
        for current_dir, dirs, files in os.walk('./static/audio/high_probability/'):
            for file in files:
                # print(os.path.join(current_dir, file))
                high_probability_paths.append(os.path.join(current_dir, file))
        # high_probability_paths = high_probability_paths[:-1]
        high_probability_paths = sorted(high_probability_paths, key=lambda x: int(re.search(r'(\d+)', x).group()), reverse=True)

        # 生起確率が閾値以上で再生時間が短い順
        for current_dir, dirs, files in os.walk('./static/audio/short_time/'):
            for file in files:
                short_time_paths.append(os.path.join(current_dir, file))
        # short_time_paths = short_time_paths[:-1]
        short_time_paths = sorted(short_time_paths, key=lambda x: int(re.search(r'(\d+)', x).group()), reverse=True)

        # print("High probability paths:", high_probability_paths)
        # print("Short time paths:", short_time_paths)

        if len(high_probability_paths) != 0:
            high_probability_path = high_probability_paths.pop()
        if len(short_time_paths) != 0:
            short_time_path = short_time_paths.pop()

        # print(high_probability_path)
        # print(os.getcwd())
        # print()
        high_probability_path = os.path.relpath(high_probability_path, os.getcwd())
        sub_history.append(high_probability_path)
        phistory = {}
        for program_name in sub_history:
            names = program_name.split('/')[6:]
            pname = '/'.join(names)
            phistory[pname] = program_name
        # for k, v in phistory.items():
        #     print(k, v)
        idx += 1
        df = pd.DataFrame(data_high)
        df.to_csv(f'../../Data/experiments3/{input_value}/recommend_and_evaluate.csv', index = False, encoding = 'utf-8')

        high_path = high_probability_path


    return render(request, "index.html", {"high_probability_path": high_probability_path, "short_time_path": short_time_path, "history": history, "programs": phistory})


def process_topic(request):
    if request.method == 'POST':
        print("こっちも呼び出されるよ！")
        topic = request.POST.get('topic')
        return HttpResponse(f'あなたが入力したトピックは: {topic}')
    return HttpResponse('無効なリクエスト')


@csrf_exempt
def close_tab(request):
    global data_high, input_value
    if request.method == "POST":
        data = json.loads(request.body)
        if data.get("action") == "tab_closed":
            # タブが閉じられたときの処理
            df = pd.DataFrame(data_high)
            df.to_csv(f'../../Data/experiments3/{input_value}.csv', index = False, encoding = 'utf-8')
            print("タブが閉じられました")
    return HttpResponse("OK")