import pandas as pd
import os


def srash(path):
    # CSVファイルの読み込み
    df = pd.read_csv(path)

    # OSの確認
    if os.name == 'posix':  # macOSやLinuxの場合
        df['file_path'] = df['file_path'].str.replace('\\', '/')
    elif os.name == 'nt':  # Windowsの場合
        df['file_path'] = df['file_path'].str.replace('/', '\\')

    # 修正後のCSVファイルを保存
    df.to_csv(path, index=False)

if __name__ == '__main__':
    path = './././Data/test_data/mid_presentation/radiowave/info.csv'
    srash(path)