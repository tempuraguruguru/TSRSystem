import pandas as pd

adjusted_df = pd.read_csv("./././Data/test_data/final_presentation/df_adjusted/topic_segments_df_new.csv")
raw_df = pd.read_csv("./././Data/test_data/final_presentation/df_raw/topic_segments_df_new.csv")

dfs = [raw_df, adjusted_df]
for df in dfs:
    represent_matching_degrees = df["represent_matching_degree"].tolist()
    represent_matching_degrees = [ele for ele in represent_matching_degrees if ele == 0]
    average_matching_degrees = df["average_matching_degree"].tolist()
    average_matching_degrees = [ele for ele in average_matching_degrees if ele == 0]
    print(f"代表ベクトル同士のコサイン類似度の最小値: {len(represent_matching_degrees)}")
    print(f"平均ベクトル同士のコサイン類似度の最小値: {len(average_matching_degrees)}")
    print()