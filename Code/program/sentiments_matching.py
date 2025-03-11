from sklearn.metrics.pairwise import cosine_similarity

def DicToList(dic):
    l = []
    for key, value in dic.items():
        l.append(value)
    return l

# 感情ベクトル = {"喜", "悲", "期待", "驚", "怒", "怖", "嫌", "信頼"}を、{"喜", "信頼", "怖", "驚", "悲", "嫌", "怒", "期待"}
def convert_setiment(svector):
    #svectorをdict型に修正
    sentiments = ["喜", "信頼", "怖", "驚", "悲", "嫌", "怒", "期待"]
    vector = {}
    for i in range(8):
        vector[sentiments[i]] = svector[sentiments[i]]
    return vector

def match(sentiments_user, sentiments_seg):
    sentiments_user = DicToList(convert_setiment(sentiments_user))
    sentiments_seg = DicToList(convert_setiment(sentiments_seg))

    user_nega = (sentiments_user[2] + sentiments_user[4] + sentiments_user[5] + sentiments_user[6])/4
    user_posi = (sentiments_user[0] + sentiments_user[1])/2
    user_polarity = "p" if user_posi > user_nega else "n"

    seg_nega = (sentiments_seg[2] + sentiments_seg[4] + sentiments_seg[5] + sentiments_seg[6])/4
    seg_posi = (sentiments_seg[0] + sentiments_seg[1])/2
    seg_polarity = "p" if seg_posi > seg_nega else "n"

    if (user_polarity == "p" and seg_polarity == "p"):
        # セグメントの与えるポジティブな感情値がユーザーのポジティブな感情値以上の場合
        if (sentiments_seg[0] >= sentiments_user[0]) or (sentiments_seg[1] >= sentiments_user[1]):
            cs = cosine_similarity([sentiments_user], [sentiments_seg])
    elif (user_polarity == "n" and seg_polarity == "p"):
        # ユーザーのポジティブとネガティブな感情値を逆転
        if (sentiments_seg[0] >= sentiments_user[0]) or (sentiments_seg[1] >= sentiments_user[1]):
            # 低いポジティブ値を逆転
            sentiments_user[0] = 1 - sentiments_user[0]
            sentiments_user[1] = 1 - sentiments_user[1]
            # 高いネガティブ値を逆転
            sentiments_user[2] = 0 #1 - sentiments_user[2]
            sentiments_user[4] = 0 #1 - sentiments_user[4]
            sentiments_user[5] = 0 #1 - sentiments_user[5]
            sentiments_user[6] = 0 #1 - sentiments_user[6]
            # ニュートラル値の処理
            sentiments_user[3] = 0
            sentiments_user[7] = 0
            for i in range(2, 8):
                sentiments_seg[i] = 0
            cs = cosine_similarity([sentiments_user], [sentiments_seg])
    else:
        cs = [[0]]
    return cs[0][0]

if __name__ == '__main__':
    example = {'喜': 0.00015195215, '悲': 0.9972882, '期待': 0.00020896211, '驚': 0.00045566674, '怒': 0.00014182813, '怖': 0.0006181228, '嫌': 0.0010567032, '信頼': 7.8520025e-05}
    sum = 0
    for key, value in example.items():
        sum += value
    print(sum)