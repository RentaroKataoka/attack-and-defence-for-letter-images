import matplotlib.pyplot as plt
import os
import cv2
import random
import numpy as np


random.seed(42)
DATADIR = "../all_data"
CATEGORIES = os.listdir(DATADIR)
# DATADIR = "../train/train"
# CATEGORIES = [chr(c + 65) for c in range(26)]
IMG_SIZE = 64
training_data = []

def create_training_data():
    for class_num, category in enumerate(CATEGORIES):
        path = os.path.join(DATADIR, category)
        for image_name in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, image_name), cv2.IMREAD_GRAYSCALE)  # 画像読み込み
                img_resize_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # 画像のリサイズ
                img_resize_array = (img_resize_array >= 127) * 2 - 1
                training_data.append([img_resize_array, class_num])  # 画像データ、ラベル情報を追加

            except Exception as e:
                pass
create_training_data()
random.shuffle(training_data)  # データをシャッフル
X_train = []  # 画像データ
y_train = []  # ラベル情報
# データセット作成
for index, (feature, label) in enumerate(training_data):
    if (index % 16) == 0:
        feature_list = []
        label_list = []
    feature = np.array([feature])
    feature_list.append(feature)
    label_list.append(label)
    if (index % 16) == 15:
        feature_list = np.array(feature_list)
        label_list = np.array(label_list, dtype=np.float64)
        X_train.append(feature_list)
        y_train.append(label_list)
# numpy配列に変換
X_train = np.array(X_train)
y_train = np.array(y_train)