import matplotlib.pyplot as plt
import os
import cv2
import random
import numpy as np

alphabet = "A"
# mode = "../"
mode = ""

random.seed(42)
DATADIR = mode + "normalPGD_alphabet/progress/" + alphabet
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
i = 0
feature_list = []
label_list = []
for feature, label in training_data:
    feature = np.array([feature])
    # if i < 23:
    #     feature_list.append(feature)
    #     i += 1
    #     label_list.append(label)
    #     continue
    feature_list.append(feature)
    feature_list = np.array(feature_list)
    label_list.append(label)
    label_list = np.array(label_list)
    X_train.append(feature_list)
    y_train.append(label_list)
    i = 0
    feature_list = []
    label_list = []
# numpy配列に変換
X_train = np.array(X_train)
y_train = np.array(y_train)



DATADIR = mode + "normalPGD_reg_dataset/val/" + alphabet
CATEGORIES = os.listdir(DATADIR)
# DATADIR = "../val/val"
# CATEGORIES = [chr(c + 65) for c in range(26)]
IMG_SIZE = 64
training_data = []
create_training_data()
random.shuffle(training_data)  # データをシャッフル
X_val = []  # 画像データ
y_val = []  # ラベル情報
# データセット作成
i = 0
feature_list = []
label_list = []
for feature, label in training_data:
    feature = np.array([feature])
    # if i < 35:
    #     feature_list.append(feature)
    #     i += 1
    #     label_list.append(label)
    #     continue
    feature_list.append(feature)
    feature_list = np.array(feature_list)
    label_list.append(label)
    label_list = np.array(label_list)
    X_val.append(feature_list)
    y_val.append(label_list)
    i = 0
    feature_list = []
    label_list = []
# numpy配列に変換
X_val = np.array(X_val)
y_val = np.array(y_val)


DATADIR = mode + "normalPGD_reg_dataset/test/" + alphabet
CATEGORIES = os.listdir(DATADIR)
# DATADIR = "../test/test"
# CATEGORIES = [chr(c + 65) for c in range(26)]
IMG_SIZE = 64
training_data = []
create_training_data()
random.shuffle(training_data)  # データをシャッフル
X_test = []  # 画像データ
y_test = []  # ラベル情報
# データセット作成
i = 0
feature_list = []
label_list = []
for feature, label in training_data:
    feature = np.array([feature])
    # if i < 40:
    #     feature_list.append(feature)
    #     i += 1
    #     label_list.append(label)
    #     continue
    feature_list.append(feature)
    feature_list = np.array(feature_list)
    label_list.append(label)
    label_list = np.array(label_list)
    X_test.append(feature_list)
    y_test.append(label_list)
    i = 0
    feature_list = []
    label_list = []
# numpy配列に変換
X_test = np.array(X_test)
y_test = np.array(y_test)