import shutil
import os
import random

for mode in ["train", "val", "test"]:
    os.makedirs("normalPGD_reg_dataset/" + mode, exist_ok=True)

for i, t in zip(range(26), [6, 4, 5, 5, 3, 8, 5, 7, 1000, 9, 6, 9, 4, 8, 4, 5, 5, 6, 6, 4, 5, 6, 5, 6, 7, 4]):
    split = 0
    count = 0
    l = os.listdir("normalPGD_alphabet/progress/" + chr(i + 65))
    os.makedirs("normalPGD_reg_dataset/train/" + chr(i +65), exist_ok=True)
    os.makedirs("normalPGD_reg_dataset/val/" + chr(i +65), exist_ok=True)
    os.makedirs("normalPGD_reg_dataset/test/" + chr(i +65), exist_ok=True)
    for j in l:
        count += sum(os.path.isfile(os.path.join("normalPGD_alphabet/progress/" + chr(i + 65) + "/" + j, name)) for name in os.listdir("normalPGD_alphabet/progress/" + chr(i + 65) + "/" + j))
    if count < 1400:
        continue
    for k in range(1, 1401): 
        for j in l:
            if os.path.exists("normalPGD_alphabet/progress/" + chr(i + 65) + "/{}/{}.png".format(j, k)):
                split += 1
                if split <= 1000:
                    os.makedirs("normalPGD_reg_dataset/train/" + chr(i +65) + "/{}".format(1), exist_ok=True)
                    os.makedirs("normalPGD_reg_dataset/train/" + chr(i +65) + "/{}".format(0), exist_ok=True)
                    if int(j) >= t:
                        shutil.copy("normalPGD_alphabet/progress/" + chr(i + 65) + "/{}/{}.png".format(j, k), "normalPGD_reg_dataset/train/" + chr(i +65) + "/{}/{}.png".format(1, k))
                    else:
                        shutil.copy("normalPGD_alphabet/progress/" + chr(i + 65) + "/{}/{}.png".format(j, k), "normalPGD_reg_dataset/train/" + chr(i +65) + "/{}/{}.png".format(0, k))
                elif split > 1000 and split <= 1200:
                    os.makedirs("normalPGD_reg_dataset/val/" + chr(i +65) + "/{}".format(1), exist_ok=True)
                    os.makedirs("normalPGD_reg_dataset/val/" + chr(i +65) + "/{}".format(0), exist_ok=True)
                    if int(j) >= t:
                        shutil.copy("normalPGD_alphabet/progress/" + chr(i + 65) + "/{}/{}.png".format(j, k), "normalPGD_reg_dataset/val/" + chr(i +65) + "/{}/{}.png".format(1, k))
                    else:
                        shutil.copy("normalPGD_alphabet/progress/" + chr(i + 65) + "/{}/{}.png".format(j, k), "normalPGD_reg_dataset/val/" + chr(i +65) + "/{}/{}.png".format(0, k))
                elif split > 1200 and split <=1400:
                    os.makedirs("normalPGD_reg_dataset/test/" + chr(i +65) + "/{}".format(1), exist_ok=True)
                    os.makedirs("normalPGD_reg_dataset/test/" + chr(i +65) + "/{}".format(0), exist_ok=True)
                    if int(j) >= t:
                        shutil.copy("normalPGD_alphabet/progress/" + chr(i + 65) + "/{}/{}.png".format(j, k), "normalPGD_reg_dataset/test/" + chr(i +65) + "/{}/{}.png".format(1, k))
                    else:
                        shutil.copy("normalPGD_alphabet/progress/" + chr(i + 65) + "/{}/{}.png".format(j, k), "normalPGD_reg_dataset/test/" + chr(i +65) + "/{}/{}.png".format(0, k))



# for i in range(26):
#     img_num = 0
#     for j in range(1, 401):
#         if sum(os.path.isfile(os.path.join("normalPGD_alphabet/progress/" + chr(i + 65) + "/{}".format(j), name)) for name in os.listdir("normalPGD_alphabet/progress/" + chr(i + 65) + "/{}".format(j))) == 0:
#             break
#         elif sum(os.path.isfile(os.path.join("normalPGD_alphabet/progress/" + chr(i + 65) + "/{}".format(j), name)) for name in os.listdir("normalPGD_alphabet/progress/" + chr(i + 65) + "/{}".format(j))) == 1000:
#             continue
#         img_num += 1
#         os.makedirs("normalPGD_reg_dataset/" + chr(i + 65) +"/{}".format(sum(os.path.isfile(os.path.join("normalPGD_alphabet/progress/" + chr(i + 65) + "/{}".format(j), name)) for name in os.listdir("normalPGD_alphabet/progress/" + chr(i + 65) + "/{}".format(j)))), exist_ok=True)
#         shutil.copy("normalPGD_alphabet/org/" + chr(i + 65) + "/{}.png".format(j), "normalPGD_reg_dataset/" + chr(i + 65) + "/{}/{}.png".format(sum(os.path.isfile(os.path.join("normalPGD_alphabet/progress/" + chr(i + 65) + "/{}".format(j), name)) for name in os.listdir("normalPGD_alphabet/progress/" + chr(i + 65) + "/{}".format(j))), img_num))