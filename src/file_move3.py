import shutil
import os
import glob
# dirname1 = "normalPGD_googlefonts/org"
# dirname2 = "GoogleFonts_reg"
# for i in range(26):
#     os.makedirs(dirname2 + "/" + chr(i + 65), exist_ok=True)
#     for j in glob.glob(dirname1 + "/" + chr(i + 65) + "/*"):
#         shutil.copy(j, dirname2 + "/" + chr(i + 65))


a = "test"
for i in range(26):
    os.makedirs("GoogleFonts/all/" + chr(i + 65), exist_ok=True)
    files = glob.glob("GoogleFonts/" + a +  "_font/**/" + chr(i + 65) + ".png")
    for index, file in enumerate(files):
        shutil.copy(file, "GoogleFonts/all/" + chr(i + 65) + "/{}.png".format(index + 1347))