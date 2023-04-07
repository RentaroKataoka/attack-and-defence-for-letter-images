import pandas as pd
from tqdm import tqdm
import shutil

df = pd.read_csv("GoogleFonts/google_fonts_drop_none.csv", index_col=0)
length = len(df)
df_list = df.values.tolist()

for i in tqdm(
    range(length),
    total = length,
    leave = False):
    if df_list[i][3] == "test":
        try:
            path1 = "GoogleFonts/google_fonts_ttf/"+df_list[i][0]+".ttf"
            # path2 = "font/binary/train/"+df_list[i][0]+""
            shutil.copyfile(path1, "GoogleFonts/test_ttf/"+df_list[i][0]+".ttf")
        except FileNotFoundError:
            pass

