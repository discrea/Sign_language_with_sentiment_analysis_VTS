import re
import os
import shutil
from glob import glob

import pandas as pd

words = pd.read_csv("./words.csv")
words = set(words["word"].values)

annotation = pd.read_csv("./KETI-Annotation.csv")

raw_dir = "./8381~9000(영상)/"
data_dir = "./data/"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

filenames = glob(raw_dir + "*.*")
filename_re = re.compile("KETI_[A-Z]+_[0-9]+")
filetype_re = re.compile("[.]+[a-zA-Z0-9]{3}")

count = 0
for filename in filenames:
    filecode = filename_re.search(filename).group(0)
    filetype = filetype_re.search(filename).group(0)
    target_label = annotation.loc[annotation["파일명"] == filecode, "한국어"].values[0]
    if target_label in words:
        shutil.move(filename, data_dir + filecode + filetype)
        count += 1
shutil.rmtree(raw_dir)

print(f"total files: {len(filenames)}")
print(f"target files: {count}")
