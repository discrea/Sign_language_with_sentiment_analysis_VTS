import pandas as pd

data_2017 = pd.read_excel("./KETI-2017-SL-Annotation-v2_1.xlsx", usecols=[0, 1, 2, 3, 4, 5, 6])
data_2018 = pd.read_excel(
    "./KETI-2018-SL-Annotation-v1.xlsx", sheet_name=1, usecols=[0, 1, 2, 3, 4, 6, 7]
)

data_2017["파일명"] = data_2017["파일명"].apply(lambda x: x.split(".")[0])

data = pd.concat([data_2017, data_2018]).dropna()
print(data.shape, data.columns)
print(data.head())

data.to_csv("KETI-Annotation.csv", index=False)
