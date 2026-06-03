import pandas as pd

df = pd.read_csv("landmarks.csv")
print(df.columns.tolist())
print(df.shape)