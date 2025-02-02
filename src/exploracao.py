import pandas as pd

df = pd.read_csv("data/forest_cover.csv")
df_wine = pd.read_csv("data/wine.csv")
print(f"Dataset Wine carregado: {df_wine.shape[0]} linhas, {df_wine.shape[1]} colunas")
print(df_wine.head())

print(f"Dataset Forest carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")
print(df.head())