import pandas as pd
from sklearn.datasets import load_wine
import os

wine = load_wine()
df_wine = pd.DataFrame(wine.data, columns=wine.feature_names)
df_wine["target"] = wine.target  

# Salvar em CSV
csv_path = "data/wine.csv"
df_wine.to_csv(csv_path, index=False)

print(f" Wine dataset salvo em {csv_path}")
print(f"🔹 {df_wine.shape[0]} linhas e {df_wine.shape[1]} colunas")
