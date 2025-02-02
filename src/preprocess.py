import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

df_wine = pd.read_csv("data/wine.csv")

X_wine = df_wine.drop(columns=["target"])
y_wine = df_wine["target"]

# Normalizar as features
scaler = StandardScaler()
X_wine_scaled = scaler.fit_transform(X_wine)

# Criar novo DataFrame processado
df_wine_processed = pd.DataFrame(X_wine_scaled, columns=X_wine.columns)
df_wine_processed["target"] = y_wine  # Adicionar rótulo novamente

# Salvar dataset processado
df_wine_processed.to_csv("data/processed/wine_processed.csv", index=False)
print("Wine Dataset processado salvo em 'data/processed/wine_processed.csv'")

# Carregar dataset
df_forest = pd.read_csv("data/forest_cover.csv")

X_forest = df_forest.drop(columns=["Cover_Type"])
y_forest = df_forest["Cover_Type"]

# Normalizar as features
X_forest_scaled = scaler.fit_transform(X_forest)

# Criar novo DataFrame processado
df_forest_processed = pd.DataFrame(X_forest_scaled, columns=X_forest.columns)
df_forest_processed["Cover_Type"] = y_forest  # Adicionar rótulo novamente

# Salvar dataset processado
df_forest_processed.to_csv("data/processed/forest_cover_processed.csv", index=False)
print("Forest Cover Dataset processado salvo em 'data/processed/forest_cover_processed.csv'")
