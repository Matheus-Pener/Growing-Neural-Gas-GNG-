import pandas as pd
import os
from urllib.request import urlretrieve
from zipfile import ZipFile

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
file_path = "data/covtype.data.gz"
csv_path = "data/forest_cover.csv"

if not os.path.exists(file_path):
    print("📥 Baixando o dataset UCI Covertype...")
    urlretrieve(url, file_path)
    print("Download concluído!")

#Conversão para csv
columns = [
    "Elevation", "Aspect", "Slope",
    "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon",
    "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points"
] + [f"Wilderness_Area_{i}" for i in range(4)] + [f"Soil_Type_{i}" for i in range(40)] + ["Cover_Type"]

df = pd.read_csv(file_path, header=None, names=columns)

# Salvar em CSV
df.to_csv(csv_path, index=False)
print(f"Dataset salvo em {csv_path}")
print(f"{df.shape[0]} linhas e {df.shape[1]} colunas")
