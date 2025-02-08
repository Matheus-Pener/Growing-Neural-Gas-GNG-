import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 🔹 Carregar o Dataset Processado
dataset_path = "data/processed/forest_cover_processed.csv"  # Ajuste o caminho se necessário
df = pd.read_csv(dataset_path)

# 🔹 Remover a coluna 'Cover_Type' (target) e padronizar os dados
X = df.drop(columns=["Cover_Type"])  
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🔹 Aplicar PCA
num_components = 7  # Definir o número de componentes principais
pca = PCA(n_components=num_components)
X_pca = pca.fit_transform(X_scaled)

# 🔹 Variância Explicada por cada Componente Principal
explained_variance = pca.explained_variance_ratio_
explained_variance_cumulative = np.cumsum(explained_variance)

# 🔹 Gerar o gráfico da variância explicada
plt.figure(figsize=(8, 5))
plt.bar(range(1, num_components + 1), explained_variance, alpha=0.6, label="Variância Individual", color="red")
plt.plot(range(1, num_components + 1), explained_variance_cumulative, marker="o", linestyle="dashed", label="Variância Acumulada", color="blue")
plt.xlabel("Número de Componentes Principais")
plt.ylabel("Proporção da Variância Explicada")
plt.title("Variância Explicada pelos Componentes Principais")
plt.legend()
plt.grid()
plt.show()

# 🔹 Exibir a variância explicada pelos 7 principais componentes
print("Variância explicada pelos componentes principais:", explained_variance)
