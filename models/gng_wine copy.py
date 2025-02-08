import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

# 🔹 Passo 1: Carregar o conjunto de dados
wine = load_wine()
wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)

# 🔹 Passo 2: Selecionar as 5 variáveis mais correlacionadas
selected_features = ['flavanoids', 'od280/od315_of_diluted_wines', 'proline', 'color_intensity', 'total_phenols']
data_selected = wine_df[selected_features]

# 🔹 Passo 3: Normalizar os dados
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data_selected)

# 🔹 Passo 4: Implementação do Growing Neural Gas (GNG)
class GrowingNeuralGas:
    def __init__(self, max_nodes=2000, max_age=90, learning_rate=0.03, neighbor_lr=0.005,   
                 error_decay=0.999, insertion_threshold=20, error_reduction_factor=0.5):
        self.graph = nx.Graph()
        self.max_nodes = max_nodes
        self.max_age = max_age
        self.learning_rate = learning_rate
        self.neighbor_lr = neighbor_lr
        self.error_decay = error_decay
        self.insertion_threshold = insertion_threshold
        self.error_reduction_factor = error_reduction_factor
        self.error = {}

    def initialize(self, data):
        idx = np.random.choice(len(data), 2, replace=False)
        self.graph.add_node(0, weight=data[idx[0]])
        self.graph.add_node(1, weight=data[idx[1]])
        self.graph.add_edge(0, 1, age=0)
        self.error[0] = 0
        self.error[1] = 0

    def find_bmu(self, x):
        distances = {node: distance.euclidean(x, self.graph.nodes[node]['weight'])
                     for node in self.graph.nodes}
        return sorted(distances, key=distances.get)[:2]

    def update_weights(self, bmu, x):
        self.graph.nodes[bmu]['weight'] += self.learning_rate * (x - self.graph.nodes[bmu]['weight'])

    def update_weights_neighbors(self, bmu, x):
        for neighbor in self.graph.neighbors(bmu):
            self.graph.nodes[neighbor]['weight'] += self.neighbor_lr * (x - self.graph.nodes[neighbor]['weight'])

    def insert_node(self):
        if len(self.graph.nodes) >= self.max_nodes:
            return
        q = max(self.error, key=self.error.get)
        neighbors_q = list(self.graph.neighbors(q))
        if not neighbors_q:
            return
        f = max(neighbors_q, key=lambda n: self.error[n])
        
        new_node = max(self.graph.nodes) + 1
        new_weight = (self.graph.nodes[q]['weight'] + self.graph.nodes[f]['weight']) / 2
        
        self.graph.add_node(new_node, weight=new_weight)
        self.graph.add_edge(new_node, q, age=0)
        self.graph.add_edge(new_node, f, age=0)
        if self.graph.has_edge(q, f):
            self.graph.remove_edge(q, f)
        
        self.error[new_node] = (self.error[q] + self.error[f]) / 2
        self.error[q] *= self.error_reduction_factor
        self.error[f] *= self.error_reduction_factor

    def train(self, data, epochs=100000):
        self.initialize(data)
        for epoch in range(epochs):
            x = data[np.random.randint(0, len(data))]
            bmu1, bmu2 = self.find_bmu(x)

            if not self.graph.has_edge(bmu1, bmu2):
                self.graph.add_edge(bmu1, bmu2, age=0)
            else:
                self.graph.edges[bmu1, bmu2]['age'] = 0

            dist_sq = distance.euclidean(x, self.graph.nodes[bmu1]['weight']) ** 2
            self.error[bmu1] += dist_sq

            self.update_weights(bmu1, x)
            self.update_weights_neighbors(bmu1, x)

            for edge in list(self.graph.edges):
                self.graph.edges[edge]['age'] += 1
                if self.graph.edges[edge]['age'] > self.max_age:
                    self.graph.remove_edge(*edge)

            if epoch % self.insertion_threshold == 0 and epoch != 0:
                self.insert_node()

            for n in self.error:
                self.error[n] *= self.error_decay

    def plot(self):
        plt.figure(figsize=(8, 6))
        for node in self.graph.nodes:
            weight = self.graph.nodes[node]['weight']
            plt.scatter(weight[0], weight[1], c='blue', s=50)
        for edge in self.graph.edges:
            n1, n2 = edge
            w1, w2 = self.graph.nodes[n1]['weight'], self.graph.nodes[n2]['weight']
            plt.plot([w1[0], w2[0]], [w1[1], w2[1]], 'k-', alpha=1.0, linewidth=1.5)
        plt.xlabel("Feature 1 (PCA ou Variável Normalizada)")
        plt.ylabel("Feature 2 (PCA ou Variável Normalizada)")
        plt.title("Clusters formados pelo GNG")
        plt.show()

# 🔹 Passo 5: Treinar o GNG na base reduzida
gng = GrowingNeuralGas(
    max_nodes=2000, max_age=90, learning_rate=0.03, neighbor_lr=0.005,
    error_decay=0.999, insertion_threshold=20, error_reduction_factor=0.5
)
gng.train(data_normalized, epochs=100000)
gng.plot()
print(f"🔥 Total de neurônios na rede após ajustes: {len(gng.graph.nodes)}")
